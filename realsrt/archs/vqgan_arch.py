import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class VQEmbedding(nn.Embedding):
    """
        VQ embedding module with ema update.
    """

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99, restart_unused_codes=True, eps=1e-5):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed) # padding just for avoiding a bug

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # (B, h, w, n_embed)
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # (B, h, w, n_embed)
        embed_idxs = distances.argmin(dim=-1)

        return embed_idxs
    
    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):

        n_embed, embed_dim = self.weight.shape[0]-1, self.weight.shape[-1]
        
        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        
        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors)
                              )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)
        
        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]
            
            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)
        
            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1-usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)
        
        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.

    Arguments:
        quant_depth (int): the quantization depth
        embed_dim (int): the dimension of embeddings
        n_embed (int): the number of embeddings (i.e., the size of codebook)
        decay (float): weight decay for exponential updating
        restart_unused_codes (bool): If True, it randomly assigns a feature vector in the curruent batch
            as the new embedding of unused codes in training. (default: True)
    """

    def __init__(self,
                 quant_depth,
                 embed_dim,
                 n_embed,
                 decay=0.99,
                 restart_unused_codes=True
                 ):
        super().__init__()

        self.quant_depth = quant_depth

        codebooks = [VQEmbedding(n_embed, 
                                 embed_dim, 
                                 decay=decay, 
                                 restart_unused_codes=restart_unused_codes
                                 ) for i in range(quant_depth)]
        self.codebooks = nn.ModuleList(codebooks)

    def quantize(self, x):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.

        Arguments:
            x (Tensor): bottleneck feature maps to quantize.

        Returns:
            quant_list (list): list of sequentially aggregated and quantized feature maps by codebooks.
            codes (LongTensor): codewords index, corresponding to quants.

        Shape:
            - x: (B, h, w, embed_dim)
            - quant_list[i]: (B, h, w, embed_dim)
            - codes: (B, h, w, d)
        """
        B, h, w, embed_dim = x.shape

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.quant_depth):
            quant, code = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))
        
        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x):
        quant_list, codes = self.quantize(x)

        commitment_loss = self.compute_commitment_loss(x, quant_list)
        quants_trunc = quant_list[-1]
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, codes
    
    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []
        
        for idx, quant in enumerate(quant_list):
            partial_loss = (x-quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)
        
        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    @torch.no_grad()
    def embed_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Decode the input codes, using [0, 1, ..., code_idx] codebooks.

        Arguments:
            code (Tensor): codes of input image
            code_idx (int): the index of the last selected codebook for decoding

        Returns:
            embeds (Tensor): quantized feature map
        """

        assert code_idx < code.shape[-1]
        
        B, h, w, _ = code.shape
        
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]
            
        if decode_type == 'select':
            embeds = embeds[code_idx].view(B, h, w, -1)
        elif decode_type == 'add':
            embeds = torch.cat(embeds[:code_idx+1], dim=-2).sum(-2)
        else:
            raise NotImplementedError(f"{decode_type} is not implemented in partial decoding")

        return embeds


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.channel_up(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class GroupNorm(nn.Module):
    def __init__(self, in_channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Encoder(nn.Module):
    def __init__(self, image_channels, embed_dim):
        super(Encoder, self).__init__()
        channels = [128, 128, 256, 256]
        num_res_blocks = 2
        layers = [nn.Conv2d(image_channels, channels[0], 3, 1, 1)]
        in_channels = channels[0]
        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != len(channels) - 1:
                layers.append(DownSampleBlock(in_channels))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], embed_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, image_channels, embed_dim):
        super(Decoder, self).__init__()
        channels = [128, 128, 256, 256]
        num_res_blocks = 2
        layers = [nn.Conv2d(embed_dim, channels[-1], 3, 1, 1)]
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        in_channels = channels[-1]
        for i in reversed(range(len(channels))):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
        layers.append(GroupNorm(channels[0]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[0], image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@ARCH_REGISTRY.register()
class VQGAN(nn.Module):
    """
        Vector Quantized Variational Autoencoder. This networks includes a encoder which maps an
        input image to a discrete latent space, and a decoder to maps the latent map back to the input domain
    """
    def __init__(
        self,
        quant_depth=1,
        embed_dim=256,
        n_embed=512,
        decay=0.99,
        latent_loss_weight=0.25
    ):
        """
        Arguments:
            quant_depth (int): the quantization depth
            embed_dim (int): the dimension of embeddings
            n_embed (int): the number of embeddings (i.e., the size of codebook)
            decay (float): weight decay for exponential updating
            latent_loss_weight (float): weight of commitment loss
        """
        super().__init__()

        self.encoder = Encoder(3, embed_dim)
        self.quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.quantizer = RQBottleneck(quant_depth, embed_dim, n_embed, decay)
        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.decoder = Decoder(3, embed_dim)
        self.latent_loss_weight = latent_loss_weight

    def forward(self, xs):
        z_e = self.encode(xs)
        z_q, quant_loss, code = self.quantizer(z_e)
        out = self.decode(z_q)
        return out, quant_loss, code

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.quant_conv(z_e).permute(0, 2, 3, 1).contiguous()
        return z_e

    def decode(self, z_q):
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = self.post_quant_conv(z_q)
        out = self.decoder(z_q)
        return out

    def get_recon_imgs(self, xs_real, xs_recon):

        # xs_real = xs_real * 0.5 + 0.5
        # xs_recon = xs_recon * 0.5 + 0.5
        xs_recon = torch.clamp(xs_recon, 0, 1)

        return xs_real, xs_recon

    def compute_loss(self, out, quant_loss, code, xs=None, valid=False):

        loss_recon = F.mse_loss(out, xs, reduction='mean')

        loss_latent = quant_loss

        if valid:
            loss_recon = loss_recon * xs.shape[0]
            loss_latent = loss_latent * xs.shape[0]

        loss_total = loss_recon + self.latent_loss_weight * loss_latent

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': loss_latent,
            'codes': [code]
        }

    def get_last_layer(self):
        return self.decoder.model[-1].weight

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.encode(xs)
        _, _, code = self.quantizer(z_e)
        return code

    @torch.no_grad()
    def decode_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Use partial codebooks and decode the codebook features.
        If decode_type == 'select', the (code_idx)-th codebook features are decoded.
        If decode_type == 'add', the [0,1,...,code_idx]-th codebook features are added and decoded.
        """
        z_q = self.quantizer.embed_partial_code(code, code_idx, decode_type)
        decoded = self.decode(z_q)
        return decoded

    @torch.no_grad()
    def forward_partial_code(self, xs, code_idx, decode_type='select'):
        r"""
        Reconstuct an input using partial codebooks.
        """
        code = self.get_codes(xs)
        out = self.decode_partial_code(code, code_idx, decode_type)
        return out