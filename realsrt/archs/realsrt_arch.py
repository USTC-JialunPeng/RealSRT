import math
import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import nn, einsum

from basicsr.utils.registry import ARCH_REGISTRY
from .vqgan_arch import VQGAN, Encoder, ResidualBlock, AttnBlock


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def gaussian_weights(size, var, device):
    midpoint = (size - 1) / 2  # -1 because index goes from 0 to size - 1
    x_probs = [np.exp(-(x-midpoint)*(x-midpoint)/(size*size)/(2*var)) / np.sqrt(2*np.pi*var) for x in range(size)]
    y_probs = [np.exp(-(y-midpoint)*(y-midpoint)/(size*size)/(2*var)) / np.sqrt(2*np.pi*var) for y in range(size)]
    weights = np.outer(y_probs, x_probs)
    weights = torch.tensor(weights, device=device).unsqueeze(0).unsqueeze(1)
    return weights

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# layers

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        scale = 8
    ):
        super().__init__()
        self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        assert not (exists(context) ^ self.cross_attend)

        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            context_mask = F.pad(context_mask, (1, 0), value = True)

            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        seq_len,
        enc_depth,
        dec_depth,
        embed_dim,
        n_embed,
        vae_embed
    ):
        super().__init__()

        self.dim_out = dim_out

        # token embeddings
        self.token_emb = nn.Embedding(n_embed + 1, embed_dim)
        self.token_emb.weight[:-1, :].data.copy_(vae_embed.weight[:-1, :].data.cuda().detach())
        self.token_emb.weight.requires_grad = False

        # mask token
        if self.dim_out != 1:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
            torch.nn.init.normal_(self.mask_token, std=.02)

        # positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_emb.shape[-1], int(seq_len**.5))
        self.pos_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # transformer encoder
        self.enc_in = nn.Linear(embed_dim, dim)
        self.enc_blocks = nn.ModuleList([])
        for _ in range(enc_depth):
            self.enc_blocks.append(nn.ModuleList([
                Attention(dim, dim_head = 64, heads = 8),
                FeedForward(dim)
            ]))
        self.enc_norm = LayerNorm(dim)

        # transformer decoder
        self.dec_in = nn.Linear(embed_dim, dim)
        self.dec_blocks = nn.ModuleList([])
        for _ in range(dec_depth):
            self.dec_blocks.append(nn.ModuleList([
                Attention(dim, dim_head = 64, heads = 8),
                Attention(dim, dim_head = 64, heads = 8, cross_attend = True),
                FeedForward(dim)
            ]))
        self.dec_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, dim_out, bias = False)

    def forward(
        self,
        x,
        lr_ids,
        mask = None,
        labels = None,
        ignore_index = -1
    ):

        # encoder
        enc = self.token_emb(lr_ids)
        enc = self.enc_in(enc)
        enc = enc + self.pos_emb

        for attn, ff in self.enc_blocks:
            enc = enc + attn(enc)
            enc = enc + ff(enc)
        enc = self.enc_norm(enc)

        # decoder
        dec = self.token_emb(x)
        dec = self.dec_in(dec)
        if self.dim_out != 1 and mask is not None:
            mask_tokens = self.mask_token.repeat(dec.shape[0], dec.shape[1], 1)
            dec = torch.where(mask.unsqueeze(-1), mask_tokens, dec)
        dec = dec + self.pos_emb

        for attn, cross_attn, ff in self.dec_blocks:
            dec = dec + attn(dec)
            dec = dec + cross_attn(dec, context = enc)
            dec = dec + ff(dec)
        dec = self.dec_norm(dec)

        logits = self.to_logits(dec)

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = ignore_index)

        return loss, logits

# noise schedules

def cubic_schedule(t):
    return 1 - t ** 3

def square_schedule(t):
    return 1 - t ** 2

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def linear_schedule(t):
    return 1 - t

def square_root_schedule(t):
    return 1 - torch.sqrt(t)

# super-resolution classes

@ARCH_REGISTRY.register()
class RealSRT(nn.Module):
    def __init__(
        self,
        use_critic = False,
        use_ccm = False,
        vae_pretrain_path = None,
        generator_pretrain_path = None,
        critic_pretrain_path = None
    ):
        super().__init__()
        self.use_critic = use_critic
        self.use_ccm = use_ccm

        self.vae = VQGAN(
                    quant_depth = 1,
                    embed_dim = 256,
                    n_embed = 512)

        if vae_pretrain_path:
            self.vae.load_state_dict(torch.load(vae_pretrain_path, map_location=lambda storage, loc: storage), strict=True)
            for param in self.vae.parameters():
                param.requires_grad = False

        self.generator = Transformer(
                    dim = 512,
                    dim_out = 512,
                    seq_len = 1024,
                    enc_depth = 4,
                    dec_depth = 16,
                    embed_dim = 256,
                    n_embed = 512,
                    vae_embed = self.vae.quantizer.codebooks[0])

        if generator_pretrain_path:
            self.load_state_dict(torch.load(generator_pretrain_path, map_location=lambda storage, loc: storage)['params_ema'], strict=True)
            for param in self.parameters():
                param.requires_grad = False

        if self.use_critic:
            self.critic = Transformer(
                    dim = 512,
                    dim_out = 1,
                    seq_len = 1024,
                    enc_depth = 4,
                    dec_depth = 12,
                    embed_dim = 256,
                    n_embed = 512,
                    vae_embed = self.vae.quantizer.codebooks[0])

        if critic_pretrain_path:
            self.load_state_dict(torch.load(critic_pretrain_path, map_location=lambda storage, loc: storage)['params_ema'], strict=True)
            for param in self.parameters():
                param.requires_grad = False

        if self.use_ccm:
            self.trainable_copy_encoder = Encoder(3, 256)
            for param, param_copy in zip(self.vae.encoder.parameters(), self.trainable_copy_encoder.parameters()):
                param_copy.data.copy_(param.data)  # initialize

            self.fuse_res1 = ResidualBlock(512, 512)
            self.fuse_attn = AttnBlock(512)
            self.fuse_res2 = ResidualBlock(512, 512)

            self.zero_out = zero_module(nn.Conv2d(512, 256, 1, 1, 0))

        self.noise_schedule = square_root_schedule
        self.mask_id = 512

    def forward(
        self,
        hr,
        lr,
        ignore_index = -1
    ):
       
        # freeze the codebook
        self.vae.eval()

        # tokenize
        hr_ids = self.vae.get_codes(hr).squeeze(-1)

        lr_upsample = F.interpolate(lr, scale_factor=4, mode='bicubic')
        lr_ids = self.vae.get_codes(lr_upsample).squeeze(-1)

        # get some basic variables
        batch, device = lr_ids.shape[0], lr_ids.device
        h, w = lr_ids.shape[1:]
        seq_len = h * w

        hr_ids = rearrange(hr_ids, 'b ... -> b (...)')
        lr_ids = rearrange(lr_ids, 'b ... -> b (...)')

        # prepare mask
        rand_time = uniform((batch,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        batch_randperm = torch.rand((batch, seq_len), device = device).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        # prepare input and labels
        x = torch.where(mask, self.mask_id, hr_ids)
        labels = torch.where(mask, hr_ids, ignore_index)

        # cross entropy loss for token generator
        ce_loss, logits = self.generator(
            x,
            lr_ids,
            mask = mask,
            labels = labels,
            ignore_index = ignore_index
        )

        if not self.use_critic:
            return ce_loss

        # binary cross entropy loss for token critic
        else:
            _, pred_ids = torch.max(logits, dim=-1)
            pred_ids = torch.where(mask, pred_ids, x)

            critic_labels = mask.float()

            bce_loss, _ = self.critic(
                pred_ids,
                lr_ids,
                labels = critic_labels
            )

            return bce_loss

    def forward_with_ccm(
        self,
        lr,
        timesteps = 4
    ):
        # freeze the codebook
        self.vae.eval()

        size = 64

        # tokenize
        lr_upsample = F.interpolate(lr, scale_factor=4, mode='bicubic')
        lr_ids = self.vae.get_codes(lr_upsample).squeeze(-1)

        # multiple prediction
        device = lr_ids.device
        ids = torch.full(lr_ids.shape, self.mask_id, dtype = torch.long, device = device)
        for step in range(timesteps):
            x = rearrange(ids, 'b ... -> b (...)')
            lr_ids = rearrange(lr_ids, 'b ... -> b (...)')
            mask = x == self.mask_id

            logits = self.generator(x, lr_ids, mask)
            _, pred_ids = torch.max(logits, dim=-1)

            pred_ids = torch.where(mask, pred_ids, x)

            pred_scores = self.critic(pred_ids, lr_ids) # critic logits
            pred_scores = pred_scores.squeeze(-1)

            if (step + 1) < timesteps:
                mask_prob = self.noise_schedule(torch.tensor([(step + 1) / timesteps], device = device))
                num_token_masked = int((mask_prob * (size//2)**2).item())

                masked_indices = pred_scores.topk(num_token_masked, dim = -1).indices
                pred_ids = pred_ids.scatter(1, masked_indices, self.mask_id)

            pred_ids = rearrange(pred_ids, 'b (h w) -> b h w', h = size//2, w = size//2)
            pred_scores = rearrange(pred_scores, 'b (h w) -> b h w', h = size//2, w = size//2)

            ids = pred_ids

        # decoding with conditional controlling module
        quant_feat = self.vae.quantizer.embed_partial_code(ids.unsqueeze(-1), 0).permute(0, 3, 1, 2).contiguous()
        h = self.vae.post_quant_conv(quant_feat)

        residual = torch.cat([h, self.trainable_copy_encoder(lr_upsample)], dim=1)
        residual = self.fuse_res1(residual)
        residual = self.fuse_attn(residual)
        residual = self.fuse_res2(residual)

        h = h + self.zero_out(residual)

        for block in self.vae.decoder.model:
            h = block(h)

        out = h

        return out
