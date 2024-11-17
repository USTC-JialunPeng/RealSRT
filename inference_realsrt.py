import os
import cv2
import glob
import time
import argparse
import numpy as np
from einops import rearrange

import torch
import torch.nn.functional as F

from realsrt.archs.realsrt_arch import RealSRT

# prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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

def gaussian_weights(size, var, device):
    midpoint = (size - 1) / 2  # -1 because index goes from 0 to size - 1
    x_probs = [np.exp(-(x-midpoint)*(x-midpoint)/(size*size)/(2*var)) / np.sqrt(2*np.pi*var) for x in range(size)]
    y_probs = [np.exp(-(y-midpoint)*(y-midpoint)/(size*size)/(2*var)) / np.sqrt(2*np.pi*var) for y in range(size)]
    weights = np.outer(y_probs, x_probs)
    weights = torch.tensor(weights, device=device).unsqueeze(0).unsqueeze(1)
    return weights

def main():
    """Inference demo for RealSRT.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input', help='Input folder')
    parser.add_argument('--output', type=str, default='output', help='Output folder')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    parser.add_argument('--stride', type=int, default=32, help='Stride of generation')
    parser.add_argument('--timesteps', type=int, default=4, help='Timesteps of generation')
    parser.add_argument('--max_minibatch', type=int, default=32, help='Limiting the maximum minibatch to save GPU memory')

    args = parser.parse_args()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = RealSRT(use_critic=True, use_ccm=True)
    
    model_path = args.model_path
    assert model_path is not None, 'model path is required'
    loadnet = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(loadnet['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    # inference
    os.makedirs(args.output, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.input, '*')))

    size = 64
    stride = args.stride
    assert stride % 2 == 0

    start_time = time.perf_counter()
    for count, path in enumerate(paths):
        img_name, extension = os.path.splitext(os.path.basename(path))
        print('Testing', count + 1, img_name)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        # pad image to fit the stride
        lr = img
        h_lr, w_lr = lr.shape[2:]
        h_lr_pad = (h_lr - size + stride - 1) // stride * stride + size
        w_lr_pad = (w_lr - size + stride - 1) // stride * stride + size
        pad_up = (h_lr_pad - h_lr) // 2
        pad_down = (h_lr_pad - h_lr) - pad_up
        pad_left = (w_lr_pad - w_lr) // 2
        pad_right = (w_lr_pad - w_lr) - pad_left

        lr_pad = F.pad(lr, (pad_left, pad_right, pad_up, pad_down), 'reflect')
        
        # crop the low-resolution image into patches
        lr_crop_list = []
        i = 0
        while (i+size) <= h_lr_pad:
            j = 0
            while (j+size) <= w_lr_pad:
                lr_crop_list.append(lr_pad[:, :, i:i+size, j:j+size])
                j += stride
            i += stride
        lr_crop = torch.cat(lr_crop_list, dim=0)

        # minibatch
        m = args.max_minibatch
        n = lr_crop.shape[0]
        
        start = 0
        out_crop_list = []
        while start < n:
            end = start + m
            if end >= n:
                end = n

            with torch.no_grad():
                out = model.forward_with_ccm(lr_crop[start:end], args.timesteps)
                out = adaptive_instance_normalization(out, lr_crop[start:end]) # color correction

            out_crop_list.append(out)
            start = end

        out_crop = torch.cat(out_crop_list, dim=0)

        # aggregation
        out_img = torch.zeros(1, 3, h_lr_pad*4, w_lr_pad*4, dtype = torch.float, device = device)
        out_weights = torch.zeros(1, 1, h_lr_pad*4, w_lr_pad*4, dtype = torch.float, device = device)
        weights = gaussian_weights(size*4, 0.05, device)

        i = 0
        start = 0
        while (i+size) <= h_lr_pad:
            j = 0
            while (j+size) <= w_lr_pad:
                out_img[:, :, i*4:(i+size)*4, j*4:(j+size)*4] += out_crop[start:start+1] * weights
                out_weights[:, :, i*4:(i+size)*4, j*4:(j+size)*4] += weights
                start += 1
                j += stride
            i += stride

        out_img = out_img / out_weights

        # original size
        out_img = out_img[:,:,pad_up*4:(pad_up+h_lr)*4,pad_left*4:(pad_left+w_lr)*4]

        out_img = out_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        out_img = np.transpose(out_img[[2, 1, 0], :, :], (1, 2, 0))
        out_img = (out_img * 255.).round().astype(np.uint8)

        save_path = os.path.join(args.output, f'{img_name}_out{extension}')
        cv2.imwrite(save_path, out_img)

    end_time = time.perf_counter()
    print(f'Total inference time: {(end_time - start_time):0.2f} seconds')

if __name__ == '__main__':
    main()
