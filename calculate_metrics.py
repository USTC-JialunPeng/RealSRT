import os
import cv2
import torch
import pyiqa
import numpy as np

from glob import glob
from natsort import natsorted

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

file_path = '/data/results/'
gt_path = '/data/target/'

file_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))

# PSNR
psnr_metric = pyiqa.create_metric('psnr', device=device)

# MS-SSIM
msssim_metric = pyiqa.create_metric('ms_ssim', device=device)

# LPIPS
lpips_metric = pyiqa.create_metric('lpips', device=device)

# DISTS
dists_metric = pyiqa.create_metric('dists', device=device)

# NIQE
niqe_metric = pyiqa.create_metric('niqe', device=device)

# CLIP-IQA
clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)

psnr_list = []
msssim_list = []
lpips_list = []
dists_list = []
niqe_list = []
clipiqa_list = []
for i in range(len(file_list)):
    print('Evaluating', i + 1)
    torch.cuda.empty_cache()

    file = cv2.cvtColor(cv2.imread(file_list[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
    file = torch.from_numpy(np.transpose(file, (2, 0, 1))).float().unsqueeze(0).to(device)

    gt = cv2.cvtColor(cv2.imread(gt_list[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
    gt = torch.from_numpy(np.transpose(gt, (2, 0, 1))).float().unsqueeze(0).to(device)

    psnr_list.append(psnr_metric(file, gt).item())
    
    msssim_list.append(msssim_metric(file, gt).item())
    
    lpips_list.append(lpips_metric(file, gt).item())
    
    dists_list.append(dists_metric(file, gt).item())

    niqe_list.append(niqe_metric(file).item())
    
    clipiqa_list.append(clipiqa_metric(file).item())

print(f'PSNR: {sum(psnr_list)/len(psnr_list):0.4f}; MS-SSIM: {sum(msssim_list)/len(msssim_list):0.6f}; LPIPS: {sum(lpips_list)/len(lpips_list):0.6f}; DISTS: {sum(dists_list)/len(dists_list):0.6f}; NIQE: {sum(niqe_list)/len(niqe_list):0.5f}; CLIP-IQA: {sum(clipiqa_list)/len(clipiqa_list):0.6f}')
