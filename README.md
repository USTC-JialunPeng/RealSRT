# RealSRT

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08254.pdf) | [BibTex](https://github.com/USTC-JialunPeng/RealSRT#citing) 

PyTorch implementation of ECCV 2024 paper 
"Confidence-Based Iterative Generation for Real-World Image Super-Resolution"

## Introduction
<div align=center>
<img src="./figures/teaser.png" width="70%" height="70%">

*Visualizations of our confidence-based iterative generation process for real-world SR.*
</div>

## Method
<div align=center>
<img src="./figures/pipeline.png">

*Overview of RealSRT.*
</div>

## Installation
This implementation is based on [BasicSR](https://github.com/XPixelGroup/BasicSR)

```
git clone https://github.com/USTC-JialunPeng/RealSRT
cd RealSRT
pip install -r requirements.txt
python setup.py develop
```

## Inference
1. Download the pre-trained [model](https://drive.google.com/file/d/1e8Ip7m174esI9nGtzD04xFS-3mA6GDOJ/view?usp=sharing) and place it in `./experiments/pretrained_models/`

2. Download the test dataset (e.g., RealSR), place input images in `/data/input/` and place target images (if available) in `/data/target/`

3. Testing
```
python inference_realsrt.py --input /data/input/ --output /data/results/ --model_path experiments/pretrained_models/net_g_80000.pth
```

4. To reproduce scores in Table 1, run
```
python calculate_metrics.py
```

## Citing
If our method is useful for your research, please consider citing.

```
@inproceedings{peng2024confidence,
  title={Confidence-Based Iterative Generation for Real-World Image Super-Resolution},
  author={Peng, Jialun and Luo, Xin and Fu, Jingjing and Liu, Dong},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
