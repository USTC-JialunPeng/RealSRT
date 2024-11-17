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
This implementation based on [BasicSR](https://github.com/XPixelGroup/BasicSR)

```
git clone https://github.com/USTC-JialunPeng/RealSRT
cd RealSRT
pip install -r requirements.txt
python setup.py develop
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
