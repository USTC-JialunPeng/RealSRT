# flake8: noqa
import sys
sys.path.append('./')

import os.path as osp
from basicsr.train import train_pipeline

import realsrt.archs
import realsrt.data
import realsrt.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
