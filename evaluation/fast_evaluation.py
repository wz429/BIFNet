import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from .evaluator import Eval_thread
from .result_loader import EvalDataset

def main(pred_dir, dataset, output_dir=""):
    threads = []
    method = 'BIFNet'
    loader = EvalDataset(pred_dir, get_dir(dataset))
    thread = Eval_thread(loader, method, dataset, output_dir, cuda=True)
    threads.append(thread)
    print(thread.run())


def get_dir(dataset):
    if dataset == 'NJU2K':
        image_root = './dataset/test/NJU2K/'
        image_source = './dataset/test/NJU2K/test.lst'
    elif dataset == 'STERE':
        image_root = './dataset/test/STERE/'
        image_source = './dataset/test/STERE/test.lst'
    elif dataset == 'RGBD135':
        image_root = './dataset/test/RGBD135/'
        image_source = './dataset/test/RGBD135/test.lst'
    elif dataset == 'LFSD':
        image_root = './dataset/test/LFSD/'
        image_source = './dataset/test/LFSD/test.lst'
    elif dataset == 'NLPR':
        image_root = './dataset/test/NLPR/'
        image_source = './dataset/test/NLPR/test.lst'
    elif dataset == 'SIP':
        image_root = './dataset/test/SIP/'
        image_source = './dataset/test/SIP/test.lst'
    return image_root + 'GT'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='DSS')
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
