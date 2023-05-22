import argparse
import os
import time
import shutil

def save_path():
    run = 0
    save_folder = "./results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)
    while os.path.exists(save_folder) :
        run += 1
        save_folder = "./results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)

    if os.path.exists(save_folder):
        is_exist_pth = 0
        for i in os.listdir(save_folder):
            if 'pth' in i:
                is_exist_pth = 1
        save_folder = "./results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)
        if is_exist_pth == 0:
            shutil.rmtree(save_folder)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)


    return save_folder


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--epoch', type=int, default=201, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=5, help='training batch size')
parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./pre_train/swin_tiny_patch4_window7_224.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='./dataset/train/RGBD_for_train_fast/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='./dataset/train/RGBD_for_train_fast/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='./dataset/train/RGBD_for_train_fast/GT/', help='the training gt images root')
parser.add_argument('--save_path', type=str, default='', help='the path to save models and logs')
opt = parser.parse_args()
print(opt.local_rank)

