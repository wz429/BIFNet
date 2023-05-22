import torch
import torch.nn.functional as F
import sys
sys.path.append('models')
import numpy as np
import os, argparse
import cv2
from model.mga_model_ddp import CMFT
from data import test_dataset
import time
from evaluation import fast_evaluation


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=224, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/data/',help='test dataset path')
opt = parser.parse_args()
weight_name= './results/xxx/epoch_xxx.pth''
res_save_path = './results/xxx/'
dataset_path = opt.test_path

#set device for test

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print(f'USE GPU {opt.gpu_id}')


#load the model
model = CMFT(img_size=opt.testsize, patch_size=4,in_chans=3,
                 embed_dim=96, depths=[2,2,6,2],num_heads_backbone=[3,6,12,24],
                 num_heads_cm=[3,6,12,24], window_size=7, pred_size=224)

model = torch.nn.DataParallel(model)
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./results/') when training.
model.load_state_dict(torch.load(weight_name)) 
model.cuda()
model.eval()

#test
test_datasets = ['NJU2K','NLPR','STERE', 'RGBD135', 'LFSD','SIP']
for dataset in test_datasets:
    save_path = res_save_path + dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        torch.cuda.synchronize()
        time_s = time.time()
        input =torch.cat((image,depth),dim=0)
        k=1
        res = model(input,k)
        torch.cuda.synchronize()
        time_e = time.time()
        print('Speed: %f FPS' % (1 / (time_e - time_s)))
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        print('save img to: ',os.path.join(save_path, name))
        cv2.imwrite(os.path.join(save_path, name),res*255)
       

    fast_evaluation.main(save_path, dataset, os.path.split(save_path)[0])
    print('Test Done!')
