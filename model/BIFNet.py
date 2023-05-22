import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.checkpoint as checkpoint
import os
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from math import sqrt
from .DSMABlock import DSMABlock
from .SwinTransformer import SwinTransformerBackbone,PatchExpand,FinalPatchExpand_X4


import time
INPUT_SIZE = 512
k=96
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class ScoreModule(nn.Module):
    def __init__(self, image_size=None):
        super(ScoreModule, self).__init__()
        ##加3×3的卷积
        self.extra_model = nn.Sequential(nn.Conv2d(k, k, 3, padding=1),
                                         nn.BatchNorm2d(k),
                                         nn.ReLU(),
                                         nn.Conv2d(k, k, 3, padding=1),
                                         nn.BatchNorm2d(k),
                                         nn.ReLU(),
                                         nn.Conv2d(k, k, 3, padding=1),
                                         nn.BatchNorm2d(k),
                                         nn.ReLU())
        self.conv_1 = nn.Conv2d(in_channels=k, out_channels=1, kernel_size=1, stride=1)
        self.image_size = image_size

    def forward(self, x):
        x = self.extra_model(x)
        x = self.conv_1(x)
        if self.image_size != None:
            print("do interpolate to size:", self.image_size)
            pred = F.interpolate(input=x, size=self.image_size, mode='bilinear', align_corners=True)
        else:
            pred = x
        # print("predict shape:", pred.shape)
        return pred

class BIFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                        embed_dim=96, depths=[2, 2, 6, 2], num_heads_backbone=[3, 6, 12, 24],
                        num_heads_cm=[3, 6, 12, 24], window_size=7, pred_size=224):
        super(BIFNet, self).__init__()
        # self.batch_size = batch_size
        self.backbone = SwinTransformerBackbone(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                        embed_dim=embed_dim, depths=depths, num_heads=num_heads_backbone, window_size=window_size)
                        
        self.cm_module = nn.ModuleList() # CMT Module
        self.num_layers = len(depths)
        self.patch_reso = img_size // patch_size
        self.upsample = nn.ModuleList()
        # self.feature_scale = []
        for i_layer in range(self.num_layers):
            fea_reso= self.patch_reso // (2 ** i_layer)
            dim = int(embed_dim * 2 ** i_layer)
            layer = DSMABlock(dim = dim, num_heads=num_heads_cm[i_layer], mlp_ratio=4.,
                                    qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            self.cm_module.append(layer)
            upsample = PatchExpand(input_resolution=[fea_reso, fea_reso], dim=dim, dim_scale=2, norm_layer=nn.LayerNorm)
            self.upsample.append(upsample)
        self.cm_module = self.cm_module[::-1]
        self.upsample = self.upsample[::-1] # reverse the list
        self.upsample_x4 = FinalPatchExpand_X4(input_resolution=[self.patch_reso, self.patch_reso], dim=embed_dim, dim_scale=4, norm_layer=nn.LayerNorm)
        if pred_size != img_size:
            self.score_module = ScoreModule(image_size=img_size)
        else:
            self.score_module = ScoreModule(image_size=None)

    def load_pretrained(self, load_path="./pretrained/swin_tiny_patch4_window7_224.pth"):
        if not os.path.exists(load_path):
            print("pretrained model path not exist")
        pretrained_dict = torch.load(load_path)
        for k, v in pretrained_dict.items():
            pretrained_dict = v

        model_dict = self.backbone.state_dict()

        renamed_dict = dict()
        for k, v in pretrained_dict.items():
            k = k.replace('layers.0.downsample', 'downsamples.0')
            k = k.replace('layers.1.downsample', 'downsamples.1')
            k = k.replace('layers.2.downsample', 'downsamples.2')
            if k in model_dict:
                renamed_dict[k] = v
        model_dict.update(renamed_dict)
        self.backbone.load_state_dict(model_dict)
             
    def forward(self, x, k):

         # backbone extract rgb-d features, with 4 side-output of different scales
        side_x = self.backbone(x)
        side_x = side_x[::-1] # reverse the list, deep features are placed in front
        # print("=============================")
        # print("backbone block 0:")
        # print("side_x shape:",side_x[0].shape)

        # fuse rgb-d features of different scales, respectively
        # the first fuse layer
        rgb_fea, d_fea = self.cm_module[0](side_x[0][0:k], side_x[0][k:2*k])
        # print("mutual fused fea shape:", rgb_fea.shape, d_fea.shape)
        fused_fea = (rgb_fea + d_fea + (rgb_fea * d_fea))
        # print("fused fea shape:", fused_fea.shape)
        for i in range(1, self.num_layers):
            # print("=============================" )
            # print("backbone block %d:"%i)
            # print("side_x shape:", side_x[i].shape)
            # up-sample the last output
            rgb_fea = self.upsample[i-1](rgb_fea)
            # print("after upsample:",rgb_fea.shape)
            d_fea = self.upsample[i-1](d_fea)
            fused_fea = self.upsample[i-1](fused_fea)
            # mutual attention fused rgb-d features, with residual connection of the last output
            rgb_fea, d_fea = self.cm_module[i](side_x[i][0:k]+rgb_fea, side_x[i][k:2*k]+d_fea)
            # print("mutual fused fea shape:",rgb_fea.shape, d_fea.shape)
            # element-wise fuse, with residual connection of the last output
            fused_fea = (rgb_fea + d_fea + (rgb_fea * d_fea)) + fused_fea
            # print("fused fea shape:", fused_fea.shape)
            # print("element-wise fused fea shape:", fused_fea.shape)
        # print("=============================")
        # the final expand_x4 layer
        fused_fea = self.upsample_x4(fused_fea)   # [1, h/4 * w/4, 96] -> [1, 96, h, w]
        # print("final feature size:",fused_fea.shape)
        # predict the saliency map
        predict = self.score_module(fused_fea) # [1, 1, h, w]
        return predict

    def load_pre(self,pretrained):
        if pretrained:
            self.load_pretrained(load_path=pretrained)
            print(f"loading pre_model ${pretrained}")


