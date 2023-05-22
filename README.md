# BIFNet

Pytorch implementation for [BIFNet: Bimodal Information Fusion Network for Salient
Object Detection based on Transformer](https://ieeexplore.ieee.org/document/9882262).


# Requirements
* Python 3.7 <br>
* Pytorch 1.7.0 <br>
* Cuda 10.0 <br>
* Tensorboard 2.5.0

# Usage

## To Test
* Download the [testing dataset](https://github.com/kerenfu/LFSOD-Survey) and have it in the 'dataset/test/' folder. 
* Download the already-trained [BIFNet model](#trained-model-for-testing) and have it in the 'trained_weight/' folder.
* Change the `weight_name` and `res_save_path` in `test.py` to the model to be evaluated.
* Start to test with
```sh
python test.py  
```

## To Train 
* Download the [training dataset](https://pan.baidu.com/s/1ckNlS0uEIPV-iCwVzjutsQ)(eb2z) and have it in the 'dataset/train/'folder 
* Download the Swin-tiny [pretrained model]() and have it in the 'pretrain/' floder
* modify setting in options.py
* Start to train with
```sh
python train.py 
```



# Download

## Trained model for testing
- We provide [trained model]() on NJU2K and NLPR.

## Saliency map
- We provide [testing results]() of 6 datasets (NJU2K, NLPR, STERE, RGBD135, LFSD, SIP) 




# Citation
Please cite our paper if you find the work useful: 

    @INPROCEEDINGS{BIFNet,
    author={Wang, Zhuo and Xiao, Minyue and He, Jing and Zhang, Chao and Fu, Keren},
    booktitle={2022 3rd International Conference on Pattern Recognition and Machine Learning (PRML)}, 
    title={Bimodal Information Fusion Network for Salient Object Detection based on Transformer}, 
    year={2022},
    pages={38-48},
    doi={10.1109/PRML56267.2022.9882262}}