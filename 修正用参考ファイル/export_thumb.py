import os
import glob
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvf
import random
from PIL import Image, ImageStat
import numpy as np

import openslide

#DATA_PATH = '../../../../nvme/Data/DataKurume'
DATA_PATH = '../../../../nvme/Data/DataKurume'

train_slide = '12345'
valid_slide = '1'

# 訓練用と検証用に症例を分割
import dataset_kurume as ds
train_DLBCL, train_FL, train_RL, valid_DLBCL, valid_FL, valid_RL = ds.slide_split(train_slide, valid_slide)
train_domain = train_DLBCL + train_FL + train_RL
#train_domain = ['ML_180025']
svs_list = os.listdir(f'{DATA_PATH}/svs')

t_size = 4
b_size = 224

for slideID in train_domain:

    svs_fn = [s for s in svs_list if slideID in s]
    svs = openslide.OpenSlide(f'{DATA_PATH}/svs/{svs_fn[0]}')
    width,height = svs.dimensions
    max_w = width / 64
    max_h = height / 64
    b_w = width // 224
    b_h = height // 224

    thumb = Image.new('RGB',(b_w * t_size, b_h * t_size))   #標本サムネイル
    thumb_s = Image.new('L',(b_w, b_h)) #彩度分布画像

    for h in range(b_h):
        for w in range(b_w):
            #サムネイル作成
            b_img = svs.read_region((w*224,h*224),0,(224,224))
            r_img = b_img.resize((t_size, t_size), Image.BILINEAR)  #サムネイル用に縮小
            thumb.paste(r_img, (w * t_size, h * t_size))
            #彩度画像作成
            h_img, s_img, v_img = b_img.convert('HSV').split()  #RGB→HSV変換，チャンネルを分離
            stat = ImageStat.Stat(s_img)    #彩度画像の統計量取得
            thumb_s.putpixel((w,h),round(stat.mean[0]))

    thumb.save(f'{DATA_PATH}/thumb/{slideID}_thumb.tif')    #標本サムネイル保存
    thumb_s.save(f'{DATA_PATH}/thumb_s/{slideID}_sat.tif')    #彩度分布画像保存

    # img = svs.read_region((22200,27400),0,(200,200))
    # img.save(f'{DATA_PATH}/tmp/40x.tif')
    # img = svs.read_region((22100,27300),0,(400,400))
    # img.save(f'{DATA_PATH}/tmp/20x.tif')
    # img = svs.read_region((21900,27100),1,(200,200))
    # img.save(f'{DATA_PATH}/tmp/10x.tif')
    # img = svs.read_region((21500,26700),1,(400,400))
    # img.save(f'{DATA_PATH}/tmp/5x.tif')

    #thumb = svs.get_thumbnail((2048,2048))#((max_w,max_h))
    #thumb.save(f'{DATA_PATH}/thumb/{slideID}.tif')
