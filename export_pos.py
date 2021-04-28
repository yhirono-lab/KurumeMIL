import os
import glob
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvf
import random
from PIL import Image, ImageStat
import numpy as np
import cv2

import openslide

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

DATA_PATH = '../Data'

train_slide = '12345' # 全部書き出したいので
valid_slide = '1'

# 訓練用と検証用に症例を分割
import dataset_kurume as ds
train_DLBCL, train_FL, train_RL, valid_DLBCL, valid_FL, valid_RL = ds.slide_split(train_slide, valid_slide)
train_domain = train_DLBCL + train_FL + train_RL
svs_list = os.listdir(f'{DATA_PATH}/svs')

t_size = 4 #サムネイル中の1パッチのサイズ
b_size = 224

for slideID in train_domain:

    svs_fn = [s for s in svs_list if slideID in s]
    svs = openslide.OpenSlide(f'{DATA_PATH}/svs/{svs_fn[0]}')
    width,height = svs.dimensions
    b_w = width // b_size # x方向のパッチ枚数
    b_h = height // b_size # y方向のパッチ枚数

    thumb = Image.new('RGB',(b_w * t_size, b_h * t_size))   #標本サムネイル
    thumb_s = Image.new('L',(b_w, b_h)) #彩度分布画像

    for h in range(b_h):
        for w in range(b_w):
            #サムネイル作成
            b_img = svs.read_region((w*b_size,h*b_size),0,(b_size,b_size)).convert('RGB')
            r_img = b_img.resize((t_size, t_size), Image.BILINEAR)  #サムネイル用に縮小
            thumb.paste(r_img, (w * t_size, h * t_size))

            b_array = np.array(b_img)

            #b_array = np.asarray(b_img, dtype=np.uint8)
            #彩度画像作成
            R_b, G_b, B_b = cv2.split(b_array)
            Max_b = np.maximum(np.maximum(R_b, G_b), B_b)
            Min_b = np.minimum(np.minimum(R_b, G_b), B_b)
            Sat_b = Max_b - Min_b
            s_img = Image.fromarray(Sat_b)
            statS = ImageStat.Stat(s_img)    #彩度画像の統計量取得
            statV = ImageStat.Stat(img_g)
            b_array = np.array(img)
            hsv = cv2.cvtColor(b_array, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            b_ratio = B_b / R_b
            #ratio_img = Image.fromarray(b_ratio)
            #statB = ImageStat.Stat(ratio_img)
            if statV.mean[0] < 230 and statV.mean[0] > 100 and np.count_nonzero(G_b > 230) < 224*224 / 2 and np.count_nonzero(G_b < 50) < 100 and statS.mean[0] > 0 and np.mean(b_ratio) > 0.9 and np.var(h) > 25 and np.count_nonzero(b_ratio > 1) > 224*224 / 16:
                thumb_s.putpixel((w,h),round(statS.mean[0]))
            else:
                thumb_s.putpixel((w,h),0)

    makedir(f'{DATA_PATH}/thumb')
    makedir(f'{DATA_PATH}/thumb_s')
    thumb.save(f'{DATA_PATH}/thumb/{slideID}_thumb.tif')    #標本サムネイル保存
    thumb_s.save(f'{DATA_PATH}/thumb_s/{slideID}_sat.tif')    #彩度分布画像保存

    s_array = np.asarray(thumb_s)   #cv形式に変換
    ret, s_mask = cv2.threshold(s_array, 0, 255, cv2.THRESH_OTSU) #判別分析法で二値化
    #s_mask = Image.fromarray(s_mask)    #PIL形式に変換

    num_i = np.count_nonzero(s_mask)
    pos = np.zeros((num_i,2))
    i = 0
    for h in range(b_h):
        for w in range(b_w):
            if not s_mask[h,w] == 0:
                pos[i][0] = w * b_size
                pos[i][1] = h * b_size
                i = i + 1

    np.savetxt(f'{DATA_PATH}/csv/{slideID}.csv', pos, delimiter=',', fmt='%d')
