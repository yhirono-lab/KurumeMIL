from itertools import count

from torch.nn.modules import loss
from torch.nn.modules.conv import LazyConvTranspose2d
from dataloader_svs import Dataset_svs
import os
import csv
from PIL import Image, ImageStat
import numpy as np
import cv2
import openslide
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


DATA_PATH = '/Dataset/Kurume_Dataset' # データディレクトリ
SVS_PATH = '/Raw/Kurume_Dataset'
slideID = '180183'
pos_list = np.loadtxt(f'{DATA_PATH}/svs_info/{slideID}/{slideID}.csv', delimiter=',', dtype='int')
np.random.shuffle(pos_list)
pos = pos_list[0,:].tolist()

b_size = 224
svs_list = os.listdir(f'{SVS_PATH}/svs')
svs_fn = [s for s in svs_list if slideID in s]
svs = openslide.OpenSlide(f'{SVS_PATH}/svs/{svs_fn[0]}')
img = svs.read_region((pos[0],pos[1]),0,(b_size,b_size)).convert('RGB')
print(img)
img.save(f'./patch_{pos[0]}_{pos[1]}.png')
# device = 'cuda:0'
# class_num_list = np.array([10,90])
# weights = torch.tensor([1/(10/100),1/(90/100)]).to(device)
# print(weights)
# x = torch.tensor([[0.3335,0.886],[0.3335,0.886]]).to(device)
# target = torch.tensor([0,1]).to(device)

# loss_fn = nn.CrossEntropyLoss().to(device)
# loss_fn_w = nn.CrossEntropyLoss(weight=weights).to(device)
# loss1 = loss_fn(x, target)
# loss2 = loss_fn_w(x, target)
# loss3 = -x[0,target[0]]+torch.log(torch.exp(x[0,0])+torch.exp(x[0,1]))
# loss4 = -x[1,target[1]]+torch.log(torch.exp(x[1,0])+torch.exp(x[1,1]))
# print(loss1, loss2, loss3, loss4, (weights[0]*loss3+weights[1]*loss4)/torch.sum(weights))

# sm = F.log_softmax(x, dim=1)
# print(F.nll_loss(sm, target, weight=weights))
# print(weights/torch.sum(weights))

# m_list = 1.0/np.sqrt(np.sqrt(class_num_list))
# m_list = m_list * 0.5
# m_list = torch.cuda.FloatTensor(m_list)
# m_list = m_list[None, :]
# index = F.one_hot(target, len(class_num_list)).type(torch.uint8)
# # index = index.type(torch.uint8)
# index_float = index.type(torch.cuda.FloatTensor)
# batch_m = torch.matmul(m_list, index_float.transpose(0,1))
# print(m_list,index,batch_m)
# x_m = x - batch_m
# print(x_m,x,index)
# output = torch.where(index, x_m, x)
# print(output)

# def check_patch(img, svs, pos_list):
#     bar = tqdm(total = len(pos_list))
#     count = 0
#     error = None
#     for pos in pos_list:
#         bar.update(1)
#         try:
#             patch = img.read_region((pos[0], pos[1]), 0, (b_size, b_size))
#         except:
#             if error is None:
#                 error = [svs,pos[0],pos[1]]
#                 print(error)
#             count += 1
#     if error is not None:
#        error = error + [len(pos_list), count]
#     return error

# DATA_PATH = '/Dataset/Kurume_Dataset' 
# SVS_PATH = '/Raw/Kurume_Dataset'

# b_size = 224

# svs_info_list = os.listdir(f'{DATA_PATH}/svs_info')
# svs_list = os.listdir(f'{SVS_PATH}/svs')

# split_num = 10
# args = sys.argv
# if len(args) > 1:
#     group = int(args[1])
#     split = len(svs_info_list)//split_num
#     if group<(split_num-1):
#         svs_info_list = svs_info_list[group*split : (group+1)*split]
#     if group==(split_num-1):
#         svs_info_list = svs_info_list[group*split : len(svs_info_list)]

# svs_info_list = ['180077','180146','180246','180091','180287','180619']
# print(svs_info_list)
# error_list = []

# for idx, svs in enumerate(svs_info_list):
#     if svs == 'error_list.txt':
#         continue
#     print(f'svs:{svs} {idx}/{len(svs_info_list)}')

#     svs_fn = [s for s in svs_list if svs in s]
#     pos_list = np.loadtxt(f'{DATA_PATH}/svs_info/{svs}/{svs}.csv', delimiter=',', dtype='int')
 
#     img = openslide.OpenSlide(f'{SVS_PATH}/svs/{svs_fn[0]}')

#     error = check_patch(img, svs, pos_list)
#     if error is not None:
#         error_list.append(error)

# if not os.path.exists('./error_list.csv'):
#     f = open('error_list.csv', 'w')
#     f_writer = csv.writer(f, lineterminator='\n')
#     f_writer.writerow(['SlideID','w_i', 'h_i', 'pos_count', 'error_count'])
#     f.close()

# f = open('error_list.csv', 'a')
# f_writer = csv.writer(f, lineterminator='\n')
# f_writer.writerows(error_list)
# f.close()
