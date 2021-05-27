from itertools import count
from dataloader_svs import Dataset_svs
import os
import csv
from PIL import Image, ImageStat
import numpy as np
import cv2
import openslide
import sys
from tqdm import tqdm

a = {'aaa':1}
print('bbb' in a)
print('aaa' in a)

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

# DATA_PATH = '/Dataset/Kurume_Dataset' # データディレクトリ
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
