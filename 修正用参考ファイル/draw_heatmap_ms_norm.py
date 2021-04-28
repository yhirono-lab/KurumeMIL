import numpy as np
from PIL import Image, ImageStat, ImageDraw
import argparse
import os, re, shutil, sys, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def str_to_float(l):
    return float(l)

'''
各bagのattentionを正規化してスライドごとに集計
result_file : single_trainの出力ファイル
attention_dir : 正規化したattentionを保存するdir
'''
def norm_attention(result_file, attention_dir):
    makedir(attention_dir)
    slide_list = []
    with open(result_file, 'r') as f:
        reader = csv.reader(f)
        #header = next(reader)  # ヘッダーを読み飛ばしたい時
        count = 0
        for row in reader:
            if(len(row)==6):
                slideID = row[0]
                if not slideID in slide_list:
                    slide_list.append(slideID)

    return slide_list

def draw_heatmap(slideID, attention_dir, save_dir):
    b_size = 224    #ブロック画像のサイズ
    t_size = 4 #サムネイル画像中の1ブロックのサイズ
    img1 = cv2.imread(f'../../../../nvme/Data/DataKurume/thumb/{slideID}_thumb.tif')
    img2 = cv2.imread(f'../../../../nvme/Data/DataKurume/thumb/{slideID}_thumb.tif')
    #thumb = Image.open(f'../../../../nvme/Data/DataKurume/thumb/{slideID}_thumb.tif')
    w, h = img1.shape[1], img1.shape[0]
    w_num = w // t_size
    h_num = h // t_size
    makedir(save_dir)

    attention_file = f'{attention_dir}/{slideID}_att.csv'
    att_data = np.loadtxt(attention_file, delimiter=',')

    att1 = []
    att2 = []
    att = []
    pos_x = []
    pos_y = []
    for i in range (int(att_data.shape[0]/3)):
        att_tensor = torch.from_numpy(att_data[i*3+2,:].astype(np.float32)).clone()
        att_sm = F.softmax(att_tensor, dim=0)#.squeeze(0)
        att_list = att_sm.tolist() # listに変換
        att_max = max(att_list)
        att_min = min(att_list)
        for j in range (len(att_list)):
            att_list[j] = (att_list[j] - att_min) / (att_max - att_min) #attentionを症例で正規化
        att1 = att1 + att_list[0:100]
        att2 = att2 + att_list[100:200]
        att = att + att_data[i*3+2,:].astype(np.float32).tolist()
        pos_x = pos_x + att_data[i*3,0:100].astype(np.int).tolist()
        pos_y = pos_y + att_data[i*3+1,0:100].astype(np.int).tolist()

    cmap = plt.get_cmap('jet')

    for i in range (len(att1)):
        cval = cmap(float(att1[i]))
        cv2.rectangle(img1, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4), int(((pos_y[i]/224)+1)*4)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        cval = cmap(float(att2[i]))
        cv2.rectangle(img2, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4), int(((pos_y[i]/224)+1)*4)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)

    cv2.imwrite(f'{save_dir}/{slideID}_map_40x.tif', img1)
    cv2.imwrite(f'{save_dir}/{slideID}_map_5x.tif', img2)

if __name__ == "__main__":
    args = sys.argv
    mag1 = '40x'
    mag2 = '5x'
    train_slide = args[1]
    DArate = 0
    result_file = f'./test_att/test_{mag1}_{mag2}_train-{train_slide}_DArate-{DArate}_ms.csv'
    attention_dir = f'./att_tmp/test_{mag1}_{mag2}_DArate-{DArate}_ms'
    #attention_dir = './attention_tmp/'+argvs[2]
    save_dir = f'./vis_att/test_{mag1}_{mag2}_DArate-{DArate}_ms_norm'
    #save_dir = '../../../../nvme/Data/DataKurume/'+argvs[3]

    slide_list = norm_attention(result_file, attention_dir)

    for slide in slide_list:
        draw_heatmap(slide, attention_dir, save_dir)
