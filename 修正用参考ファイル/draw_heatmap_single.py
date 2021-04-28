import numpy as np
from PIL import Image, ImageStat, ImageDraw
import argparse
import os, re, shutil, sys, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2

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
                summary_file = f'{attention_dir}/{slideID}_att.csv'
                if not os.path.exists(summary_file):
                    f = open(summary_file, 'w')
                    f.close()
                    slide_list.append(slideID)
            else: #attention書き込み
                f = open(summary_file, 'a')
                f_writer = csv.writer(f, lineterminator='\n')
                f_writer.writerow(row)
                f.close()

    return slide_list

def draw_heatmap(slideID, attention_dir, save_dir):
    b_size = 224    #ブロック画像のサイズ
    t_size = 4 #サムネイル画像中の1ブロックのサイズ
    img = cv2.imread(f'../../../../nvme/Data/DataKurume/thumb/{slideID}_thumb.tif')
    #thumb = Image.open(f'../../../../nvme/Data/DataKurume/thumb/{slideID}_thumb.tif')
    w, h = img.shape[1], img.shape[0]
    w_num = w // t_size
    h_num = h // t_size
    #draw = ImageDraw.Draw(thumb)
    makedir(save_dir)

    attention_file = f'{attention_dir}/{slideID}_att.csv'
    att_data = np.loadtxt(attention_file, delimiter=',')

    att = []
    pos_x = []
    pos_y = []
    for i in range (int(att_data.shape[0]/3)):
        att = att + att_data[i*3+2,:].astype(np.float32).tolist()
        pos_x = pos_x + att_data[i*3,:].astype(np.int).tolist()
        pos_y = pos_y + att_data[i*3+1,:].astype(np.int).tolist()
    att_max = max(att)
    att_min = min(att)
    for i in range (len(att)):
        att[i] = (att[i] - att_min) / (att_max - att_min) #attentionを症例で正規化

    cmap = plt.get_cmap('jet')

    for i in range (len(att)):
        cval = cmap(float(att[i]))
        cv2.rectangle(img, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4), int(((pos_y[i]/224)+1)*4)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
        #cv2.rectangle(img_att, (cb_w*4, cb_h*4), ((cb_w+1)*4, (cb_h+1)*4), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)
    cv2.imwrite(f'{save_dir}/{slideID}_map.tif', img)

if __name__ == "__main__":
    args = sys.argv
    mag = args[1]
    train_slide = args[2]
    DArate = 0
    result_file = f'./test_att/test_{mag}_train-{train_slide}_DArate-{DArate}_opt.csv'
    attention_dir = f'./att_tmp/test_{mag}_DArate-{DArate}'
    save_dir = f'./vis_att/test_{mag}_DArate-{DArate}'
    #save_dir = '../../../../nvme/Data/DataKurume/'+argvs[3]

    slide_list = norm_attention(result_file, attention_dir)

    for slide in slide_list:
        draw_heatmap(slide, attention_dir, save_dir)
    #    x=input()

    #draw_heatmap('ML_180387', attention_dir, save_dir)
