# .svsのファイルから組織領域部分の座標を.csvファイルに書き出す
# すでにあるファイルも上書き処理をして時間を食うので要修正
import os
import csv
from PIL import Image, ImageStat
import numpy as np
import cv2
import openslide

from tqdm import tqdm

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def readCSV(filepath):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    for row in reader:
        file_data.append(row)
    csv_data.close()
    return file_data

def saveDic(filepath, dict):
    f = open(filepath, 'w', encoding='UTF-8')
    for key, value in dict.items():
        f.write(f'{key}:{value}\n')

DATA_PATH = './data' # 画像があるディレクトリへのパス

txt_data = readCSV('./data/Data_SimpleName.csv')
svs_tn_list = [t[0] for t in txt_data]
svs_fn_list = os.listdir(f'{DATA_PATH}/svs')

t_size = 4 #サムネイル中の1パッチのサイズ
b_size = 224

bar = tqdm(total = len(svs_tn_list))
for idx, slideID in enumerate(svs_tn_list):
    bar.set_description(slideID)
    bar.update(1)

    svs_fn = [s for s in svs_fn_list if slideID in s]
    if len(svs_fn) == 0:
        continue
    print(svs_fn)
    svs = openslide.OpenSlide(f'{DATA_PATH}/svs/{svs_fn[0]}')

    width,height = svs.dimensions
    b_w = width // b_size # x方向のパッチ枚数
    b_h = height // b_size # y方向のパッチ枚数
    print(width, height, b_w, b_h)

    thumb = Image.new('RGB',(b_w * t_size, b_h * t_size))   #標本サムネイル
    thumb_s = Image.new('L',(b_w, b_h)) #彩度分布画像

    for h_i in range(b_h):
        for w_i in range(b_w):
            #サムネイル作成
            b_img = svs.read_region((w_i * b_size, h_i * b_size),0,(b_size,b_size)).convert('RGB')
            r_img = b_img.resize((t_size, t_size), Image.BILINEAR)  #サムネイル用に縮小
            thumb.paste(r_img, (w_i * t_size, h_i * t_size))

            b_array = np.array(b_img)

            #b_array = np.asarray(b_img, dtype=np.uint8)
            #彩度画像作成
            R_b, G_b, B_b = cv2.split(b_array)
            Max_b = np.maximum(np.maximum(R_b, G_b), B_b)
            Min_b = np.minimum(np.minimum(R_b, G_b), B_b)
            Sat_b = Max_b - Min_b
            s_img = Image.fromarray(Sat_b)
            g_img = b_img.convert('L')
            statS = ImageStat.Stat(s_img)    #彩度画像の統計量取得
            statV = ImageStat.Stat(g_img)
            hsv = cv2.cvtColor(b_array, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            b_ratio = B_b / R_b
            #ratio_img = Image.fromarray(b_ratio)
            #statB = ImageStat.Stat(ratio_img)
            if statV.mean[0] < 230 and statV.mean[0] > 100 and np.count_nonzero(G_b > 230) < 224*224 / 2 and np.count_nonzero(G_b < 50) < 100 and statS.mean[0] > 0 and np.mean(b_ratio) > 0.9 and np.var(h) > 25 and np.count_nonzero(b_ratio > 1) > 224*224 / 16:
                thumb_s.putpixel((w_i,h_i),round(statS.mean[0]))
            else:
                thumb_s.putpixel((w_i,h_i),0)

    makedir(f'{DATA_PATH}/svs_info/{slideID}')
    thumb.save(f'{DATA_PATH}/svs_info/{slideID}/{slideID}_thumb.tif')    #標本サムネイル保存
    thumb_s.save(f'{DATA_PATH}/svs_info/{slideID}/{slideID}_sat.tif')    #彩度分布画像保存

    s_array = np.asarray(thumb_s)   #cv形式に変換
    ret, s_mask = cv2.threshold(s_array, 0, 255, cv2.THRESH_OTSU) #判別分析法で二値化
    cv2.imwrite(f'{DATA_PATH}/svs_info/{slideID}/{slideID}_mask.tif', s_mask)
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
    makedir(f'{DATA_PATH}/svs_info/{slideID}')
    np.savetxt(f'{DATA_PATH}/svs_info/{slideID}/{slideID}.csv', pos, delimiter=',', fmt='%d')

    info = {
        'file name':slideID,
        'svs size':[width, height],
        'patch size':[b_size, b_size],
        'patch count':[b_w, b_h],
        'sample count':num_i,
        }
    saveDic(f'{DATA_PATH}/svs_info/{slideID}/info.txt', info)
