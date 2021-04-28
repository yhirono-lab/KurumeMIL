import os
import glob
import torch
from torchvision import transforms
from torchvision.transforms import functional as tvf
import random
from PIL import Image
import numpy as np

import openslide

DATA_PATH = '../Data' # データディレクトリ

class Dataset_svs(torch.utils.data.Dataset):
    def __init__(self, dataset, mag='40x', train = True, transform = None, bag_num=50, bag_size=100):

        self.transform = transform
        self.train = train
        self.mag = mag

        self.bag_list = [] # 各バッグのパッチ情報を格納

        for slide_data in dataset:

            slideID = slide_data[0] #　症例ID
            label = slide_data[1] # クラスラベル

            # 座標ファイル読み込み
            pos = np.loadtxt(f'{DATA_PATH}/csv/{slideID}.csv', delimiter=',', dtype='int')
            if not self.train: # テストのときはシャッフルのシードを固定
                np.random.seed(seed=int(slideID.replace('ML_','')))
            np.random.shuffle(pos) #パッチをシャッフル
            #最大でbag_num個のバッグを作成
            if pos.shape[0] > bag_num*bag_size:
                pos = pos[0:(bag_num*bag_size),:]
            for i in range(pos.shape[0]//bag_size):
                patches = pos[i*bag_size:(i+1)*bag_size,:].tolist()
                self.bag_list.append([patches, slideID, label])

        if self.train: # trainの場合，バッグをシャッフル
            random.shuffle(self.bag_list)
        self.data_num = len(self.bag_list)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        pos_list = self.bag_list[idx][0]
        patch_len = len(pos_list) # パッチ数

        b_size = 224 #パッチ画像サイズ

        # 症例IDを含む名前のsvsファイルを取得
        svs_list = os.listdir(f'{DATA_PATH}/svs')
        svs_fn = [s for s in svs_list if self.bag_list[idx][1] in s]
        svs = openslide.OpenSlide(f'{DATA_PATH}/svs/{svs_fn[0]}')

        # 出力バッグ
        bag = torch.empty(patch_len, 3, 224, 224, dtype=torch.float)
        i = 0
        # 画像読み込み
        for pos in pos_list:
            if self.transform: # どの倍率も中心座標は同じ
                if self.mag == '40x':
                    img = svs.read_region((pos[0],pos[1]),0,(b_size,b_size)).convert('RGB')
                elif self.mag == '20x':
                    img = svs.read_region((pos[0]-(int(b_size/2)),pos[1]-(int(b_size/2))),0,(b_size*2,b_size*2)).convert('RGB')
                elif self.mag == '10x':
                    img = svs.read_region((pos[0]-(int(b_size*3/2)),pos[1]-(int(b_size*3/2))),1,(b_size,b_size)).convert('RGB')
                elif self.mag == '5x':
                    img = svs.read_region((pos[0]-(int(b_size*7/2)),pos[1]-(int(b_size*7/2))),1,(b_size*2,b_size*2)).convert('RGB')
                img = self.transform(img)
                bag[i] = img
            i += 1

        slideID = self.bag_list[idx][1]
        label = self.bag_list[idx][2]
        label = torch.LongTensor([label])
        # バッグとラベルを返す
        if self.train:
            return bag, slideID, label
        else:
            return bag, slideID, label, pos_list
