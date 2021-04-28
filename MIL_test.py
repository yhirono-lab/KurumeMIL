# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import csv
import os
import HoriDataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import sys

#正誤確認関数(正解:ans=1, 不正解:ans=0)
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def select_epoch(log_file):
    train_log = np.loadtxt(log_file, delimiter=',', dtype='str')
    valid_acc = train_log[1:,3].astype(np.float32)
    acc_list = []
    num_gpu = 1
    total_epoch = valid_acc.shape[0]/num_gpu # GPU数で割る
    for i in range(int(total_epoch)):
        tmp = valid_acc[i*num_gpu:(i+1)*num_gpu]
        if i < 5:
            acc_list.append(10000)
        else:
            acc_list.append(np.sum(tmp))
    return acc_list.index(min(acc_list))

import random
import utils

def test(model, device, test_loader, output_file):
    model.eval() #テストモードに変更

    for (input_tensor, slideID, class_label, pos_list) in test_loader:
        input_tensor = input_tensor.to(device)
        #class_label = class_label.to(rank, non_blocking=True)
        # MILとバッチ学習のギャップを吸収
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor[bag_num])

            class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
            class_softmax = class_softmax.tolist() # listに変換

            # bagの分類結果と各パッチのattention_weightを出力
            f = open(output_file, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            slideid_tlabel_plabel = [slideID[bag_num], int(class_label[bag_num]), class_hat] + class_softmax # [slideID, 真のラベル, 予測ラベル] + [y_prob[0], y_prob[1], y_prob[2]]
            f_writer.writerow(slideid_tlabel_plabel)
            pos_x = []
            pos_y = []
            for pos in pos_list:
                pos_x.append(int(pos[0]))
                pos_y.append(int(pos[1]))
            f_writer.writerow(pos_x) # 座標書き込み
            f_writer.writerow(pos_y) # 座標書き込み
            attention_weights = A.cpu().squeeze(0) # 1次元目削除[1,100] --> [100]
            attention_weights_list = attention_weights.tolist()
            att_list = []
            for att in attention_weights_list:
                att_list.append(float(att[0]))
            #att_list = attention_weights.tolist()
            f_writer.writerow(att_list) # 各instanceのattention_weight書き込み
            f.close()

def test_model(train_slide, test_slide):

    ##################実験設定#######################################
    #train_slide = '123'
    #valid_slide = '4'
    #test_slide = '5'
    mag = '40x' # ('5x' or '10x' or '20x')
    EPOCHS = 10
    device = 'cuda:0'
    ################################################################
    # 訓練用と検証用に症例を分割
    import dataset_kurume as ds
    train_DLBCL, train_FL, train_RL, test_DLBCL, test_FL, test_RL = ds.slide_split(train_slide, test_slide)
    train_domain = train_DLBCL + train_FL + train_RL
    test_domain = test_DLBCL + test_FL + test_RL
    domain_num = len(train_domain)
    test_dataset = []
    for slideID in test_DLBCL:
        test_dataset.append([slideID, 0])
    for slideID in test_FL:
        test_dataset.append([slideID, 1])
    for slideID in test_RL:
        test_dataset.append([slideID, 2])

    log = f'train_log/log_{mag}_train-{train_slide}.csv'
    epoch_m = select_epoch(log)
    makedir('test_result')
    result = f'test_result/test_{mag}_train-{train_slide}.csv'
    makedir('model_params')
    model_params = f'./model_params/{mag}_train-{train_slide}_epoch-{epoch_m}.pth'

    f = open(result, 'w')
    f.close()

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # model読み込み
    from model import feature_extractor, class_predictor, MIL
    # 各ブロック宣言
    feature_extractor = feature_extractor()
    class_predictor = class_predictor()
    # DAMIL構築
    model = MIL(feature_extractor, class_predictor)
    model.load_state_dict(torch.load(model_params,map_location='cpu'))
    model = model.to(device)

    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    data_test = dataloader_svs.Dataset_svs(
        train=False,
        transform=transform,
        dataset=test_dataset,
        mag=mag,
        bag_num=50,
        bag_size=100
    )

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    # 学習
    test(model, device, test_loader, result)

if __name__ == '__main__':

    args = sys.argv
    train_slide = args[1]
    test_slide = args[2]

    test_model(train_slide, test_slide)
