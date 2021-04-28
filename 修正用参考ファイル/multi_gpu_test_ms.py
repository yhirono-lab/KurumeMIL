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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' #適当な数字で設定すればいいらしいがよくわかっていない

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

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
    valid_acc = train_log[1:,4].astype(np.float32)
    acc_list = []
    total_epoch = valid_acc.shape[0]/8
    for i in range(int(total_epoch)):
        tmp = valid_acc[i*8:(i+1)*8]
        if i < 5:
            acc_list.append(0)
        else:
            acc_list.append(np.sum(tmp))
    return acc_list.index(max(acc_list))

import random
import utils
def train(model, rank, loss_fn, optimizer, train_loader, DArate):
    model.train() #訓練モードに変更
    train_class_loss = 0.0
    train_domain_loss = 0.0
    train_total_loss = 0.0
    correct_num = 0
    for (input_tensor, class_label, domain_label) in train_loader:
        # MILとバッチ学習のギャップを吸収
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        domain_label = domain_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            optimizer.zero_grad() #勾配初期化
            class_prob, domain_prob, class_hat = model(input_tensor[bag_num], 'train', DArate)
            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num])
            domain_loss = loss_fn(domain_prob, domain_label[bag_num])
            total_loss = class_loss + domain_loss
            train_class_loss += class_loss.item()
            train_domain_loss += domain_loss.item()

            #print('train_loss='+str(class_loss.item()))

            total_loss.backward() #逆伝播
            optimizer.step() #パラメータ更新
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return train_class_loss, correct_num

def valid(model, rank, loss_fn, test_loader):
    model.eval() #訓練モードに変更
    test_class_loss = 0.0
    correct_num = 0
    for (input_tensor, class_label, domain_label) in test_loader:
        # MILとバッチ学習のギャップを吸収
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor[bag_num], 'test', 0)
            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num])
            test_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])

            #print('valid_loss='+str(class_loss.item()))

    return test_class_loss, correct_num

def test(model, device, test_loader, output_file):
    model.eval() #テストモードに変更

    for (input_tensor1, input_tensor2, slideID, class_label, domain_label, pos_list) in test_loader:
        input_tensor1 = input_tensor1.to(device)
        input_tensor2 = input_tensor2.to(device)
        #class_label = class_label.to(rank, non_blocking=True)
        # MILとバッチ学習のギャップを吸収
        for bag_num in range(input_tensor1.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor1[bag_num], input_tensor2[bag_num])

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
            pos_x = pos_x + pos_x
            pos_y = pos_y + pos_y
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

#if __name__ == "__main__":
#マルチプロセス (GPU) で実行される関数
#rank : mp.spawnで呼び出すと勝手に追加される引数で, GPUが割り当てられている
#world_size : mp.spawnの引数num_gpuに相当
def test_model(train_slide, test_slide):

    ##################実験設定#######################################
    #train_slide = '123'
    #valid_slide = '4'
    #test_slide = '5'
    mag1 = '40x' # ('5x' or '10x' or '20x')
    mag2 = '5x' # ('5x' or '10x' or '20x')
    EPOCHS = 10
    #DArate = 0.01
    DArate = 0
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
        test_dataset.append([slideID, 0, 0])
    for slideID in test_FL:
        test_dataset.append([slideID, 1, 0])
    for slideID in test_RL:
        test_dataset.append([slideID, 2, 0])

    makedir('test_att')
    result = f'test_att/test_{mag1}_{mag2}_train-{train_slide}_DArate-{DArate}_ms.csv'

    f = open(result, 'w')
    # f_writer = csv.writer(f, lineterminator='\n')
    # csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "time"]
    # f_writer.writerow(csv_header)
    f.close()

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # model読み込み
    from model import feature_extractor, class_predictor, domain_predictor, DAMIL, MSDAMIL
    # 各ブロック宣言
    log = f'train_log/log_{mag1}_{mag2}_train-{train_slide}_DArate-{DArate}_ms.csv'
    epoch_m = select_epoch(log)
    makedir('model_params')
    model_params = f'../../../../nvme/hashimoto.n/Output/model_params/{mag1}_{mag2}_train-{train_slide}_DArate-{DArate}_epoch-{epoch_m}_ms.pth'
    feature_extractor_mag1 = feature_extractor()
    feature_extractor_mag2 = feature_extractor()
    class_predictor = class_predictor()
    #domain_predictor = domain_predictor(domain_num)
    model = MSDAMIL(feature_extractor_mag1, feature_extractor_mag2, class_predictor)# DAMIL構築
    #model = DAMIL(feature_extractor, class_predictor, domain_predictor)
    #state_dict = torch.load(model_params,map_location='cpu')
    model.load_state_dict(torch.load(model_params,map_location='cpu'))
    model = model.to(device)

    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    hori_test = HoriDataset.HoriDataset_multi(
        train=False,
        transform=transform,
        pyvips=False,
        dataset=test_dataset,
        mag1=mag1,
        mag2=mag2,
        bag_num=1000,
        bag_size=100
    )

    #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
    test_loader = torch.utils.data.DataLoader(
        hori_test,
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
