# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import csv
import os
import dataloader_svs
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

import random
import utils
def train(model, rank, loss_fn, optimizer, train_loader):
    model.train() #訓練モードに変更
    train_class_loss = 0.0
    correct_num = 0
    for (input_tensor, slideID, class_label) in train_loader:
        # MILとバッチ学習のギャップを吸収
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            optimizer.zero_grad() #勾配初期化
            class_prob, class_hat, A = model(input_tensor[bag_num])
            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num])
            train_class_loss += class_loss.item()

            class_loss.backward() #逆伝播
            optimizer.step() #パラメータ更新
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return train_class_loss, correct_num

def valid(model, rank, loss_fn, valid_loader):
    model.eval() #訓練モードに変更
    test_class_loss = 0.0
    correct_num = 0
    for (input_tensor, slideID, class_label) in valid_loader:
        # MILとバッチ学習のギャップを吸収
        input_tensor = input_tensor.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor[bag_num])
            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num])
            test_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return test_class_loss, correct_num

#if __name__ == "__main__":
#マルチプロセス (GPU) で実行される関数
#rank : mp.spawnで呼び出すと勝手に追加される引数で, GPUが割り当てられている
#world_size : mp.spawnの引数num_gpuに相当
def train_model(rank, world_size, train_slide, valid_slide):
    setup(rank, world_size)

    ##################実験設定#######################################
    if rank == 0:
        print('train:'+train_slide)
        print('valid:'+valid_slide)
    #train_slide = '123'
    #valid_slide = '4'
    #test_slide = '5'
    mag = '40x' # ('5x' or '10x' or '20x')
    EPOCHS = 10
    #device = 'cuda'
    ################################################################
    # 訓練用と検証用に症例を分割
    import dataset_kurume as ds
    train_DLBCL, train_FL, train_RL, valid_DLBCL, valid_FL, valid_RL = ds.slide_split(train_slide, valid_slide)
    train_domain = train_DLBCL + train_FL + train_RL
    valid_domain = valid_DLBCL + valid_FL + valid_RL
    # 訓練slideにクラスラベル(DLBCL:1, nonDLBCL:0)とドメインラベル付与
    train_dataset = []
    for slideID in train_DLBCL:
        train_dataset.append([slideID, 0])
    for slideID in train_FL:
        train_dataset.append([slideID, 1])
    for slideID in train_RL:
        train_dataset.append([slideID, 2])

    valid_dataset = []
    for slideID in valid_DLBCL:
        valid_dataset.append([slideID, 0])
    for slideID in valid_FL:
        valid_dataset.append([slideID, 1])
    for slideID in valid_RL:
        valid_dataset.append([slideID, 2])

    makedir('train_log')
    log = f'train_log/log_{mag}_train-{train_slide}.csv'

    if rank == 0:
        #ログヘッダー書き込み
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
        f_writer.writerow(csv_header)
        f.close()

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # model読み込み
    from model import feature_extractor, class_predictor, MIL
    # 各ブロック宣言
    feature_extractor = feature_extractor()
    class_predictor = class_predictor()
    # MIL構築
    model = MIL(feature_extractor, class_predictor)
    model = model.to(rank)
    process_group = torch.distributed.new_group([i for i in range(world_size)])
    #modelのBatchNormをSyncBatchNormに変更してくれる
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    #modelをmulti GPU対応させる
    ddp_model = DDP(model, device_ids=[rank])

    # クロスエントロピー損失関数使用
    loss_fn = nn.CrossEntropyLoss()
    # SGDmomentum法使用
    #optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    lr = 0.001
    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    # 訓練開始
    for epoch in range(EPOCHS):

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        data_train = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=train_dataset,
            mag=mag,
            bag_num=50,
            bag_size=100
        )
        #Datasetをmulti GPU対応させる
        #下のDataLoaderでbatch_sizeで設定したbatch_sizeで各GPUに分配
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
            sampler=train_sampler
        )

        if epoch > 1 and epoch % 5 == 0:
            lr = lr * 0.1 #学習率調整
        optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        class_loss, correct_num = train(ddp_model, rank, loss_fn, optimizer, train_loader)

        train_loss += class_loss
        train_acc += correct_num

        data_valid = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=valid_dataset,
            mag=mag
            bag_num=50,
            bag_size=100
        )
        #Datasetをmulti GPU対応させる
        #下のDataLoaderでbatch_sizeで設定したbatch_sizeで各GPUに分配
        valid_sampler = torch.utils.data.distributed.DistributedSampler(data_valid, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        valid_loader = torch.utils.data.DataLoader(
            data_valid,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
            sampler=valid_sampler
        )

        # 学習
        class_loss, correct_num = valid(ddp_model, rank, loss_fn, valid_loader)

        valid_loss += class_loss
        valid_acc += correct_num

        train_loss /= float(len(train_loader.dataset))
        train_acc /= float(len(train_loader.dataset))
        valid_loss /= float(len(valid_loader.dataset))
        valid_acc /= float(len(valid_loader.dataset))

        f = open(log, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])
        f.close()
        # epochごとにmodelのparams保存
        if rank == 0:
            makedir('model_params')
            model_params = f'./model_params/{mag}_train-{train_slide}_epoch-{epoch}.pth'
            torch.save(ddp_model.module.state_dict(), model_params)

if __name__ == '__main__':

    num_gpu = 1 #GPU数

    args = sys.argv
    train_slide = args[1]
    valid_slide = args[2]

    #マルチプロセスで実行するために呼び出す
    #train_model : マルチプロセスで実行する関数
    #args : train_modelの引数
    #nprocs : プロセス (GPU) の数
    mp.spawn(train_model, args=(num_gpu, train_slide, valid_slide), nprocs=num_gpu, join=True)
