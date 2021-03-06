# -*- coding: utf-8 -*-
from ast import parse
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import os
import dataloader_svs
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import sys
import argparse
from tqdm import tqdm

import utils

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' #適当な数字で設定すればいいらしいがよくわかっていない

    # initialize the process group
    # winではncclは使えないので、gloo
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

#正誤確認関数(正解:ans=1, 不正解:ans=0)
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

def train(model, rank, loss_fn, optimizer, train_loader):
    model.train() #訓練モードに変更
    accuracy = 0.0 #pbarの初期値用
    class_loss = 0.0 #pbarの初期値用
    train_class_loss = 0.0
    correct_num = 0
    count = 0

    bar = tqdm(total = len(train_loader))
    for input_tensor, slideID, class_label in train_loader:
        postfix = f'accuracy:{accuracy:.3f}, loss:{class_loss:.3f}'
        bar.set_postfix_str(postfix)
        bar.update(1)

        input_tensor = input_tensor.to(rank, non_blocking=True).squeeze(0)
        class_label = class_label.to(rank, non_blocking=True).squeeze(0)
        
        optimizer.zero_grad() #勾配初期化
        class_prob, class_hat, A = model(input_tensor)
        
        # 各loss計算
        class_loss = loss_fn(class_prob, class_label)
        # print(class_loss,class_prob,class_label)
        train_class_loss += class_loss.item()
        correct_num += eval_ans(class_hat, class_label)
        count += 1
        accuracy = correct_num / count

        class_loss.backward() #逆伝播
        optimizer.step() #パラメータ更新

    return train_class_loss, correct_num

def valid(model, rank, loss_fn, valid_loader):
    model.eval() #訓練モードに変更
    accuracy = 0.0 #pbarの初期値用
    class_loss = 0.0 #pbarの初期値用
    test_class_loss = 0.0
    correct_num = 0
    count = 0.0

    bar = tqdm(total = len(valid_loader))
    for input_tensor, slideID, class_label in valid_loader:
        postfix = f'accuracy:{accuracy:.3f}, loss:{class_loss:.3f}'
        bar.set_postfix_str(postfix)
        bar.update(1)

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
            count += 1
            accuracy = correct_num / count

    return test_class_loss, correct_num

SAVE_PATH = '.'

#マルチプロセス (GPU) で実行される関数
#rank : mp.spawnで呼び出すと勝手に追加される引数で, GPUが割り当てられている
#world_size : mp.spawnの引数num_gpuに相当
def train_model(rank, world_size, args):
    setup(rank, world_size)

    ##################実験設定#######################################
    EPOCHS = 20
    #device = 'cuda'
    ################################################################
    dir_name = utils.make_dirname(args)
    if rank == 0:
        print(dir_name)

    # # 訓練用と検証用に症例を分割
    import dataset_kurume as ds
    if args.classify_mode == 'normal_tree' or args.classify_mode == 'kurume_tree':
        train_dataset, valid_dataset, label_num = ds.load_leaf(args)
    elif args.classify_mode == 'subtype':
        train_dataset, valid_dataset, label_num = ds.load_svs(args)

    label_count = np.zeros(label_num)
    for i in range(label_num):
        label_count[i] = len([d for d in train_dataset if d[1] == i])
    if rank == 0:
        print(f'train split:{args.train} train slide count:{len(train_dataset)}')
        print(f'valid split:{args.valid}   valid slide count:{len(valid_dataset)}')
        print(f'train label count:{label_count}')
   
    utils.makedir(f'{SAVE_PATH}/train_log/{dir_name}')
    log = f'{SAVE_PATH}/train_log/{dir_name}/log_{args.mag}_{args.lr}_train-{args.train}.csv'

    if rank == 0 and not args.restart:
        #ログヘッダー書き込み
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
        f_writer.writerow(csv_header)
        f.close()

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # model読み込み
    from model import feature_extractor, class_predictor, MIL, CEInvarse, LDAMLoss, FocalLoss
    # 各ブロック宣言
    feature_extractor = feature_extractor(args.model)
    class_predictor = class_predictor(label_num)
    # model構築
    model = MIL(feature_extractor, class_predictor)

    # 途中で学習が止まってしまったとき用
    if args.restart:
        model_params_dir = f'{SAVE_PATH}/model_params/{dir_name}/{args.mag}_{args.lr}_train-{args.train}'
        if os.path.exists(model_params_dir):
            model_params_list = sorted(os.listdir(model_params_dir))
            model_params_file = f'{model_params_dir}/{model_params_list[-1]}'
            model.load_state_dict(torch.load(model_params_file))
            restart_epoch = len(model_params_list)
        else:
            restart_epoch = 0
    model = model.to(rank)
    if rank == 0:
        print(model)
    
    if args.fc:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    #MultiGPUに対応する処理
    process_group = torch.distributed.new_group([i for i in range(world_size)])
    #modelのBatchNormをSyncBatchNormに変更してくれる
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    #modelをmulti GPU対応させる
    ddp_model = DDP(model, device_ids=[rank])

    # クロスエントロピー損失関数使用
    if args.loss_mode == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(rank)
    if args.loss_mode == 'ICE':
        loss_fn = CEInvarse(rank, label_count).to(rank)
    if args.loss_mode == 'LDAM':
        loss_fn = LDAMLoss(rank, label_count, Constant=float(args.constant)).to(rank)
    if args.loss_mode == 'focal':
        loss_fn = FocalLoss(rank, label_count, gamma=float(args.gamma)).to(rank)
    if args.loss_mode == 'focal-weight':
        loss_fn = FocalLoss(rank, label_count, gamma=float(args.gamma), weight_flag=True).to(rank)
    lr = args.lr
    
     # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    # 訓練開始
    for epoch in range(EPOCHS):
        if rank == 0:
            print(f'epoch:{epoch}')

        if epoch > 1 and epoch % 5 == 0:
            lr = lr * 0.5 #学習率調整
        
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        if args.restart and epoch<restart_epoch:
            if rank == 0:
                print('this epoch already trained')
            continue

        data_train = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=train_dataset,
            class_count=label_num,
            mag=args.mag,
            bag_num=50,
            bag_size=100
        )
        
        #Datasetをmulti GPU対応させる
        #下のDataLoaderで設定したbatch_sizeで各GPUに分配
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=os.cpu_count()//world_size,
            sampler=train_sampler
        )

        optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        class_loss, correct_num = train(ddp_model, rank, loss_fn, optimizer, train_loader)

        train_loss += class_loss
        train_acc += correct_num

        data_valid = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=valid_dataset,
            class_count=label_num,
            mag=args.mag,
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
            num_workers=os.cpu_count()//world_size,
            sampler=valid_sampler
        )

        # 学習
        class_loss, correct_num = valid(ddp_model, rank, loss_fn, valid_loader)

        valid_loss += class_loss
        valid_acc += correct_num

        # GPU１つあたりの精度を計算
        train_loss /= float(len(train_loader.dataset))/float(world_size)
        train_acc /= float(len(train_loader.dataset))/float(world_size)
        valid_loss /= float(len(valid_loader.dataset))/float(world_size)
        valid_acc /= float(len(valid_loader.dataset))/float(world_size)

        # epochごとにlossなどを保存
        if rank == 0:
            f = open(log, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])
            f.close()

        # epochごとにmodelのparams保存
        if rank == 0:
            utils.makedir(f'{SAVE_PATH}/model_params/{dir_name}/{args.mag}_{args.lr}_train-{args.train}')
            model_params_dir = f'{SAVE_PATH}/model_params/{dir_name}/{args.mag}_{args.lr}_train-{args.train}/{args.mag}_{args.lr}_train-{args.train}_epoch-{epoch}.pth'
            torch.save(ddp_model.module.state_dict(), model_params_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('valid', help='choose valid data split')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--data', default='2nd', choices=['1st', '2nd', '3rd'])
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'vgg11'])
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('--num_gpu', default=1, type=int, help='input gpu num')
    parser.add_argument('-c', '--classify_mode', default='kurume_tree', choices=['normal_tree', 'kurume_tree', 'subtype'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='ICE', choices=['CE','ICE','LDAM','focal','focal-weight'], help='select loss type')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-C', '--constant', default=None)
    parser.add_argument('-g', '--gamma', default=None)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('-r', '--restart', action='store_true')
    parser.add_argument('--fc', action='store_true')
    parser.add_argument('--reduce', action='store_true')
    args = parser.parse_args()

    num_gpu = args.num_gpu #argでGPUを入力
    
    if args.data == '2nd' or args.data == '3rd':
        args.reduce = True

    if args.classify_mode != 'subtype':
        if args.depth == None:
            print(f'mode:{args.classify_mode} needs depth param')
            exit()
    
    if args.loss_mode == 'LDAM' and args.constant == None:
        print(f'when loss_mode is LDAM, input Constant param')
        exit()
    
    if (args.loss_mode == 'focal' or args.loss_mode == 'focal-weight') and args.gamma == None:
        print(f'when loss_mode is focal, input gamma param')
        exit()

    #マルチプロセスで実行するために呼び出す
    #train_model : マルチプロセスで実行する関数
    #args : train_modelの引数
    #nprocs : プロセス (GPU) の数
    mp.spawn(train_model, args=(num_gpu, args), nprocs=num_gpu, join=True)

    # プログラムが終わったらメールを送信するプログラムです．
    # utils.pyに自分のメールアドレスなどを設定したら使えます．
    # utils.send_email(body=str(args))
