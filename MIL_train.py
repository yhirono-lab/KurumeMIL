# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
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

def makedir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            return


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
    for (input_tensor, slideID, class_label) in valid_loader:
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

SAVE_PATH = '/Dataset/Kurume_Dataset/yhirono/KurumeMIL'

#マルチプロセス (GPU) で実行される関数
#rank : mp.spawnで呼び出すと勝手に追加される引数で, GPUが割り当てられている
#world_size : mp.spawnの引数num_gpuに相当
def train_model(rank, world_size, train_slide, valid_slide, name_mode, depth, leaf, mag, classify_mode):
    setup(rank, world_size)

    ##################実験設定#######################################
    # mag = '20x' # ('5x' or '10x' or '20x' or '40x')
    EPOCHS = 40
    #device = 'cuda'
    ################################################################
    if classify_mode == 'subtype':
        dir_name = f'subtype_classify'
    elif leaf is not None:
        dir_name = f'depth-{depth}_leaf-{leaf}'
    else:
        dir_name = f'depth-{depth}_leaf-all'
    
    # # 訓練用と検証用に症例を分割
    import dataset_kurume as ds
    if classify_mode == 'leaf':
        train_dataset, valid_dataset, label_count = ds.load_leaf(train_slide, valid_slide, name_mode, depth, leaf)
    elif classify_mode == 'subtype':
        train_dataset, valid_dataset, label_count = ds.load_svs(train_slide, valid_slide, name_mode)
    
    if rank == 0:
        print(f'train split:{train_slide} train slide count:{len(train_dataset)}')
        print(f'valid split:{valid_slide}   valid slide count:{len(valid_dataset)}')
    
    makedir(f'{SAVE_PATH}/train_log/{dir_name}')
    log = f'{SAVE_PATH}/train_log/{dir_name}/log_{mag}_train-{train_slide}.csv'

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
    class_predictor = class_predictor(label_count)
    # model構築
    model = MIL(feature_extractor, class_predictor)
    model = model.to(rank)
    if rank == 0:
        print(model)

    #MultiGPUに対応する処理
    process_group = torch.distributed.new_group([i for i in range(world_size)])
    #modelのBatchNormをSyncBatchNormに変更してくれる
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    #modelをmulti GPU対応させる
    ddp_model = DDP(model, device_ids=[rank])

    # クロスエントロピー損失関数使用
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.001
    
     # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    # 訓練開始
    for epoch in range(EPOCHS):
        if rank == 0:
            print(f'epoch:{epoch}')

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
        #下のDataLoaderで設定したbatch_sizeで各GPUに分配
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
            lr = lr * 0.5 #学習率調整
        optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        class_loss, correct_num = train(ddp_model, rank, loss_fn, optimizer, train_loader)

        train_loss += class_loss
        train_acc += correct_num

        data_valid = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=valid_dataset,
            mag=mag,
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
            makedir(f'{SAVE_PATH}/model_params/{dir_name}')
            model_params_dir = f'{SAVE_PATH}/model_params/{dir_name}/{mag}_train-{train_slide}_epoch-{epoch}.pth'
            torch.save(ddp_model.module.state_dict(), model_params_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('valid', help='choose valid data split')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('--num_gpu', default=1, type=int, help='input gpu num')
    parser.add_argument('-c', '--classify_mode', default='leaf', choices=['leaf', 'subtype'], help='leaf->based on tree, simple->based on subtype')
    args = parser.parse_args()

    num_gpu = args.num_gpu #argでGPUを入力

    train_slide = args.train
    valid_slide = args.valid

    name_mode = args.name
    depth = args.depth
    leaf = args.leaf
    mag = args.mag # ('5x' or '10x' or '20x' or '40x')
    classify_mode = args.classify_mode

    #マルチプロセスで実行するために呼び出す
    #train_model : マルチプロセスで実行する関数
    #args : train_modelの引数
    #nprocs : プロセス (GPU) の数
    mp.spawn(train_model, args=(num_gpu, train_slide, valid_slide, name_mode, depth, leaf, mag, classify_mode), nprocs=num_gpu, join=True)
