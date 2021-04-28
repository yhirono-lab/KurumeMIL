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
def train(model, rank, loss_fn, optimizer, train_loader):
    model.train() #訓練モードに変更
    train_class_loss = 0.0
    correct_num = 0
    for (input_tensor1, input_tensor2, slideID, class_label, domain_label) in train_loader:
        # MILとバッチ学習のギャップを吸収
        input_tensor1 = input_tensor1.to(rank, non_blocking=True)
        input_tensor2 = input_tensor2.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor1.shape[0]):
            optimizer.zero_grad() #勾配初期化
            class_prob, class_hat, A = model(input_tensor1[bag_num], input_tensor2[bag_num])
            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num])
            train_class_loss += class_loss.item()

            #print('train_loss='+str(class_loss.item()))

            class_loss.backward() #逆伝播
            optimizer.step() #パラメータ更新
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return train_class_loss, correct_num

def valid(model, rank, loss_fn, test_loader):
    model.eval() #訓練モードに変更
    test_class_loss = 0.0
    correct_num = 0
    for (input_tensor1, input_tensor2, slideID, class_label, domain_label) in test_loader:
        # MILとバッチ学習のギャップを吸収
        input_tensor1 = input_tensor1.to(rank, non_blocking=True)
        input_tensor2 = input_tensor2.to(rank, non_blocking=True)
        class_label = class_label.to(rank, non_blocking=True)
        for bag_num in range(input_tensor1.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor1[bag_num], input_tensor2[bag_num])
            # 各loss計算
            class_loss = loss_fn(class_prob, class_label[bag_num])
            test_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, class_label[bag_num])

    return test_class_loss, correct_num

def test(model, device, test_data, output_file):
    model.eval() #テストモードに変更
    for data in test_data:
        #データ読み込み
        input_tensor, class_label, instance_list = utils.test_data_load(data)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            class_prob, class_hat, A = model(input_tensor, 'test', 0)

        class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
        class_softmax = class_softmax.tolist() # listに変換

        # bagの分類結果と各パッチのattention_weightを出力
        bag_id = data[0][0].split('/')[8]
        f = open(output_file, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        slideid_tlabel_plabel = [bag_id, int(class_label), class_hat] + class_softmax # [Bagの名前, 真のラベル, 予測ラベル] + [y_prob[1], y_prob[2]]
        f_writer.writerow(slideid_tlabel_plabel)
        f_writer.writerow(instance_list) # instance書き込み
        attention_weights = A.squeeze(0) # 1次元目削除[1,100] --> [100]
        attention_weights_list = attention_weights.tolist()
        f_writer.writerow(attention_weights_list) # 各instanceのattention_weight書き込み
        f.close()

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
    mag1 = '40x' # ('5x' or '10x' or '20x')
    mag2 = '5x' # ('5x' or '10x' or '20x')
    EPOCHS = 10
    DA = 'False'
    if DA == 'True':
        DArate = 0.01
    else:
        DArate = 0
    #DArate = 0.01
    #DArate = 0
    #device = 'cuda'
    ################################################################
    # 訓練用と検証用に症例を分割
    import dataset_kurume as ds
    train_DLBCL, train_FL, train_RL, valid_DLBCL, valid_FL, valid_RL = ds.slide_split(train_slide, valid_slide)
    train_domain = train_DLBCL + train_FL + train_RL
    valid_domain = valid_DLBCL + valid_FL + valid_RL
    domain_num = len(train_domain)
    if rank == 0:
        print('domain_num: '+str(domain_num))
    # 訓練slideにクラスラベル(DLBCL:1, nonDLBCL:0)とドメインラベル付与
    train_dataset = []
    for slideID in train_DLBCL:
        domain_idx = train_domain.index(slideID)
        train_dataset.append([slideID, 0, domain_idx])
    for slideID in train_FL:
        domain_idx = train_domain.index(slideID)
        train_dataset.append([slideID, 1, domain_idx])
    for slideID in train_RL:
        domain_idx = train_domain.index(slideID)
        train_dataset.append([slideID, 2, domain_idx])

    valid_dataset = []
    for slideID in valid_DLBCL:
        valid_dataset.append([slideID, 0, 0])
    for slideID in valid_FL:
        valid_dataset.append([slideID, 1, 0])
    for slideID in valid_RL:
        valid_dataset.append([slideID, 2, 0])

    # 出力ファイル
    if rank < 9:
        makedir('train_log')
        log = f'train_log/log_{mag1}_{mag2}_train-{train_slide}_DArate-{DArate}_ms.csv'
        makedir('valid_result')
        valid_result = f'valid_result/{mag1}_{mag2}_train-{train_slide}_valid-{valid_slide}_DArate-{DArate}_ms.csv'
        log1 = f'train_log/log_{mag1}_train-{train_slide}_DArate-{DArate}_opt.csv'
        log2 = f'train_log/log_{mag2}_train-{train_slide}_DArate-{DArate}_opt.csv'

    if rank == 0:
        #ログヘッダー書き込み
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "time"]
        f_writer.writerow(csv_header)
        f.close()

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # model読み込み
    from model import feature_extractor, class_predictor, domain_predictor, DAMIL, MSDAMIL
    # 各ブロック宣言
    epoch1 = select_epoch(log1)
    epoch2 = select_epoch(log2)
    if rank == 0:
        print(epoch1)
        print(epoch2)
    model_params1 = f'../../../../nvme/hashimoto.n/Output/model_params/{mag1}_train-{train_slide}_DArate-{DArate}_epoch-{epoch1}_opt.pth'
    model_params2 = f'../../../../nvme/hashimoto.n/Output/model_params/{mag2}_train-{train_slide}_DArate-{DArate}_epoch-{epoch2}_opt.pth'
    feature_extractor = feature_extractor()
    class_predictor = class_predictor()
    domain_predictor = domain_predictor(domain_num)
    #for p in domain_predictor.domain_classifier.parameters():
    #    p.requires_grad = False
    # DAMIL構築
    DAMIL = DAMIL(feature_extractor, class_predictor, domain_predictor)
    DAMIL.load_state_dict(torch.load(model_params1,map_location='cpu'))
    feature_extractor_mag1 = DAMIL.feature_extractor
    DAMIL.load_state_dict(torch.load(model_params2,map_location='cpu'))
    feature_extractor_mag2 = DAMIL.feature_extractor
    model = MSDAMIL(feature_extractor_mag1, feature_extractor_mag2, class_predictor)
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

    #count_rank = 0
    # 訓練開始
    for epoch in range(EPOCHS):
        # 訓練bag作成(epochごとにbag再構築)

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        #if rank == 0:
        start_t = time.time()

        hori_train = HoriDataset.HoriDataset_multi(
            train=True,
            transform=transform,
            pyvips=False,
            dataset=train_dataset,
            mag1=mag1,
            mag2=mag2,
            bag_num=50,
            bag_size=100
        )
        #Datasetをmulti GPU対応させる
        #下のDataLoaderでbatch_sizeで設定したbatch_sizeで各GPUに分配
        train_sampler = torch.utils.data.distributed.DistributedSampler(hori_train, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        train_loader = torch.utils.data.DataLoader(
            hori_train,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=4,
            sampler=train_sampler
        )

        if epoch > 1 and epoch % 5 == 0:
            lr = lr * 0.1
        optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        class_loss, correct_num = train(ddp_model, rank, loss_fn, optimizer, train_loader)
        #class_loss, domain_loss, acc = load(ddp_model, rank, loss_fn, optimizer, train_loader, lamda)
        # logに書き込み

        train_loss += class_loss
        train_acc += correct_num

        hori_valid = HoriDataset.HoriDataset_multi(
            train=True,
            transform=transform,
            pyvips=False,
            dataset=valid_dataset,
            mag1=mag1,
            mag2=mag2,
            bag_num=50,
            bag_size=100
        )
        #Datasetをmulti GPU対応させる
        #下のDataLoaderでbatch_sizeで設定したbatch_sizeで各GPUに分配
        valid_sampler = torch.utils.data.distributed.DistributedSampler(hori_valid, rank=rank)

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        valid_loader = torch.utils.data.DataLoader(
            hori_valid,
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
        elapsed_t = time.time() - start_t

        f = open(log, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc, elapsed_t])
        f.close()
        # epochごとにmodelのparams保存
        if rank == 0:
            makedir('model_params')
            model_params = f'../../../../nvme/hashimoto.n/Output/model_params/{mag1}_{mag2}_train-{train_slide}_DArate-{DArate}_epoch-{epoch}_ms.pth'
            torch.save(ddp_model.module.state_dict(), model_params)

if __name__ == '__main__':

    num_gpu = 8 #GPU数

    args = sys.argv
    train_slide = args[1]
    valid_slide = args[2]

    #マルチプロセスで実行するために呼び出す
    #train_model : マルチプロセスで実行する関数
    #args : train_modelの引数
    #nprocs : プロセス (GPU) の数
    mp.spawn(train_model, args=(num_gpu, train_slide, valid_slide), nprocs=num_gpu, join=True)
