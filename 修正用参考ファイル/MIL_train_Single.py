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


def train(model, device, loss_fn, optimizer, train_loader):
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

        input_tensor = input_tensor.to(device, non_blocking=True).squeeze(0)
        class_label = class_label.to(device, non_blocking=True).squeeze(0)
        
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

def valid(model, device, loss_fn, valid_loader):
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
        input_tensor = input_tensor.to(device, non_blocking=True)
        class_label = class_label.to(device, non_blocking=True)
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


def train_model(device, train_slide, valid_slide, name_mode, depth, leaf, mag, classify_mode, loss_mode, constant, augmentation, restart, fc_flag):
    ##################実験設定#######################################
    EPOCHS = 20
    #device = 'cuda'
    ################################################################
    if classify_mode == 'subtype':
        dir_name = f'subtype_classify'
        if fc_flag:
            dir_name = f'fc_{dir_name}'
    elif leaf is not None:
        dir_name = classify_mode
        if loss_mode != 'normal':
            dir_name = f'{dir_name}_{loss_mode}'
        if loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{constant}'
        if augmentation:
            dir_name = f'{dir_name}_aug'
        if fc_flag:
            dir_name = f'fc_{dir_name}'
        dir_name = f'{dir_name}/depth-{depth}_leaf-{leaf}'
    else:
        dir_name = classify_mode
        if loss_mode != 'normal':
            dir_name = f'{dir_name}_{loss_mode}'
        if loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{constant}'
        if augmentation:
            dir_name = f'{dir_name}_aug'
        if fc_flag:
            dir_name = f'fc_{dir_name}'
        dir_name = f'{dir_name}/depth-{depth}_leaf-all'
    
    # # 訓練用と検証用に症例を分割
    import dataset_kurume as ds
    if classify_mode == 'leaf' or classify_mode == 'new_tree':
        train_dataset, valid_dataset, label_num = ds.load_leaf(train_slide, valid_slide, name_mode, depth, leaf, classify_mode, augmentation)
    elif classify_mode == 'subtype':
        train_dataset, valid_dataset, label_num = ds.load_svs(train_slide, valid_slide, name_mode)

    label_count = np.zeros(label_num)
    for i in range(label_num):
        label_count[i] = len([d for d in train_dataset if d[1] == i])
    print(f'train split:{train_slide} train slide count:{len(train_dataset)}')
    print(f'valid split:{valid_slide}   valid slide count:{len(valid_dataset)}')
    print(f'train label count:{label_count}')
    
    makedir(f'{SAVE_PATH}/train_log/{dir_name}')
    log = f'{SAVE_PATH}/train_log/{dir_name}/log_{mag}_train-{train_slide}.csv'

    if not restart:
        #ログヘッダー書き込み
        f = open(log, 'w')
        f_writer = csv.writer(f, lineterminator='\n')
        csv_header = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
        f_writer.writerow(csv_header)
        f.close()

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # model読み込み
    from model import feature_extractor, class_predictor, MIL, set_LossFunction, CEInvarse, LDAMLoss
    # 各ブロック宣言
    feature_extractor = feature_extractor()
    class_predictor = class_predictor(label_num)
    # model構築
    model = MIL(feature_extractor, class_predictor)

    # 途中で学習が止まってしまったとき用
    if restart:
        model_params_dir = f'{SAVE_PATH}/model_params/{dir_name}/{mag}_train-{train_slide}'
        if os.path.exists(model_params_dir):
            model_params_list = sorted(os.listdir(model_params_dir))
            model_params_file = f'{model_params_dir}/{model_params_list[-1]}'
            model.load_state_dict(torch.load(model_params_file))
            restart_epoch = len(model_params_list)
        else:
            restart_epoch = 0
    model = model.to(device)

    if fc_flag:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    # クロスエントロピー損失関数使用
    if loss_mode == 'normal':
        loss_fn = nn.CrossEntropyLoss().to(device)
    if loss_mode == 'myinvarse':
        loss_fn = CEInvarse(device, label_count).to(device)
    if loss_mode == 'LDAM':
        loss_fn = LDAMLoss(device, label_count, Constant=float(constant)).to(device)
    lr = 0.001
    
     # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    # 訓練開始
    for epoch in range(EPOCHS):
        print(f'epoch:{epoch}')

        if epoch > 1 and epoch % 5 == 0:
            lr = lr * 0.5 #学習率調整
        
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        if restart and epoch<restart_epoch:
            print('this epoch already trained')
            continue

        data_train = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=train_dataset,
            class_count=label_num,
            mag=mag,
            bag_num=50,
            bag_size=100
        )

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=os.cpu_count()//2
        )

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        class_loss, correct_num = train(model, device, loss_fn, optimizer, train_loader)

        train_loss += class_loss
        train_acc += correct_num

        data_valid = dataloader_svs.Dataset_svs(
            train=True,
            transform=transform,
            dataset=valid_dataset,
            class_count=label_num,
            mag=mag,
            bag_num=50,
            bag_size=100
        )

        #pin_memory=Trueの方が早くなるらしいが, pin_memory=Trueにすると劇遅になるケースがあり原因不明
        valid_loader = torch.utils.data.DataLoader(
            data_valid,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=os.cpu_count()//2
        )

        # 学習
        class_loss, correct_num = valid(model, device, loss_fn, valid_loader)

        valid_loss += class_loss
        valid_acc += correct_num

        # GPU１つあたりの精度を計算
        train_loss /= float(len(train_loader.dataset))
        train_acc /= float(len(train_loader.dataset))
        valid_loss /= float(len(valid_loader.dataset))
        valid_acc /= float(len(valid_loader.dataset))

        # epochごとにlossなどを保存
        f = open(log, 'a')
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])
        f.close()

        # epochごとにmodelのparams保存
        makedir(f'{SAVE_PATH}/model_params/{dir_name}/{mag}_train-{train_slide}')
        model_params_dir = f'{SAVE_PATH}/model_params/{dir_name}/{mag}_train-{train_slide}/{mag}_train-{train_slide}_epoch-{epoch}.pth'
        torch.save(model.module.state_dict(), model_params_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('valid', help='choose valid data split')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('--gpu', default=1, type=int, help='input gpu num')
    parser.add_argument('-c', '--classify_mode', default='new_tree', choices=['leaf', 'subtype', 'new_tree'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='normal', choices=['normal','invarse','myinvarse','LDAM'], help='select loss type')
    parser.add_argument('-C', '--constant', default=None)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('-r', '--restart', action='store_true')
    parser.add_argument('--fc', action='store_true')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' #argでGPUを入力

    if args.classify_mode != 'subtype':
        if args.depth == None:
            print(f'mode:{args.classify_mode} needs depth param')
            exit()
    
    if args.loss_mode == 'LDAM' and args.constant == None:
        print(f'when loss_mode is LDAM, input Constant param')
        exit()

    train_model(
        device, args.train, args.valid, args.name, args.depth, args.leaf,
        args.mag, args.classify_mode, args.loss_mode, args.constant, args.augmentation, args.restart, args.fc
    )