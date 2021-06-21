# -*- coding: utf-8 -*-
import re
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torch.distributed
import numpy as np
import csv
import os
import dataloader_svs
import torch.multiprocessing as mp
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
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
    if not os.path.isdir(path):
        os.makedirs(path)

def select_epoch(log_file):
    if not os.path.exists(log_file):
        exit()
    train_log = np.loadtxt(log_file, delimiter=',', dtype='str')
    if train_log.shape[0]<7:
        exit()
    valid_loss = train_log[1:,3].astype(np.float32)
    valid_loss[:5]=1000
    return np.argmin(valid_loss)

def update_test_result(test_dir, test_name, epoch_m):
    test_fn_list = os.listdir(test_dir)
    for test_fn in test_fn_list:
        if test_name in test_fn and not f'epoch-{epoch_m}' in test_fn:
            os.remove(f'{test_dir}/{test_fn}')
            print(f'{test_fn} is removed')


def test(model, device, test_loader, output_file):
    model.eval() #テストモードに変更

    pred_label = []
    bar = tqdm(total = len(test_loader))
    for input_tensor, slideID, class_label, pos_list in test_loader:
        bar.update(1)
        input_tensor = input_tensor.to(device)
        #class_label = class_label.to(rank, non_blocking=True)
        # MILとバッチ学習のギャップを吸収
        for bag_num in range(input_tensor.shape[0]):
            with torch.no_grad():
                class_prob, class_hat, A = model(input_tensor[bag_num])
            
            pred_label.append([class_label, class_hat])

            class_softmax = F.softmax(class_prob, dim=1).squeeze(0)
            class_softmax = class_softmax.tolist() # listに変換

            # bagの分類結果と各パッチのattention_weightを出力
            f = open(output_file, 'a')
            f_writer = csv.writer(f, lineterminator='\n')
            slideid_tlabel_plabel = ['', slideID[bag_num], int(class_label[bag_num]), class_hat] + class_softmax # [slideID, 真のラベル, 予測ラベル] + [y_prob[0], y_prob[1], y_prob[2]]
            f_writer.writerow(slideid_tlabel_plabel)
            pos_x = []
            pos_y = []
            for pos in pos_list:
                pos_x.append(int(pos[0]))
                pos_y.append(int(pos[1]))
            f_writer.writerow(['pos_x']+pos_x) # 座標書き込み
            f_writer.writerow(['pos_y']+pos_y) # 座標書き込み
            attention_weights = A.cpu().squeeze(0) # 1次元目削除[1,100] --> [100]
            att_list = attention_weights.tolist()
            f_writer.writerow(['attention']+att_list) # 各instanceのattention_weight書き込み
            f.close()

SAVE_PATH = '.'

def test_model(gpu, train_slide, test_slide, name_mode, depth, leaf, mag, classify_mode, loss_mode, constant, augmentation):

    ##################実験設定#######################################
    device = f'cuda:{gpu}'
    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード
    ################################################################
    if classify_mode == 'subtype':
        dir_name = f'subtype_classify'
    elif leaf is not None:
        dir_name = classify_mode
        if loss_mode != 'normal':
            dir_name = f'{dir_name}_{loss_mode}'
        if loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{constant}'
        if augmentation:
            dir_name = f'{dir_name}_aug'
        dir_name = f'{dir_name}/depth-{depth}_leaf-{leaf}'
    else:
        dir_name = classify_mode
        if loss_mode != 'normal':
            dir_name = f'{dir_name}_{loss_mode}'
        if loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{constant}'
        if augmentation:
            dir_name = f'{dir_name}_aug'
        dir_name = f'{dir_name}/depth-{depth}_leaf-all'

    # 訓練用と検証用に症例を分割
    import dataset_kurume as ds
    if classify_mode == 'leaf' or classify_mode == 'new_tree':
        _, test_dataset, label_num = ds.load_leaf(train_slide, test_slide, name_mode, depth, leaf, classify_mode)
    elif classify_mode == 'subtype':
        _, test_dataset, label_num = ds.load_svs(train_slide, test_slide, name_mode)

    log = f'{SAVE_PATH}/train_log/{dir_name}/log_{mag}_train-{train_slide}.csv'
    epoch_m = select_epoch(log)
    print(f'best epoch is {epoch_m}')

    # resultファイルの作成
    makedir(f'{SAVE_PATH}/test_result/{dir_name}')
    result = f'{SAVE_PATH}/test_result/{dir_name}/test_{mag}_train-{train_slide}_epoch-{epoch_m}.csv'
    update_test_result(f'{SAVE_PATH}/test_result/{dir_name}', f'test_{mag}_train-{train_slide}', epoch_m)
    if os.path.exists(result):
        print(f'[{dir_name}/test_{mag}_train-{train_slide}_epoch-{epoch_m}.csv] has been already done')
        exit()
    f = open(result, 'w')
    f.close()

    # model読み込み
    from model import feature_extractor, class_predictor, MIL
    # 各ブロック宣言
    feature_extractor = feature_extractor()
    class_predictor = class_predictor(label_num)
    # DAMIL構築
    model = MIL(feature_extractor, class_predictor)
    model_params = f'{SAVE_PATH}/model_params/{dir_name}/{mag}_train-{train_slide}/{mag}_train-{train_slide}_epoch-{epoch_m}.pth'
    model.load_state_dict(torch.load(model_params, map_location='cuda'))
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
        class_count=label_num,
        mag=mag,
        bag_num=50,
        bag_size=100
    )

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # 学習
    test(model, device, test_loader, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('train', help='choose train data split')
    parser.add_argument('test', help='choose valid data split')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('--gpu', default=1, type=int, help='input gpu num')
    parser.add_argument('-c', '--classify_mode', default='leaf', choices=['leaf', 'subtype', 'new_tree'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='normal', choices=['normal','invarse','myinvarse','LDAM'], help='select loss type')
    parser.add_argument('-C', '--constant', default=None)
    parser.add_argument('-a', '--augmentation', action='store_true')
    args = parser.parse_args()

    if args.classify_mode != 'subtype':
        if args.depth == None:
            print(f'mode:{args.classify_mode} needs depth param')
            exit()

    if args.loss_mode == 'LDAM' and args.constant == None:
        print(f'when loss_mode is LDAM, input Constant param')
        exit()

    test_model(args.gpu, args.train, args.test, args.name, args.depth, args.leaf,
            args.mag, args.classify_mode, args.loss_mode, args.constant, args.augmentation)
