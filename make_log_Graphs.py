import os
from re import S
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def load_logfile(file_dir, mag):
    log_fn_list = os.listdir(f'{SAVE_PATH}/train_log/{file_dir}')
    log_fn_list = [log_fn for log_fn in log_fn_list if mag in log_fn]
    print(log_fn_list)
    log_list = []
    max_epoch = 0
    for log_fn in log_fn_list:
        log = np.loadtxt(f'{SAVE_PATH}/train_log/{file_dir}/{log_fn}', delimiter=',', dtype='str')
        if len(log.shape)==1:
            continue
        print(log.shape)
        log = log[1:,1:].astype(np.float32)
        log_list.append(log)
        if max_epoch < log.shape[0]:
            max_epoch=log.shape[0]
    return log_list, max_epoch

def load_testresult(file_dir, mag):
    result_fn_list = os.listdir(f'{SAVE_PATH}/test_result/{file_dir}')
    result_fn_list = [result_fn for result_fn in result_fn_list if mag in result_fn and 'epoch' in result_fn]
    print(result_fn_list)
    result_data = []
    for result_fn in result_fn_list:
        csv_data = open(f'{SAVE_PATH}/test_result/{file_dir}/{result_fn}')
        reader = csv.reader(csv_data)
        for row in reader:
            if len(row) == 6 or len(row) == 9:
                result_data.append([int(row[2]), int(row[3])])
        print(len(result_data))
    return np.array(result_data)

def cal_log_ave(log_list, max_epoch):
    ave_log_data = np.zeros((max_epoch, 4))    
    for epoch in range(max_epoch):
        count = 0
        total_data = np.zeros(4)
        for log in log_list:
            if epoch < log.shape[0]:
                count += 1
                total_data += log[epoch]
        ave_data = total_data/count
        ave_log_data[epoch] += ave_data
    return ave_log_data

def save_graph(data, max_epoch, save_dir):
    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set1')

    max_epoch = max_epoch if max_epoch<20 else 20

    x = np.array(range(max_epoch))
    train_loss = data[:max_epoch,0]
    train_acc = data[:max_epoch,1]
    valid_loss = data[:max_epoch,2]
    valid_acc = data[:max_epoch,3]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, train_loss, label='train_loss')
    ax.plot(x, train_acc, label='train_acc')
    ax.plot(x, valid_loss, label='valid_loss')
    ax.plot(x, valid_acc, label='valid_acc')

    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    # ax.set_ylim(0, 1.5)
    plt.show()

    makedir(save_dir)
    plt.savefig(f'{save_dir}/acc_loss_graph.png')

    f = open(f'{save_dir}/average_log.csv', 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    f_writer.writerow(['epoch','train_loss','train_acc','valid_loss','valid_acc'])
    for idx, d in enumerate(data):
        f_writer.writerow([idx]+d.tolist())
    f.close()

def save_test_cm(data, save_dir):
    cm = confusion_matrix(y_true=data[:,0], y_pred=data[:,1], labels=np.unique(data[:,0]).tolist())
    print(cm)

    f = open(f'{save_dir}/test_analytics.csv', 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    f_writer.writerow(['']+[f'pred:{i}' for i in range(len(cm))])
    total = 0
    correct = 0
    recall_list = []
    for i in range(len(cm)):
        row_total = 0
        for j in range(len(cm)):
            total += cm[i][j]
            row_total += cm[i][j]
        recall_list.append(cm[i][i]/row_total)
        correct += cm[i][i]
        f_writer.writerow([f'true:{i}']+cm[i].tolist())
    acc = correct/total
    f_writer.writerow(['recall']+recall_list)
    f_writer.writerow(['total',total])
    f_writer.writerow(['accuracy',acc])
    f.close()

def make_log_Graphs(depth, leaf, mag, classify_mode, loss_mode, constant, augmentation):
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

    log_list, max_epoch = load_logfile(dir_name, mag)
    ave_log = cal_log_ave(log_list, max_epoch)
    save_graph(ave_log, max_epoch, f'{SAVE_PATH}/graphs/{dir_name}')

    test_data = load_testresult(dir_name, mag)
    save_test_cm(test_data, f'{SAVE_PATH}/graphs/{dir_name}')

SAVE_PATH = '/Dataset/Kurume_Dataset/yhirono/KurumeMIL'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
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

    make_log_Graphs(args.depth, args.leaf,
            args.mag, args.classify_mode, args.loss_mode, args.constant, args.augmentation)