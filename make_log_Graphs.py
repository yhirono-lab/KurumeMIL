import os
import shutil
from re import S
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import MetaEstimatorMixin
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import utils

def load_logfile(file_dir, mag, lr):
    log_fn_list = os.listdir(f'{SAVE_PATH}/train_log/{file_dir}')
    print(lr,type(lr))
    log_fn_list = [log_fn for log_fn in log_fn_list if mag in log_fn and lr in log_fn]
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

def load_testresult(file_dir, mag, lr):
    result_fn_list = os.listdir(f'{SAVE_PATH}/test_result/{file_dir}')
    result_fn_list = [result_fn for result_fn in result_fn_list if mag in result_fn and lr in result_fn and 'epoch' in result_fn]
    print('load_testresult:',result_fn_list)
    result_data = []
    for result_fn in result_fn_list:
        csv_data = open(f'{SAVE_PATH}/test_result/{file_dir}/{result_fn}')
        reader = csv.reader(csv_data)
        for row in reader:
            if len(row) == 6 or len(row) == 9:
                result_data.append([int(row[2]), int(row[3])])
        print(len(result_data))
    return np.array(result_data)

# スライド単位の事後確率とラベルのリストを返す
def get_slide_prob_label(file_dir, mag, lr):
    pred_corpus = {}
    label_corpus = {}
    slide_id_list = []

    result_fn_list = os.listdir(f'{SAVE_PATH}/test_result/{file_dir}')
    result_fn_list = [result_fn for result_fn in result_fn_list if mag in result_fn and lr in result_fn and 'epoch' in result_fn]
    print('load_bagresult:',result_fn_list)

    for result_fn in result_fn_list:
        csv_file = f'{SAVE_PATH}/test_result/{file_dir}/{result_fn}'
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if(len(row)==6 or len(row)==9):
                    slide_id = row[1]
                    if len(row)==6:
                        prob_list = [float(row[4]), float(row[5])] # [DLBCLの確率, FLの確率, RLの確率]
                    elif len(row)==9:
                        prob_list = [float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])] # [DLBCLの確率, FLの確率, RLの確率]
                    
                    if(slide_id not in pred_corpus):
                        pred_corpus[slide_id] = []
                        label_corpus[slide_id] = int(row[2]) #正解ラベル

                    pred_corpus[slide_id].append(prob_list)
                    if(slide_id not in slide_id_list):
                        slide_id_list.append(slide_id)

    # slide単位の事後確率計算
    slide_prob = []
    true_label_list = []
    pred_label_list = []

    for slide_id in slide_id_list:
        prob_list = pred_corpus[slide_id]
        bag_num = len(prob_list) # Bagの数

        total_prob_list = [0.0 for i in prob_list[0]]
        for prob in prob_list:
            total_prob_list = total_prob_list + np.log(prob)
        total_prob_list = np.exp(total_prob_list / bag_num) 

        slide_prob.append(list(total_prob_list))
        true_label_list.append(label_corpus[slide_id])

        pred_label_list.append(np.argmax(total_prob_list))

    return slide_id_list, slide_prob, true_label_list, pred_label_list

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

def save_graph(args, data, max_epoch, save_dir, filename):
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
    plt.title(filename)

    utils.makedir(save_dir)
    utils.makedir(f'./graphs/all/{args.mag}_{args.lr}')
    plt.savefig(f'{save_dir}/acc_loss_graph.png')
    plt.savefig(f'./graphs/all/{args.mag}_{args.lr}/{filename}_acc_loss_graph.png')

    f = open(f'{save_dir}/average_log.csv', 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    f_writer.writerow(['epoch','train_loss','train_acc','valid_loss','valid_acc'])
    for idx, d in enumerate(data):
        f_writer.writerow([idx]+d.tolist())
    f.close()

def save_test_cm(args, bag_label, slide_label, save_dir, filename):
    cm = confusion_matrix(y_true=bag_label[0], y_pred=bag_label[1], labels=np.unique(bag_label[0]).tolist())
    print('bag result\n',cm)

    f = open(f'{save_dir}/test_analytics.csv', 'w')
    f_writer = csv.writer(f, lineterminator='\n')
    f_writer.writerow([filename])
    f_writer.writerow(['Bag']+[f'pred:{i}' for i in range(len(cm))])
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

    f_writer.writerow([])

    cm = confusion_matrix(y_true=slide_label[0], y_pred=slide_label[1], labels=np.unique(slide_label[0]).tolist())
    print('slide result\n',cm)
    f_writer.writerow(['Slide']+[f'pred:{i}' for i in range(len(cm))])
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

    shutil.copyfile(f'{save_dir}/test_analytics.csv', f'./graphs/all/{args.mag}_{args.lr}/{filename}_test_analytics.csv')

    print(classification_report(y_true=bag_label[0], y_pred=bag_label[1]))

def make_log_Graphs(args): 
    dir_name = utils.make_dirname(args)
    filename = utils.make_filename(args)
    print(filename)

    log_list, max_epoch = load_logfile(dir_name, args.mag, str(args.lr))
    ave_log = cal_log_ave(log_list, max_epoch)
    save_graph(args, ave_log, max_epoch, f'{SAVE_PATH}/graphs/{dir_name}/{args.mag}_{args.lr}', filename)

    test_data = load_testresult(dir_name, args.mag, str(args.lr))
    _, _, true_label_list, pred_label_list = get_slide_prob_label(dir_name, args.mag, str(args.lr))

    bag_label = [test_data[:,0], test_data[:,1]]
    slide_label = [true_label_list, pred_label_list]
    save_test_cm(args, bag_label, slide_label, f'{SAVE_PATH}/graphs/{dir_name}/{args.mag}_{args.lr}', filename)

SAVE_PATH = '.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--data', default='2nd', choices=['1st', '2nd', '3rd'])
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'vgg11'])
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('-c', '--classify_mode', default='kurume_tree', choices=['normal_tree', 'kurume_tree', 'subtype'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='ICE', choices=['CE','ICE','LDAM','focal','focal-weight'], help='select loss type')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('-C', '--constant', default=None)
    parser.add_argument('-g', '--gamma', default=None)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('--fc', action='store_true')
    parser.add_argument('--reduce', action='store_true')
    args = parser.parse_args()

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

    make_log_Graphs(args)