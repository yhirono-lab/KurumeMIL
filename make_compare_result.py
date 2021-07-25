import csv
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## make_log_Graphs.pyで生成されるlog_averageとtest_analyticsをまとめるプログラム

DATA_PATH = './graphs'

def makedir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            return

def load_recall_data(path):
    csv_data = open(path)
    reader = csv.reader(csv_data)
    for row in reader:
        if row[0] == 'recall':
            leaf0 = '{:.3f}'.format(float(row[1]))
            leaf1 = '{:.3f}'.format(float(row[2]))
            return [leaf0, leaf1]

def make_all_recall_list():
    recall_all = []
    for mode in model_name_list:
        recall_all += ['', mode, '']
    recall_all = np.array([recall_all])
    
    for i, data in enumerate(data_list):
        data_name = data_name_list[i]

        recall_model = None
        for j, model in enumerate(model_list):
            model_name = model_name_list[j]
            recall_list = [['Hodgkin', 'other']]

            for loss in loss_list:
                path = f'graphs/{data}{model}new_tree{loss}/depth-{args.depth}_leaf-{args.leaf}/test_analytics.csv'

                if not os.path.exists(path):
                    recall_list.append(['', ''])
                else:
                    recall_list.append(load_recall_data(path))
            
            if recall_model is None:
                space = [[data_name]]+[[l] for l in loss_name_list]
                recall_model = np.concatenate((space, recall_list), axis=1)
            else:
                space = [[''] for a in range(recall_model.shape[0])]
                recall_model = np.concatenate((recall_model, space, recall_list), axis=1)

        if recall_all.shape[0]==1:
            recall_all = np.concatenate((recall_all, recall_model), axis=0)
        else:
            space = [['' for a in range(recall_model.shape[1])]]
            recall_all = np.concatenate((recall_all, space, recall_model), axis=0)

    print(recall_all)
    np.savetxt('graphs/all_recall_list.csv', recall_all, delimiter=',', fmt='%s')

def make_all_log_graph():
    # loss_all = {}
    graph_title_list = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']

    for i, loss in enumerate(loss_list):
        graph_label_list = []
        log_data_list = [] # axis=2 => [train_loss, train_acc, valid_loss, valid_acc]

        for j, data in enumerate(data_list):
            for k, model in enumerate(model_list):
                path = f'graphs/{data}{model}new_tree{loss}/depth-{args.depth}_leaf-{args.leaf}/average_log.csv'
                
                if os.path.exists(path):
                    graph_label = f'{data_name_list[j]}_{model_name_list[k]}'
                    graph_label_list.append(graph_label)

                    log = np.loadtxt(path, delimiter=',', dtype='str')
                    log = log[1:21,1:].astype(np.float32)
                    log_data_list.append(log)

        log_data_list = np.stack(log_data_list, axis=0)
        # loss_all[loss_name_list[i]] = [log_data_list, graph_label_list]

        fig = plt.figure()
        for j in range(4):
            plt.subplot(2,2,j+1)
            log = log_data_list[:,:,j]
            x = np.array(range(log.shape[1]))+1

            for k,l in enumerate(log):
                plt.plot(x, l, label=graph_label_list[k])
            plt.legend(fontsize=8)
            plt.title(graph_title_list[j])
            plt.xlabel('epoch')
            plt.gca().set_ylim(bottom=-0.5)
            if j%2==1:
                plt.gca().set_ylim(top=1.5)
            plt.grid()

        fig.tight_layout(rect=[0,0,1,0.96])
        plt.subplots_adjust(wspace=0.25, hspace=0.55)
        fig.suptitle(f'{loss_name_list[i]}')
        plt.savefig(f'./graphs/compare/{loss_name_list[i]}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default='1', help='choose depth')
    parser.add_argument('--leaf', default='01', help='choose leafs')
    args = parser.parse_args()

    place = 'depth-1_leaf-01'

    model_list = ['', 'fc_', 'vgg11_']
    model_name_list = ['vgg16', 'vgg16_fc', 'vgg11']
    loss_list = ['', '_myinvarse', '_LDAM-0.1', '_LDAM-0.3', '_LDAM-0.5']
    loss_name_list = ['Cross_Entropy', 'CE-invarse', 'LDAM-0.1', 'LDAM-0.3', 'LDAM-0.5']
    data_list = ['', 'add_reduce_']
    data_name_list = ['few', 'few_add']

    # make_all_recall_list()
    make_all_log_graph()
                
                