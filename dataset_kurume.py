# slide(症例)を訓練用とテスト(valid)用に分割
from MIL_train import train
import os
import csv
import numpy as np

def read_leafCSV(filepath, label):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    for row in reader:
        if os.path.exists(f'/Dataset/Kurume_Dataset/svs_info/{row[0]}') and row[1] != 'OI_ILPD':
        # if row[1] != 'OI_ILPD':
            file_data.append([row[0], label])
        elif row[1] == 'OI_ILPD':
            # print(f'slideID{row[0]}はOI_ILPDです')
            continue
        else:
            # print(f'SlideID-{row[0]}は存在しません')
            continue
    csv_data.close()
    return file_data

def reduce_data(data):
    flag_list = {}
    csv_data = open('../KurumeTree/add_data/add_flag_list.csv')
    reader = csv.reader(csv_data)
    for row in reader:
        flag_list[row[0]]=int(row[1])
    csv_data.close()
    
    reduced_data = []
    for d in data:
        if flag_list[d[0]] == 0:
           reduced_data.append([d[0], d[1]])

    return reduced_data 

def load_leaf(args):
    if args.classify_mode == 'leaf':
        dir_path = f'../KurumeTree/{args.data}result/{args.name}/unu_depth{args.depth}/leafs_data'
    if args.classify_mode == 'new_tree':
        dir_path = f'../KurumeTree/{args.data}result_teacher/FDC/{args.name}/unu_depth{args.depth}/leafs_data'

    if args.leaf != None:
        if int(args.leaf[0])%2!=0 or int(args.leaf[1])-int(args.leaf[0])!=1:
            print('隣接した葉ではありません')
            exit()

        dataset = []
        for num in args.leaf:
            leaf_data = read_leafCSV(f'{dir_path}/leaf_{num}.csv', int(num)%2)
            dataset.append(leaf_data)
        min_leaf = np.argmin([len(dataset[0]), len(dataset[1])])
        max_leaf = np.argmax([len(dataset[0]), len(dataset[1])])
        ratio = len(dataset[max_leaf])//len(dataset[min_leaf])
        print(f'{min_leaf}:{len(dataset[min_leaf])},{max_leaf}:{len(dataset[max_leaf])}')
        if args.reduce:
            dataset[max_leaf] = reduce_data(dataset[max_leaf])
            min_leaf = np.argmin([len(dataset[0]), len(dataset[1])])
            max_leaf = np.argmax([len(dataset[0]), len(dataset[1])])
            ratio = len(dataset[max_leaf])//len(dataset[min_leaf])
            print(f'many data reduced \n{min_leaf}:{len(dataset[min_leaf])},{max_leaf}:{len(dataset[max_leaf])}')

        train_dataset = []
        valid_dataset = []
        for num in args.leaf:
            for idx,slide in enumerate(dataset[int(num)]):
                if str((idx%5)+1) in args.train:
                    train_dataset.append(slide+[0])

                    if args.augmentation and int(num) == min_leaf:
                        for aug_i in range(np.min([7, ratio-1])):
                            train_dataset.append(slide+[aug_i+1])
                
                if str((idx%5)+1) in args.valid:
                    valid_dataset.append(slide+[0])
        
        return train_dataset, valid_dataset, 2

    else: 
        print('Please input leaf number')
        exit()
    #     leaf_list = os.listdir(dir_path)
    #     leaf_count = len(leaf_list)
        
    #     train_dataset = []
    #     valid_dataset = []
    #     for num in range(leaf_count):
    #         leaf_data = read_leafCSV(f'{dir_path}/leaf_{num}.csv', num)
    #         for idx, slide in enumerate(leaf_data):
    #             if str((idx%5)+1) in args.train:
    #                 train_dataset.append(slide)
                
    #             if str((idx%5)+1) in args.valid:
    #                 valid_dataset.append(slide)
        
    #     return train_dataset, valid_dataset, leaf_count


def read_CSV(filepath):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    name_data = {'DLBCL':0, 'FL':1, 'Reactive':2, 'CHL':3}
    for row in reader:
        if os.path.exists(f'/Dataset/Kurume_Dataset/svs_info/{row[0]}'):
            if row[1] in name_data:
                file_data.append([row[0], name_data[row[1]]])
            else:
                file_data.append([row[0], 4])
        else:
            # print(f'SlideID-{row[0]}は存在しません')
            continue
    csv_data.close()
    return file_data
    
def load_svs(args):
    file_name = f'./{args.data}data/Data_{args.name}Name.csv'
    svs_data = read_CSV(file_name)
    
    train_dataset = []
    valid_dataset = []
    for idx, slide in enumerate(svs_data):
        if str((idx%5)+1) in args.train:
            train_dataset.append(slide)
            
        if str((idx%5)+1) in args.valid:
            valid_dataset.append(slide)

    return train_dataset, valid_dataset, 5


                



