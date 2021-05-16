
# slide(症例)を訓練用とテスト(valid)用に分割
import os
import csv

def readCSV(filepath, label):
    csv_data = open(filepath)
    reader = csv.reader(csv_data)
    file_data = []
    for row in reader:
        if os.path.exists(f'/Dataset/Kurume_Dataset/svs_info/{row[0]}'):
            file_data.append([row[0], label])
        else:
            # print(f'SlideID-{row[0]}は存在しません')
            continue
    csv_data.close()
    return file_data

def load_leaf(train_num, valid_num, name_mode, depth, leaf):
    dir_path = f'../KurumeTree/result/{name_mode}/unu_depth{depth}/leafs_data'
    if leaf == None:
        leaf_list = os.listdir(dir_path)
        leaf_count = len(leaf_list)
        
        train_dataset = []
        valid_dataset = []
        for num in range(leaf_count):
            leaf_data = readCSV(f'{dir_path}/leaf_{num}.csv', num)
            for idx, slide in enumerate(leaf_data):
                if str((idx%5)+1) in train_num:
                    train_dataset.append(slide)
                
                if str((idx%5)+1) in valid_num:
                    valid_dataset.append(slide)
        
        return train_dataset, valid_dataset, leaf_count

    else:
        if int(leaf[0])%2!=0 or int(leaf[1])-int(leaf[0])!=1:
            print('隣接した葉ではありません')
            exit()

        train_dataset = []
        valid_dataset = []
        for num in leaf:
            leaf_data = readCSV(f'{dir_path}/leaf_{num}.csv', int(num)%2)
            for idx, slide in enumerate(leaf_data):
                if str((idx%5)+1) in train_num:
                    train_dataset.append(slide)
                
                if str((idx%5)+1) in valid_num:
                    valid_dataset.append(slide)
        
        return train_dataset, valid_dataset, 2
    


                



