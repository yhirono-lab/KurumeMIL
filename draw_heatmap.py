from make_log_Graphs import load_logfile
import numpy as np
from PIL import Image, ImageStat, ImageDraw
import argparse
import os, re, shutil, sys, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import openslide
import torch
import torchvision
from tqdm import tqdm

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def get_slideID_name():
    file_name = './data/Data_FullName.csv'
    csv_data = np.loadtxt(file_name, delimiter=',', dtype='str')
    name_list = {}

    for i in range(1,csv_data.shape[0]):
        if csv_data[i,1] == 'N/A':
            csv_data[i,1] = 'NA'
        if csv_data[i,1] == 'CLL/SLL':
            csv_data[i,1] = 'CLL-SLL'
        name_list[csv_data[i,0]] = csv_data[i,1]

    return name_list

def load_att_data(dir_name):
    test_fn_list = os.listdir(f'{SAVE_PATH}/test_result/{dir_name}')
    att_data_list = {}
    for test_fn in test_fn_list:
        csv_data = open(f'{SAVE_PATH}/test_result/{dir_name}/{test_fn}')
        reader = csv.reader(csv_data)
        row_number = 0
        for row in reader:
            if len(row) == 6 or len(row) == 9:
                row_number = 0
                slideID = row[1]
                if slideID not in att_data_list:
                    att_data_list[slideID] = [row[2:],[],[],[]]
            else:
                att_data_list[slideID][row_number] += row[1:]
            row_number += 1
    print(len(att_data_list))
    
    return att_data_list

def load_bagatt_data(dir_name):
    test_fn_list = os.listdir(f'{SAVE_PATH}/test_result/{dir_name}')
    bagatt_data_list = {}
    for test_fn in test_fn_list:
        csv_data = open(f'{SAVE_PATH}/test_result/{dir_name}/{test_fn}')
        reader = csv.reader(csv_data)
        row_number = 0
        for row in reader:
            if len(row) == 6 or len(row) == 9:
                row_number = 0
                slideID = row[1]
                true_label = row[2]
                pred_label = row[3]

                if slideID not in bagatt_data_list:
                    bagatt_data_list[slideID] = [row[2:],[[],[],[]],[[],[],[]]]
            else:
                if true_label == pred_label:
                    bagatt_data_list[slideID][1][row_number] += row[1:]
                elif true_label != pred_label:
                    bagatt_data_list[slideID][2][row_number] += row[1:]
                row_number += 1
    print(len(bagatt_data_list))

    return bagatt_data_list

def draw_heatmap(args, dir_name):
    b_size = 224
    t_size = 4
    att_data_list = load_att_data(dir_name)

    save_dir = f'{SAVE_PATH}/attention_map/{dir_name}'
    makedir(save_dir)

    bar = tqdm(total = len(att_data_list))
    for slideID in att_data_list:
        bar.update(1)
        pos_x = [int(x) for x in att_data_list[slideID][1]]
        pos_y = [int(y) for y in att_data_list[slideID][2]]
        att = [float(a) for a in att_data_list[slideID][3]]
        att_max = max(att)
        att_min = min(att)
        for i in range(len(att)):
            att[i] = (att[i] - att_min) / (att_max - att_min) #attentionを症例で正規化
        
        img = cv2.imread(f'{DATA_PATH}/svs_info/{slideID}/{slideID}_thumb.tif')
        thumb = cv2.imread(f'{DATA_PATH}/svs_info/{slideID}/{slideID}_thumb.tif')

        height, width = img.shape[0], img.shape[1]
        w_num = width // t_size
        h_num = height // t_size

        cmap = plt.get_cmap('jet')
        att_map = np.ones((h_num, w_num,3))*255
        for i in range(len(att)):
            x = pos_x[i]//b_size
            y = pos_y[i]//b_size

            cval = cmap(float(att[i]))
            att_map[y,x,:] = [cval[2]*255, cval[1]*255, cval[0]*255]

            cv2.rectangle(img, (int(pos_x[i]*4/224), int(pos_y[i]*4/224)), (int(((pos_x[i]/224)+1)*4), int(((pos_y[i]/224)+1)*4)), (cval[2] * 255, cval[1] * 255, cval[0] * 255), thickness=-1)

        att_map = cv2.resize(np.uint8(att_map), (width, height))
        cv2.imwrite(f'{save_dir}/{slideID}_map.tif', att_map)
        cv2.imwrite(f'{save_dir}/{slideID}_blend.tif', img)
        cv2.imwrite(f'{save_dir}/{slideID}_thumb.tif', thumb)
    
def save_high_low_patches(args, dir_name):
    bagatt_data_list = load_bagatt_data(dir_name)

    save_dir = f'{SAVE_PATH}/attention_patch/{dir_name}'
    makedir(save_dir)

    bar = tqdm(total = len(bagatt_data_list))
    for slideID in bagatt_data_list:
        bar.update(1)
        
        data = bagatt_data_list[slideID]
        label = data[0][1]
        true_data = data[1]
        false_data = data[2]
        att = [float(a) for a in true_data[2]+false_data[2]]

        att_max = max(att)
        att_min = min(att)
        
        true_data[0] = [int(x) for x in true_data[0]]
        true_data[1] = [int(y) for y in true_data[1]]
        true_data[2] = [(float(a)-att_min)/(att_max-att_min) for a in true_data[2]]

        false_data[0] = [int(x) for x in false_data[0]]
        false_data[1] = [int(y) for y in false_data[1]]
        false_data[2] = [(float(a)-att_min)/(att_max-att_min) for a in false_data[2]]

        save_patch(slideID, label, true_data, save_dir, 'correct')
        save_patch(slideID, label, false_data, save_dir, 'incorrect') 
        
def save_patch(slideID, label, data, save_dir, flag):
    if len(data[2]) > 0:
        b_size = 224
        svs_fn = [s for s in svs_fn_list if slideID in s]
        svs = openslide.OpenSlide(f'/Raw/Kurume_Dataset/svs/{svs_fn[0]}')

        sort_idx = np.argsort(data[2])[::-1]
        save_many_patch(slideID, svs, sort_idx, label, data, save_dir, flag)

        fig, ax = plt.subplots(3, 3)
        for i in range(9):
            idx = sort_idx[i]
            pos_x = data[0][idx]
            pos_y = data[1][idx]
            att = data[2][idx]

            b_img = svs.read_region((pos_x, pos_y), 0, (b_size,b_size)).convert('RGB')
            # b_img.save(f'{save_dir}/save.png')
            # exit()
            b_img = np.array(b_img)

            plt.subplot(3,3,i+1)
            plt.title('{:.4f}'.format(att), fontsize=10)
            plt.tick_params(color='white')
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.imshow(b_img)
        
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0.25)
        img_name = f'{slideID}_{label}_{flag}_{slideID_name_dict[slideID]}_high'
        fig.suptitle(f'{img_name}')
        makedir(f'{save_dir}/{label}_{flag}')
        plt.savefig(f'{save_dir}/{label}_{flag}/{img_name}.tif', bbox_inches='tight', pad_inches=0.1, format='tif', dpi=300)


        sort_idx = np.argsort(data[2])
        for i in range(9):
            idx = sort_idx[i]
            pos_x = data[0][idx]
            pos_y = data[1][idx]
            att = data[2][idx]

            b_img = svs.read_region((pos_x, pos_y), 0, (b_size,b_size)).convert('RGB')
            b_img = np.array(b_img)

            plt.subplot(3,3,i+1)
            plt.title('{:.4f}'.format(att), fontsize=10)
            plt.tick_params(color='white')
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.imshow(b_img)
        
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0.25)
        img_name = f'{slideID}_{label}_{flag}_{slideID_name_dict[slideID]}_low'
        fig.suptitle(f'{img_name}')
        makedir(f'{save_dir}/{label}_{flag}')
        plt.savefig(f'{save_dir}/{label}_{flag}/{img_name}.tif', bbox_inches='tight', pad_inches=0.1, format='tif', dpi=300)


def save_many_patch(slideID, svs, sort_idx, label, data, save_dir, flag):
    b_size = 224
    img_num = 50
    images = np.zeros((img_num, b_size, b_size, 3), np.uint8)
    fig, ax = plt.subplots()
    for i in range(img_num):
        idx = sort_idx[i]
        pos_x = data[0][idx]
        pos_y = data[1][idx]
        att = data[2][idx]

        b_img = svs.read_region((pos_x, pos_y), 0, (b_size,b_size)).convert('RGB')
        images[i] = b_img
    
    images = np.transpose(images, [0,3,1,2]) # NHWC -> NCHW に変換
    images_tensor = torch.as_tensor(images)
    joined_images_tensor = torchvision.utils.make_grid(images_tensor, nrow=10, padding=10)
    joined_images = joined_images_tensor.numpy()

    jointed = np.transpose(joined_images, [1,2,0]) # NCHW -> NHWCに変換
    plt.tick_params(color='white', labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    img_name = f'{slideID}_{label}_{flag}_{slideID_name_dict[slideID]}'
    plt.title(f'{img_name}')
    plt.imshow(jointed)
    makedir(f'{save_dir}/{label}_{flag}/many_patch/attention')
    plt.savefig(f'{save_dir}/{label}_{flag}/many_patch/{img_name}.tif', bbox_inches='tight', pad_inches=0.1, format='tif', dpi=600)

    f = open(f'{save_dir}/{label}_{flag}/many_patch/attention/{img_name}.csv', 'w')
    for d in data:
        f_writer = csv.writer(f, lineterminator='\n')
        f_writer.writerow(d[0:50])
    f.close()


DATA_PATH = '/Dataset/Kurume_Dataset'
SAVE_PATH = '.'

svs_fn_list = os.listdir(f'/Raw/Kurume_Dataset/svs')
slideID_name_dict = get_slideID_name()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is MIL using Kurume univ. data')
    parser.add_argument('--depth', default=None, help='choose depth')
    parser.add_argument('--leaf', default=None, help='choose leafs')
    parser.add_argument('--mag', default='40x', choices=['5x', '10x', '20x', '40x'], help='choose mag')
    parser.add_argument('--name', default='Simple', choices=['Full', 'Simple'], help='choose name_mode')
    parser.add_argument('--gpu', default=1, type=int, help='input gpu num')
    parser.add_argument('-c', '--classify_mode', default='leaf', choices=['leaf', 'subtype', 'new_tree'], help='leaf->based on tree, simple->based on subtype')
    parser.add_argument('-l', '--loss_mode', default='normal', choices=['normal','myinvarse','LDAM'], help='select loss type')
    parser.add_argument('-C', '--constant', default=None)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('--fc', action='store_true')
    args = parser.parse_args()

    if args.classify_mode != 'subtype':
        if args.depth == None:
            print(f'mode:{args.classify_mode} needs depth param')
            exit()

    if args.loss_mode == 'LDAM' and args.constant == None:
        print(f'when loss_mode is LDAM, input Constant param')
        exit()

    if args.classify_mode == 'subtype':
        dir_name = f'subtype_classify'
        if args.fc:
            dir_name = f'fc_{dir_name}'
    elif args.leaf is not None:
        dir_name = args.classify_mode
        if args.loss_mode != 'normal':
            dir_name = f'{dir_name}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{args.constant}'
        if args.augmentation:
            dir_name = f'{dir_name}_aug'
        if args.fc:
            dir_name = f'fc_{dir_name}'
        dir_name = f'{dir_name}/depth-{args.depth}_leaf-{args.leaf}'
    else:
        dir_name = args.classify_mode
        if args.loss_mode != 'normal':
            dir_name = f'{dir_name}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{args.constant}'
        if args.augmentation:
            dir_name = f'{dir_name}_aug'
        if args.fc:
            dir_name = f'fc_{dir_name}'
        dir_name = f'{dir_name}/args.depth-{args.depth}_leaf-all'

    draw_heatmap(args, dir_name)
    save_high_low_patches(args, dir_name)
