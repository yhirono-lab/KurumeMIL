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

def make_dirname(args):
    if args.classify_mode == 'subtype':
        dir_name = f'subtype_classify'
        if args.fc:
            dir_name = f'fc_{dir_name}'
        if args.model != '':
            dir_name = f'{args.model}_{dir_name}'
        if args.data != '':
            if args.reduce:
                dir_name = f'reduce_{dir_name}'
            dir_name = f'{args.data}{dir_name}'
    elif args.leaf is not None:
        dir_name = args.classify_mode
        if args.loss_mode != 'normal':
            dir_name = f'{dir_name}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{args.constant}'
        if args.loss_mode == 'focal':
            dir_name = f'{dir_name}-{args.gamma}'
        if args.augmentation:
            dir_name = f'{dir_name}_aug'
        if args.fc:
            dir_name = f'fc_{dir_name}'
        if args.model != '':
            dir_name = f'{args.model}_{dir_name}'
        if args.data != '':
            if args.reduce:
                dir_name = f'reduce_{dir_name}'
            dir_name = f'{args.data}{dir_name}'
        dir_name = f'{dir_name}/depth-{args.depth}_leaf-{args.leaf}'
    else:
        dir_name = args.classify_mode
        if args.loss_mode != 'normal':
            dir_name = f'{dir_name}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            dir_name = f'{dir_name}-{args.constant}'
        if args.loss_mode == 'focal':
            dir_name = f'{dir_name}-{args.gamma}'
        if args.augmentation:
            dir_name = f'{dir_name}_aug'
        if args.fc:
            dir_name = f'fc_{dir_name}'
        if args.model != '':
            dir_name = f'{args.model}_{dir_name}'
        if args.data != '':
            if args.reduce:
                dir_name = f'reduce_{dir_name}'
            dir_name = f'{args.data}{dir_name}'
        dir_name = f'{dir_name}/depth-{args.depth}_leaf-all'
    
    return dir_name

def make_filename(args):
    if args.classify_mode == 'subtype':
        filename = f'subtype_classify'
        if args.fc:
            filename = f'fc_{filename}'
        if args.data != '':
            if args.reduce:
                filename = f'reduce_{filename}'
            filename = f'{args.data}{filename}'
    elif args.leaf is not None:
        filename = args.classify_mode
        if args.loss_mode != 'normal':
            filename = f'{filename}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            filename = f'{filename}-{args.constant}'
        if args.loss_mode == 'focal':
            filename = f'{filename}-{args.gamma}'
        if args.augmentation:
            filename = f'{filename}_aug'
        if args.fc:
            filename = f'fc_{filename}'
        if args.data != '':
            if args.reduce:
                filename = f'reduce_{filename}'
            filename = f'{args.data}{filename}'
        filename = f'{filename}_depth-{args.depth}_leaf-{args.leaf}'
    else:
        filename = args.classify_mode
        if args.loss_mode != 'normal':
            filename = f'{filename}_{args.loss_mode}'
        if args.loss_mode == 'LDAM':
            filename = f'{filename}-{args.constant}'
        if args.loss_mode == 'focal':
            filename = f'{filename}-{args.gamma}'
        if args.augmentation:
            filename = f'{filename}_aug'
        if args.fc:
            filename = f'fc_{filename}'
        if args.data != '':
            if args.reduce:
                filename = f'reduce_{filename}'
            filename = f'{args.data}{filename}'
        filename = f'{filename}_depth-{args.depth}_leaf-all'
    
    return filename

def makedir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            return