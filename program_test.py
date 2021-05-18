import os
import csv
from PIL import Image, ImageStat
import numpy as np
import cv2
import openslide

from tqdm import tqdm

print(os.path.exists('/Dataset/Kurume_Dataset/svs_info/180637/180637.csv'))
pos = np.loadtxt(f'/Dataset/Kurume_Dataset/svs_info/180637/180637.csv', delimiter=',', dtype='int')
