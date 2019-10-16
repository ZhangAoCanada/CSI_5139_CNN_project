import sys
sys.path.insert(0, '../model/mrcnn')
import random
import math
import numpy as np
import skimage.io
import skimage.color
import matplotlib
matplotlib.use('tkagg')
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

from glob import glob
from tqdm import tqdm

Target_dir = "train_2015_GANet_color/"

name_len = 6

all_img_names = glob(Target_dir + "*.png")
total_disparity_num = len(all_img_names)

wield_number = 0


for i in tqdm(range(wield_number, total_disparity_num)):
    img_count = str(i)
    zero_len = name_len - len(img_count)
    img_name = (zero_len * "0") + img_count + "_10.png"

    GANet_img = skimage.io.imread(Target_dir + img_name)

    print(GANet_img.shape)