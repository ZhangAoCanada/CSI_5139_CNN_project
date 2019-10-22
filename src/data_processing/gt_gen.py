# import os
# ##### set specific gpu #####
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

# import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob


def ReadInput(dir):
    """
    Function:
        Read the disparity map from the dirctory.
    """
    name_len = 6
    all_img_names = glob(dir + "/*.png")
    num_imgs = len(all_img_names)
    
    plt.ion()
    fig = plt.figure() 
    ax1 = fig.add_subplot(111)

    for  i in range(num_imgs):
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10"
        current_img = Image.open(dir + "/" + img_name + ".png")

        plt.cla()
        ax1.clear()
        ax1.imshow(current_img)
        fig.canvas.draw()
        



train_2015_input_dir = "train_2015_GANet"
train_2012_input_dir = "train_2012_GANet"

train_2015_gt_dir = "train_2015_mrcnn"
train_2012_gt_dir = "train_2012_mrcnn"


ReadInput(train_2015_input_dir)

