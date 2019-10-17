from __future__ import print_function
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
#from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from models.GANet_deep import GANet
from dataloader.data import get_test_set
import numpy as np
from models.GANet_deep import GANet
from glob import glob
from tqdm import tqdm

# feel free to tune this
max_disp = 192
pre_trained_weights = "./checkpoint/kitti2015_final.pth"
crop_height = 384
crop_width = 1248


if not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building model')
# build model
model = GANet(max_disp)
# GPU option
model = torch.nn.DataParallel(model).cuda()
# read pre-trained weights
if os.path.isfile(pre_trained_weights):
    print("=> loading checkpoint '{}'".format(pre_trained_weights))
    checkpoint = torch.load(pre_trained_weights)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    print("=> no checkpoint found at '{}'".format(pre_trained_weights))

def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]	
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test(leftname, rightname, savename):
  #  count=0
    
    input1, input2, height, width = test_transform(load_data(leftname, rightname), crop_height, crop_width)

    
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    input1 = input1.cuda()
    input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)
     
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= crop_height and width <= crop_width:
        temp = temp[0, crop_height - height: crop_height, crop_width - width: crop_width]
    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))

def PredDisp(kitti_dir, save_dir):
    left_img_dir = kitti_dir + "colored_0/"
    right_img_dir = kitti_dir + "colored_1/"

    name_len = 6

    all_img_names = glob(left_img_dir + "*10.png")
    total_disparity_num = len(all_img_names)

    for i in tqdm(range(total_disparity_num)):
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10.png"

        left_img_name = left_img_dir + img_name
        right_img_name = right_img_dir + img_name
        save_name = save_dir + img_name
        test(left_img_name, right_img_name, save_name)


if __name__ == "__main__":
    kitti_dir = "/home/azhang/Documents/kitti/2012/training/"
    save_dir = "./result/"

    PredDisp(kitti_dir, save_dir)