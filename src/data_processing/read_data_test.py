import sys
import numpy as np
from glob import glob
import os
import matplotlib
from skimage import io
import matplotlib.pyplot as plt

try:
    os.chdir(os.path.join(os.getcwd(), 'src/data_processing'))
    print(os.getcwd())
except:
    pass
all_files = glob("train_out/*.npy")


for i in range(len(all_files)):
    img = np.array(io.imread("train_in/" + str(i) + ".png"))
    gt = np.load("train_out/" + str(i) + ".npy")
    r = img.shape[0]
    c = img.shape[1]
    deltar = 0
    deltac = 0
    if r < 376:
        deltar = 376 - r
    if c < 1242:
        deltac = 1242 - c
    img_pad = np.pad(img, ((0, deltar), (0, deltac)))
    gt_pad = np.pad(gt, ((0, deltar), (0, deltac)))
    io.imsave("train_pad_in/" + str(i) + ".png",img_pad)
    gt_pad = np.save("train_pad_out/" + str(i) + ".npy", gt_pad)

    # print(str(i)+" shape:\t{}, shape:\t{}".format(img_pad.shape, gt_pad.shape))
