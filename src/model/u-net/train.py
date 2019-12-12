import os
import random
from glob import glob
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import concatenate_images, imread, imshow, show
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout,
                                     GlobalMaxPool2D, Input, Lambda,
                                     MaxPooling2D, RepeatVector, Reshape, add,
                                     concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  array_to_img, img_to_array,
                                                  load_img)
from tqdm import tnrange, tqdm_notebook

from model import Unet


def pad_data(data, width=1280, height=384):
    r = data.shape[0]
    c = data.shape[1]

    deltar = 0
    deltac = 0
    if r < height:
        deltar = height - r
    if c < width:
        deltac = width - c
    data_pad = np.pad(data, ((0, deltar), (0, deltac)))
    return data_pad


def resize_data(data, width=1280, height=384):
    data = resize(data, (height, width, 1),
                  mode='constant', preserve_range=True)
    return data


def read_img(filepath, width=1280, height=384):
    img_list = []
    all_files = glob(filepath + "/*png")
    for i in range(len(all_files)):
        img = np.array(
            imread("./src/data_processing/train_in/" + str(i) + ".png"))
        img_pad = pad_data(img)
        img_pad = resize_data(img_pad)
        img_list.append(img_pad)
    return np.asfarray(img_list)


def read_gt(filepath, width=1280, height=384):
    gt_list = []
    all_files = glob(filepath+"/*npy")
    for i in range(len(all_files)):
        gt = np.load("./src/data_processing/train_out/" + str(i) + ".npy")
        gt_pad = pad_data(gt)
        gt_pad = resize_data(gt_pad)
        gt_list.append(gt_pad)
    return np.asfarray(gt_list)


img_list = read_img("./src/data_processing/train_in")
gt_list = read_gt("./src/data_processing/train_out")

print("Img List shape={}\tGt List shaple={}".format(
    img_list.shape, gt_list.shape))
print("Img's shape={}\t Gt's shaple={}".format(
    img_list[0].shape, gt_list[0].shape))

# Test padding IMG and Masks
#######################################################

# Select random index from trainning set.
ix = random.randint(0, len(img_list))
has_mask = gt_list[ix].max() > 0
fig, ax = plt.subplots(2, 1)
print("show img {} and its mask".format(ix))
ax[0].imshow(img_list[ix, ..., 0], cmap='seismic')
if has_mask:
    ax[0].contour(gt_list[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Seismic')

ax[1].imshow(gt_list[ix].squeeze(),  cmap='gray')
ax[1].set_title('Salt')
show()
#######################################################

# Split training data into valid and train sets
X_train, X_valid, y_train, y_valid = train_test_split(
    img_list, gt_list, test_size=0.15, random_state=2019)
print("X_train: {}\ty_train: {}\tX_valid: {}\ty_valid: {}\t".format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))

im_width = 1280
im_height = 384
input_img = Input((im_height, im_width, 1), name='img')
unet = Unet()
model = unet.get_unet(input_img)
lossNames = ["CE", "MSE", "Dice", "Weighted"]
unet.compileModel(lossNames[0])

log_dir = "./logs_00"
callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1,
                    save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir=log_dir, update_freq='batch')
]

results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))
