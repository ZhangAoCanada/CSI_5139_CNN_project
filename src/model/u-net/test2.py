import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm


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
    for i in tqdm(range(len(all_files))):
        img = np.array(
            imread("./src/data_processing/test_in/" + str(i) + ".png"))
        img_pad = pad_data(img)
        img_pad = resize_data(img_pad)
        img_list.append(img_pad)
    return np.asfarray(img_list)


def read_gt(filepath, width=1280, height=384):
    gt_list = []
    all_files = glob(filepath+"/*npy")
    for i in tqdm(range(len(all_files))):
        gt = np.load("./src/data_processing/test_out/" + str(i) + ".npy")
        gt_pad = pad_data(gt)
        gt_pad = resize_data(gt_pad)
        gt_list.append(gt_pad)
    return np.asfarray(gt_list)


def main():
    # Load testing set into RAM
    img_list = read_img("./src/data_processing/test_in")
    gt_list = read_gt("./src/data_processing/test_out")

    print("Img List shape={}\tGt List shaple={}".format(
        img_list.shape, gt_list.shape))
    print("Img's shape={}\t Gt's shaple={}".format(
        img_list[0].shape, gt_list[0].shape))

    # Load model file and predict on testing set
    model = load_model("./model_folder/model-2019-10.h5")
    preds_test = model.predict(img_list, verbose=1)

    # Threshold predictions
    preds_test = (preds_test > 0.5).astype(np.uint8)
    print(preds_test.shape)

    # calculate IOU
    intersection = np.logical_and(gt_list, preds_test)
    union = np.logical_or(gt_list, preds_test)
    iou_score = np.sum(intersection) / np.sum(union)

    print("IOU = {}".format(iou_score))

    # Select random index from testing set and show.
    for i in range(5):
        ix = random.randint(0, 300)
        has_mask = preds_test[ix].max() > 0
        fig, ax = plt.subplots(2, 2)
        print("show img {} and its mask".format(ix))
        ax[0, 0].imshow(img_list[ix, ..., 0], cmap='seismic')
        if has_mask:
            ax[0, 0].contour(preds_test[ix].squeeze(),
                             colors='k', levels=[0.5])
        ax[0, 0].set_title('Seismic')

        ax[0, 1].imshow(preds_test[ix].squeeze(),  cmap='gray')
        ax[0, 1].set_title('test')

        ax[1, 0].imshow(img_list[ix, ..., 0], cmap='seismic')
        if has_mask:
            ax[1, 0].contour(gt_list[ix].squeeze(), colors='k', levels=[0.5])
        ax[1, 0].set_title('Seismic')

        ax[1, 1].imshow(gt_list[ix].squeeze(),  cmap='gray')
        ax[1, 1].set_title('ground truth')
        show()


main()
