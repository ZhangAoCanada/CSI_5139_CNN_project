import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from glob import glob
from PIL import Image
import cv2
import skimage.io
import skimage.color
from skimage.util import img_as_ubyte

import numpy as np
import tensorflow as tf
import keras.layers as L
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from SegNetModel import SegNet
from ConvSegModel import ConvSegNet

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon


def ChangeSize(gt, input_size_orig, scale):
    gt = cv2.resize(gt, (gt.shape[1]//scale, gt.shape[0]//scale))
    w, h = gt.shape
    # get the final input width and height
    input_w, input_h = input_size_orig
    input_w = input_w // scale
    input_h = input_h // scale
    # get zero paddings sizes
    w_add = input_w - w
    h_add = input_h - h
    w_left = w_add // 2
    h_top = h_add // 2
    gt_new = np.zeros((input_w, input_h)).astype(np.float32)
    gt_new[w_left:w_left+w, h_top:h_top+h] = gt
    return gt_new

def GetInputGt(img_name, gt_name, input_size_orig, scale):
    gt = np.load(gt_name).astype(np.float32)
    gt = ChangeSize(gt, input_size_orig, scale)
    gt = np.expand_dims(gt, axis = -1)
    gt = np.expand_dims(gt, axis = 0)

    img = np.array(Image.open(img_name)).astype(np.float32)
    img = ChangeSize(img, input_size_orig, scale)
    img = np.expand_dims(img, axis = -1)
    img = np.expand_dims(img, axis = 0)
    return img, gt

def TransferPred(pred):
    pred1 = pred > 0.5
    pred1 = pred1.astype(np.float32)
    return pred1

def RandomColors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def ApplyMask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    colors = RandomColors(1)
    color = colors[0]
    masked_image = image.astype(np.uint32).copy()
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def main(model_name, loss_name, test_img_num):
    batch_size = 5
    epoches = 2000
    input_size_orig = (384, 1280)
    scale = 2
    model_input_size = (input_size_orig[0]//scale, input_size_orig[1]//scale)

    oneimg_name = "../../data_processing/test_in/" + str(test_img_num) + ".png"
    onegt_name = "../../data_processing/test_out/" + str(test_img_num) + "10.npy" 
    log_dir = 'logs2/' + 'newdatalr_' + model_name + '_' + loss_name + "/"
    oneimg, onegt = GetInputGt(oneimg_name, onegt_name, input_size_orig, scale)
    oneimg /= 45000

    # GPU memory management
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))

    # build the model
    if model_name == "segnet":
        customModel = SegNet(model_input_size)
        model = customModel.SegBody()
    elif model_name == "convsegnet":
        customModel = ConvSegNet(model_input_size)
        model = customModel.ConvSegBody()
    else:
        raise ValueError("wrong model name input.")

    model.load_weights(log_dir + 'trained_weights_stage_1.h5')
    pred = model.predict(oneimg)
    pred = TransferPred(pred)

    test_image = np.squeeze(oneimg)
    mask_pred = np.squeeze(pred)
    test_mask = np.squeeze(onegt)

    test_image = skimage.color.gray2rgb(test_image)
    test_image = img_as_ubyte(test_image)

    masked_image_pred = ApplyMask(test_image, mask_pred)
    masked_image_gt = ApplyMask(test_image, test_mask)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.imshow(masked_image_gt)
    ax2.imshow(masked_image_pred)
    plt.show()

if __name__ == "__main__":
    """
    model_name :
        "segnet"
        "convsegnet"

    loss_name:
        "regular"
        "dice"
        "weighted"
        "MSE"
    """
    model_names = ["segnet", "convsegnet"]
    loss_names = ["regular", "dice", "MSE", "weighted"]

    model_name = model_names[0]
    loss_name = loss_names[0]

    main(model_name, loss_name, 10)
