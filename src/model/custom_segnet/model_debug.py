import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from glob import glob
from PIL import Image
import cv2

import numpy as np
import tensorflow as tf
import keras.layers as L
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from SegNetModel import SegNet
from ConvSegModel import ConvSegNet

###############################################################
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
###############################################################

class DataGenerator:
    def __init__(self, input_dir, output_dir, input_size_ori, scale, batch_size):
        self.input_size = input_size_ori
        self.scale = scale
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.all_imgs = glob(input_dir + "*.png")
        self.all_labels = glob(output_dir + "*.npy")
        self.num_batch = len(self.all_imgs) // self.batch_size

    def ChangeSize(self, gt):
        gt = cv2.resize(gt, (gt.shape[1]//self.scale, gt.shape[0]//self.scale))
        w, h = gt.shape
        # get the final input width and height
        input_w, input_h = self.input_size
        input_w = input_w // self.scale
        input_h = input_h // self.scale
        # get zero paddings sizes
        w_add = input_w - w
        h_add = input_h - h
        w_left = w_add // 2
        h_top = h_add // 2
        gt_new = np.zeros((input_w, input_h)).astype(np.float32)
        gt_new[w_left:w_left+w, h_top:h_top+h] = gt
        return gt_new

    def GetInputGt(self, img_name, gt_name):
        gt = np.load(gt_name).astype(np.float32)
        gt = self.ChangeSize(gt)
        gt = np.expand_dims(gt, axis = -1)
        gt = np.expand_dims(gt, axis = 0)

        img = np.array(Image.open(img_name)).astype(np.float32)
        # img = img / img.max()
        img = self.ChangeSize(img)
        img = np.expand_dims(img, axis = -1)
        img = np.expand_dims(img, axis = 0)
        return img, gt

    def GetBatchData(self):
        all_index = np.arange(len(self.all_imgs))
        np.random.seed(1010101)
        np.random.shuffle(all_index)
        num_batches = len(all_index) // self.batch_size
        batch_i = 0
        while True:
            if batch_i >= num_batches - 1:
                batch_i = 0
            else:
                batch_i += 1
            batch_index = all_index[batch_i*self.batch_size: (batch_i+1)*self.batch_size]
            batch_imgs = []
            batch_labels = []
            for ind in batch_index:
                img_name = self.input_dir + str(ind) + ".png"
                gt_name = self.output_dir + str(ind) + ".npy"
                img, gt = self.GetInputGt(img_name, gt_name)
                img = img / 45000.
                # if len(gt[gt == 1]) == 0:
                #     continue
                batch_imgs.append(img)
                batch_labels.append(gt)
            batch_imgs = np.concatenate(batch_imgs, axis = 0)
            batch_labels = np.concatenate(batch_labels, axis = 0)
            yield batch_imgs, batch_labels


def debug(model_name, loss_name):
    import matplotlib
    matplotlib.use("tkagg")
    import matplotlib.pyplot as plt

    batch_size = 5
    epoches = 2000
    input_size_orig = (384, 1280)
    scale = 2
    learning_rate_init = 1e-3
    model_input_size = (input_size_orig[0]//scale, input_size_orig[1]//scale)

    oneimg_name = "../../data_processing/train_in/10.png"
    onegt_name = "../../data_processing/train_out/10.npy"
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

    # choose loss function to compile model
    if loss_name == "regular":
        model.compile(loss=customModel.RegularLoss,
                optimizer=Adam(lr=learning_rate_init),
                metrics=[customModel.MetricsIOU, customModel.MetricsP, customModel.MetricsR])
    elif loss_name == "dice":
        model.compile(loss=customModel.DiceLoss,
                optimizer=Adam(lr=learning_rate_init),
                metrics=[customModel.MetricsIOU, customModel.MetricsP, customModel.MetricsR])
    elif loss_name == "weighted":
        model.compile(loss=customModel.WeightedLoss,
                optimizer=Adam(lr=learning_rate_init),
                metrics=[customModel.Jaccard, customModel.MetricsP, customModel.MetricsR])
    else:
        raise ValueError("wrong loss name input.")

    model.summary()

if __name__ == "__main__":
    """
    model_name :
        "segnet"
        "convsegnet"

    loss_name:
        "regular"
        "dice"
        "weighted"
        "jaccard"
        "MSE"
    """
    model_name = "convsegnet"
    loss_names = ["regular", "dice", "MSE", "weighted"]

    debug(model_name, "regular")
    # for i in range(len(loss_names)):
    #     loss_name = loss_names[i]
    #     main(model_name, loss_name)
    
