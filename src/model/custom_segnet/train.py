import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# import matplotlib
# matplotlib.use("tkagg")
# import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
import cv2

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from model import ConvSegNet

input_size_orig = (384, 1280)
scale = 2

input_size = (input_size_orig[0]//scale, input_size_orig[1]//scale)

# def ChangeImageSize(img, input_size, scale):
#     """
#     Function:
#         Get the input image into the standard size for the input of the model.
#     """
#     # resize the image according to the scale
#     img = np.array(img.resize((img.size[0]//scale, img.size[1]//scale)))
#     # get the resized image shape
#     w, h = img.shape
#     # get the final input width and height
#     input_w, input_h = input_size
#     input_w = input_w // scale
#     input_h = input_h // scale
#     # get zero paddings sizes
#     w_add = input_w - w
#     h_add = input_h - h
#     w_left = w_add // 2
#     h_top = h_add // 2
#     # pad the image to the standard size
#     img_new = np.zeros((input_w, input_h)).astype(np.float32)
#     img_new[w_left:w_left+w, h_top:h_top+h] = img
#     return img_new

def ChangeSize(gt, input_size, scale):
    """
    Function:
        Get the input image into the standard size for the input of the model.
    """
    gt = cv2.resize(gt, (gt.shape[1]//2, gt.shape[0]//2))
    w, h = gt.shape
    # get the final input width and height
    input_w, input_h = input_size
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

all_ims = glob("../../data_processing/train_in/20.png")
all_las = glob("../../data_processing/train_out/20.npy")

###############################################################
# plt.ion()
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# for i in range(len(all_ims)):
#     img = np.array(Image.open("../../data_processing/train_in/" + str(i) + ".png")).astype(np.float32)
#     img = ChangeSize(img, input_size_orig, scale)
#     gt = np.load("../../data_processing/train_out/" + str(i) + ".npy")
#     gt = ChangeSize(gt, input_size_orig, scale)

#     plt.cla()
#     ax1.clear()
#     ax1.imshow(img)
#     ax2.clear()
#     ax2.imshow(gt)
#     fig.canvas.draw()
#     plt.pause(0.3)
###############################################################

gt = np.load(all_las[0]).astype(np.float32)
gt = ChangeSize(gt, input_size_orig, scale)
gt = np.expand_dims(gt, axis = -1)
gt = np.expand_dims(gt, axis = 0)

img = np.array(Image.open(all_ims[0])).astype(np.float32)
img = ChangeSize(img, input_size_orig, scale)

img = np.expand_dims(img, axis = -1)
img = np.expand_dims(img, axis = 0)

convsegModel = ConvSegNet(input_size)

t1 = convsegModel.ConvSegBody()
l1 = convsegModel.WeightLoss()

# initialization
init = tf.global_variables_initializer()

# GPU settings
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    test1 = sess.run(l1, feed_dict = {convsegModel.X: img,
                                        convsegModel.Y: gt})
    print(test1)