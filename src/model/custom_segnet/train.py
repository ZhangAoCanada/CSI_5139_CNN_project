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

class DataGenerator:
    def __init__(self, input_dir, output_dir, input_size_ori, scale, batch_size):
        self.input_size = input_size_ori
        self.scale = scale
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.all_imgs = glob(input_dir + "*.png")
        self.all_labels = glob(output_dir + "*.npy")

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
        img = self.ChangeSize(img)
        img = np.expand_dims(img, axis = -1)
        img = np.expand_dims(img, axis = 0)
        return img, gt

    def GetBatchData(self):
        all_index = np.arange(len(self.all_imgs))
        np.random.shuffle(all_index)
        num_batches = len(all_index) // self.batch_size
        for batch_i in range(num_batches):
            batch_index = all_index[batch_i*self.batch_size: (batch_i+1)*self.batch_size]
            batch_imgs = []
            batch_labels = []
            for ind in batch_index:
                img_name = self.input_dir + str(ind) + ".png"
                gt_name = self.output_dir + str(ind) + ".npy"
                img, gt = self.GetInputGt(img_name, gt_name)
                batch_imgs.append(img)
                batch_labels.append(gt)
            batch_imgs = np.concatenate(batch_imgs, axis = 0)
            batch_labels = np.concatenate(batch_labels, axis = 0)
            yield batch_imgs, batch_labels


batch_size = 2
epoches = 100
input_size_orig = (384, 1280)
scale = 2
model_input_size = (input_size_orig[0]//scale, input_size_orig[1]//scale)
input_dir = "../../data_processing/train_in/"
output_dir = "../../data_processing/train_out/"

dataGo = DataGenerator(input_dir, output_dir, input_size_orig, scale, batch_size)

convsegModel = ConvSegNet(model_input_size)

t1 = convsegModel.ConvSegBody()
# l = convsegModel.RegularLoss()
# l = convsegModel.WeightLoss()
l = convsegModel.IoULoss()

# initialization
init = tf.global_variables_initializer()

# GPU settings
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for batch_imgs, batch_gts in dataGo.GetBatchData():
        test1 = sess.run(t1, feed_dict = {convsegModel.X: batch_imgs,
                                            convsegModel.Y: batch_gts})
        print(test1.shape)




###############################################################
# i = 60
# img_name = "../../data_processing/train_in/" + str(i) + ".png"
# gt_name = "../../data_processing/train_out/" + str(i) + ".npy"
# img, gt = GetInputGt(img_name, gt_name)

# img = np.squeeze(img)
# gt = np.squeeze(gt)

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax1.imshow(img)
# ax2.imshow(gt)
# plt.show()
###############################################################


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