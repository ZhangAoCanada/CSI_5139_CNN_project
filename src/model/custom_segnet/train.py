import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import matplotlib
# matplotlib.use("tkagg")
# import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
import cv2

import numpy as np
import tensorflow as tf
import keras.layers as L
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from model2 import ConvSegNet

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
        img = self.ChangeSize(img)
        img = np.expand_dims(img, axis = -1)
        img = np.expand_dims(img, axis = 0)
        return img, gt

    def GetBatchData(self):
        all_index = np.arange(len(self.all_imgs))
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
                batch_imgs.append(img)
                batch_labels.append(gt)
            batch_imgs = np.concatenate(batch_imgs, axis = 0)
            batch_labels = np.concatenate(batch_labels, axis = 0)
            yield batch_imgs, batch_labels


batch_size = 5
epoches = 100
input_size_orig = (384, 1280)
scale = 2
model_input_size = (input_size_orig[0]//scale, input_size_orig[1]//scale)
train_input_dir = "../../data_processing/train_in/"
train_output_dir = "../../data_processing/train_out/"
test_input_dir = "../../data_processing/test_in/"
test_output_dir = "../../data_processing/test_out/"
log_dir = 'logs/000/'

###############################################################
oneimg_name = "../../data_processing/train_in/10.png"
onegt_name = "../../data_processing/train_out/10.npy"
oneimg, onegt = GetInputGt(oneimg_name, onegt_name, input_size_orig, scale)
oneimg /= oneimg.max()
print([oneimg.shape, onegt.shape])
###############################################################

trainGo = DataGenerator(train_input_dir, train_output_dir, input_size_orig, scale, batch_size)
testGo = DataGenerator(test_input_dir, test_output_dir, input_size_orig, scale, batch_size)

# GPU memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
# build the model
convsegModel = ConvSegNet(model_input_size)
model = convsegModel.ConvSegBody()
# set tensorboard
logging = TensorBoard(log_dir=log_dir)
# set check point
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
            monitor='loss', save_weights_only=True, save_best_only=True, period=3)
# set learning rate reduce
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
# get all necessary loss and performances
model.compile(loss=convsegModel.RegularLoss,
              optimizer='adam',
              metrics=[convsegModel.MetricsIOU, convsegModel.MetricsP, convsegModel.MetricsR])
# fit data
model.fit_generator(trainGo.GetBatchData(), 
                    steps_per_epoch=trainGo.num_batch,
                    validation_data=testGo.GetBatchData(),
                    validation_steps=testGo.num_batch,
                    epochs=500,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint])

# pred = model.predict(oneimg)
# pred1 = pred > 0.5
# pred1 = pred1.astype(np.float32)

# fig = plt.figure()

# ax1 = fig.add_subplot(311)
# ax2 = fig.add_subplot(312)
# ax3 = fig.add_subplot(313)
# oneimg = np.squeeze(oneimg)
# onegt = np.squeeze(onegt)
# pred = np.squeeze(pred)
# pred1 = np.squeeze(pred1)
# ax1.imshow(oneimg)
# ax2.imshow(pred)
# ax3.imshow(pred1)
# plt.show()


# convsegModel = ConvSegNet(model_input_size)

# loss = convsegModel.RegularLoss()
# # loss = convsegModel.WeightLoss()
# # loss = convsegModel.IoULoss()

# training = convsegModel.Optimization()
# # overall, precision, recall = convsegModel.Metrics()

# # summary_loss = tf.summary.scalar("loss", loss)
# # streaming_overall, streaming_overall_update = tf.contrib.metrics.streaming_mean(overall)
# # streaming_precision, streaming_precision_update = tf.contrib.metrics.streaming_mean(precision)
# # streaming_recall, streaming_recall_update = tf.contrib.metrics.streaming_mean(recall)
# # summary_overall = tf.summary.scalar("iou", streaming_overall)
# # summary_precision = tf.summary.scalar("precision", streaming_precision)
# # summary_recall = tf.summary.scalar("recall", streaming_recall)

# # initialization
# init = tf.global_variables_initializer()

# # GPU settings
# gpu_options = tf.GPUOptions(allow_growth=True)
# print(K.learning_phase())

# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#     K.tensorflow_backend.set_session(sess)
#     # summaries_train = 'logs/train/'
#     # summaries_test = 'logs/test/'
#     # train_writer = tf.summary.FileWriter(summaries_train + "original_loss", sess.graph)
#     # test_writer = tf.summary.FileWriter(summaries_test + "original_loss", sess.graph)
#     sess.run(init)
#     with sess.as_default():
#         for i in range(1000):
#             l, _ = sess.run([loss, training], feed_dict = {convsegModel.X: oneimg,
#                                                             convsegModel.Y: onegt,})
#                                                             # K.learning_phase(): 1})
#             print("-------------------------------")
#             print("loss:", l)
#     # for batch_imgs, batch_gts in dataGo.GetBatchData():
#     #     test1 = sess.run(overall, feed_dict = {convsegModel.X: batch_imgs,
#     #                                         convsegModel.Y: batch_gts})




###############################################################
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# ax1.imshow(oneimg[0, :, :, 0])
# ax2.imshow(onegt[0, :, :, 0])
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