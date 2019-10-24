import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.nn as N
import numpy as np

class ConvSegNet:
    def __init__(self, input_size, ):
        if (isinstance(input_size, tuple) is not True) or (len(input_size) != 2):
            raise ValueError("Wrong image input size, it should be a tuple with size 2.")
        else:
            self.w, self.h = input_size
        if (self.w % 2 != 0) or (self.h % 2 != 0):
            raise ValueError("Wrong image input size, the width and height should be even numbers.")
        self.X = tf.placeholder(tf.float32, shape = (None, self.w, self.h, 1))
        self.Y = tf.placeholder(tf.float32, shape = (None, self.w, self.h, 1))
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_start = 1e-3
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        500, 0.96, staircase=True)

    def Conv2D_BN_ReLU(self, x, filters, kernel, strides, padding):
        x = L.Conv2D(filters, kernel, strides=strides, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        # x = L.LeakyReLU(alpha = 0.1)(x)
        return x
    
    def DeConv2D_BN_ReLU(self, x, filters, kernel, strides, padding):
        x = L.Conv2DTranspose(filters, kernel, strides=strides, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        # x = L.LeakyReLU(alpha = 0.1)(x)
        return x        
    
    def ConvIterateBlock(self, x, filters, num_iteration):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(2,2), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        for i in range(num_iteration):
            x = self.Conv2D_BN_ReLU(x, filters//2, [1,1], strides=(1,1), padding='same')
            x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        return x

    def DeConvIterateBlock(self, x, filters, num_iteration):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        for i in range(num_iteration):
            x = self.Conv2D_BN_ReLU(x, filters//2, [1,1], strides=(1,1), padding='same')
            x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.DeConv2D_BN_ReLU(x, filters, [3,3], strides=(2,2), padding='same')
        return x

    def FirstLayer(self, x, filters, num_iteration):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        for i in range(num_iteration):
            x = self.Conv2D_BN_ReLU(x, filters, [1,1], strides=(1,1), padding='same')
            x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        return x
    
    def LastLayer(self, x, filters, num_iteration):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        for i in range(num_iteration):
            x = self.Conv2D_BN_ReLU(x, filters, [1,1], strides=(1,1), padding='same')
            filters = filters // 2
            x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, 1, [3,3], strides=(1,1), padding='same')
        return x        

    def ConvSegBody(self):
        x = self.FirstLayer(self.X, 32, 3)
        x = self.ConvIterateBlock(x, 32, 3)
        # x = self.ConvIterateBlock(x, 64, 3)
        # x = self.ConvIterateBlock(x, 128, 3)
        # x = self.ConvIterateBlock(x, 256, 3)
        # x = self.ConvIterateBlock(x, 256, 3)

        # x = self.DeConvIterateBlock(x, 256, 3)
        # x = self.DeConvIterateBlock(x, 256, 3)
        # x = self.DeConvIterateBlock(x, 128, 3)
        # x = self.DeConvIterateBlock(x, 64, 3)
        x = self.DeConvIterateBlock(x, 32, 3)
        x = self.LastLayer(x, 32, 3)
        return x

    def SegPred(self):
        seg_out = self.ConvSegBody()
        seg_pred = tf.sigmoid(seg_out)
        return seg_pred

    def RegularLoss(self):
        pred = self.ConvSegBody()
        # logits = tf.reshape(pred, [-1,])
        # labels = tf.reshape(self.Y, [-1,])
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        # loss = tf.reduce_mean(loss)
        loss = tf.reduce_mean(tf.square(tf.sigmoid(pred) - self.Y))
        return loss

    def WeightLoss(self):
        pred = self.ConvSegBody()
        Y_all = tf.reshape(self.Y, [-1,])
        Y_fg = tf.boolean_mask(self.Y, tf.cast(self.Y, tf.bool))
        pred_fg = tf.boolean_mask(pred, tf.cast(self.Y, tf.bool))
        Y_bg = tf.boolean_mask(self.Y, tf.cast(1 - self.Y, tf.bool))
        pred_bg = tf.boolean_mask(pred, tf.cast(1 - self.Y, tf.bool))
        alpha = tf.cast(tf.shape(Y_bg)[0], tf.float32) / (tf.cast(tf.shape(Y_all)[0], tf.float32) \
                                                            + tf.constant([1e-10], dtype=tf.float32))
        loss_fg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_fg, logits=pred_fg))
        loss_bg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_bg, logits=pred_bg))

        loss_bg = tf.cond(tf.less(alpha[0], 1e-3), lambda: tf.zeros_like(loss_bg), lambda: loss_bg)
        loss_fg = tf.cond(tf.greater(alpha[0], 1 - 1e-3), lambda: tf.zeros_like(loss_fg), lambda: loss_fg)
        
        loss = alpha * loss_fg + (1 - alpha) * loss_bg
        return loss

    def IoULoss(self):
        # pred = self.ConvSegBody()
        seg_pred = self.SegPred()
        seg_pred_mask_float = tf.cast(tf.where(seg_pred > 0.5, tf.ones_like(seg_pred), \
                                                tf.zeros_like(seg_pred)), tf.float32)
        iou, _, _ = self.MaskIoU(seg_pred_mask_float, self.Y)
        loss = 1 - iou
        return loss

    def MaskIoU(self, mask1, mask2):
        mask1 = tf.reshape(mask1, [-1, ])
        mask2 = tf.reshape(mask2, [-1, ])
        mask1_bool = tf.cast(mask1, tf.bool)
        mask2_bool = tf.cast(mask2, tf.bool)
        intersection = tf.cast(tf.logical_and(mask1_bool, mask2_bool), tf.float32)
        union = tf.cast(tf.logical_or(mask1_bool, mask2_bool), tf.float32)
        iou = tf.reduce_sum(intersection) / (tf.reduce_sum(union) + 1e-10)
        acc_mask1 = tf.reduce_sum(intersection) / (tf.reduce_sum(mask1) + 1e-10)
        acc_mask2 = tf.reduce_sum(intersection) / (tf.reduce_sum(mask2) + 1e-10)
        return iou, acc_mask1, acc_mask2

    def Optimization(self):
        loss = self.RegularLoss()
        # optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        # learning_operation = optimizer.minimize(loss, global_step = self.global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
        learning_operation = optimizer.minimize(loss)
        return learning_operation

    def Metrics(self):
        seg_pred = self.SegPred()
        seg_pred_mask_float = tf.cast(tf.where(seg_pred > 0.5, tf.ones_like(seg_pred), \
                                                tf.zeros_like(seg_pred)), tf.float32)
        overall_iou, precision, recall = self.MaskIoU(seg_pred_mask_float, self.Y)
        return overall_iou, precision, recall



        

