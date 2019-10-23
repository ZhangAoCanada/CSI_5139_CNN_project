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
                                                        100, 0.96, staircase=True)

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
            x = self.Conv2D_BN_ReLU(x, filters, [1,1], strides=(1,1), padding='same')
            x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        return x

    def DeConvIterateBlock(self, x, filters, num_iteration):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        for i in range(num_iteration):
            x = self.Conv2D_BN_ReLU(x, filters, [1,1], strides=(1,1), padding='same')
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
        x = self.ConvIterateBlock(x, 64, 3)
        x = self.ConvIterateBlock(x, 128, 3)
        x = self.ConvIterateBlock(x, 256, 3)
        x = self.ConvIterateBlock(x, 256, 3)

        x = self.DeConvIterateBlock(x, 256, 3)
        x = self.DeConvIterateBlock(x, 256, 3)
        x = self.DeConvIterateBlock(x, 128, 3)
        x = self.DeConvIterateBlock(x, 64, 3)
        x = self.DeConvIterateBlock(x, 32, 3)
        x = self.LastLayer(x, 32, 3)
        return x

    def WeightLoss(self):
        pred = self.ConvSegBody()
        loss = tf.sigmoid(pred)
        Y_all = tf.reshape(self.Y, [-1,])
        Y_fg = tf.boolean_mask(self.Y, tf.cast(self.Y, tf.bool))
        pred_fg = tf.boolean_mask(pred, tf.cast(self.Y, tf.bool))
        Y_bg = tf.boolean_mask(self.Y, tf.cast(1 - self.Y, tf.bool))
        pred_bg = tf.boolean_mask(pred, tf.cast(1 - self.Y, tf.bool))
        alpha = tf.shape(Y_bg)[0] / tf.shape(Y_all)[0]
        return alpha


        

