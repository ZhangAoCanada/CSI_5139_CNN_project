import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import keras.layers as L
import keras.backend as K
import keras.metrics as M
import keras.regularizers as R
from keras.models import Model
import numpy as np

class SegNet:
    def __init__(self, input_size, ):
        if (isinstance(input_size, tuple) is not True) or (len(input_size) != 2):
            raise ValueError("Wrong image input size, it should be a tuple with size 2.")
        else:
            self.w, self.h = input_size
        if (self.w % 2 != 0) or (self.h % 2 != 0):
            raise ValueError("Wrong image input size, the width and height should be even numbers.")
        self.X = L.Input(shape = (self.w, self.h, 1))

    def Conv2D_BN_ReLU(self, x, filters, kernel, strides, padding):
        # kernel_regularizer = R.l2(5e-4)
        # bias_regularizer = R.l2(5e-4)
        kernel_regularizer = None
        bias_regularizer = None
        x = L.Conv2D(filters, kernel, strides=strides, padding=padding, \
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        return x

    def GatingConv2D(self, x, filters, kernel, strides, padding):
        # kernel_regularizer = R.l2(5e-4)
        # bias_regularizer = R.l2(5e-4)
        kernel_regularizer = None
        bias_regularizer = None
        x = L.Conv2D(filters, kernel, strides=strides, padding=padding, activation='sigmoid', \
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(x)
        return x

    def MaxPool(self, x):
        x = L.MaxPooling2D()(x)
        return x

    def UpSampling(self, x):
        x = L.UpSampling2D()(x)
        return x      

    def EncodeTwoBlock(self, x, filters):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.MaxPool(x)
        return x
    
    def EncodeThreeBlock(self, x, filters):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.MaxPool(x)
        return x

    def DecodeThreeBlock(self, x ,filters, final_filters):
        x = self.UpSampling(x)
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, final_filters, [3,3], strides=(1,1), padding='same')
        return x

    def DecodeTwoBlock(self, x ,filters, final_filters):
        x = self.UpSampling(x)
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, final_filters, [3,3], strides=(1,1), padding='same')
        return x
    
    def LastLayer(self, x ,filters, final_filters):
        x = self.UpSampling(x)
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.GatingConv2D(x, final_filters, [1,1], strides=(1,1), padding='same')
        return x

    def SegBody(self):
        x = self.EncodeTwoBlock(self.X, 64)
        x = self.EncodeTwoBlock(x, 128)
        x = self.EncodeThreeBlock(x, 256)
        x = self.EncodeThreeBlock(x, 512)
        x = self.EncodeThreeBlock(x, 512)

        x = self.DecodeThreeBlock(x, 512, 512)
        x = self.DecodeThreeBlock(x, 512, 256)
        x = self.DecodeThreeBlock(x, 256, 128)
        x = self.DecodeTwoBlock(x, 128, 64)
        x = self.LastLayer(x, 64, 1)
        return Model(self.X, x)

    def RegularLoss(self, y_true, y_pred):
        logits = K.reshape(y_pred, [-1,])
        labels = K.reshape(y_true, [-1,])
        loss = K.binary_crossentropy(target=labels, output=logits)
        loss = K.mean(loss)
        return loss

    def WeightedLoss(self, y_true, y_pred):
        one_weight = 0.89
        zero_weight = 0.11
        logits = K.reshape(y_pred, [-1,])
        labels = K.reshape(y_true, [-1,])
        loss = K.binary_crossentropy(target=labels, output=logits)
        weight_vector = labels * one_weight + (1.-labels) * zero_weight
        loss = weight_vector * loss
        loss = K.mean(loss)
        return loss

    def DiceLoss(self, y_true, y_pred):
        numerator = 2 * K.sum(y_true * y_pred, axis=-1)
        denominator = K.sum(y_true + y_pred, axis=-1)
        return 1 - (numerator + 1) / (denominator + 1)

    def MaskIoU(self, mask1, mask2):
        # mask1 = K.reshape(mask1, [-1, ])
        # mask2 = K.reshape(mask2, [-1, ])
        mask1 = tf.squeeze(mask1)
        mask2 = tf.squeeze(mask2)
        mask1_bool = tf.cast(mask1, tf.bool)
        mask2_bool = tf.cast(mask2, tf.bool)
        intersection = tf.cast(tf.logical_and(mask1_bool, mask2_bool), tf.float32)
        union = tf.cast(tf.logical_or(mask1_bool, mask2_bool), tf.float32)
        # union = K.cast(K.any(K.stack([mask1, mask2], axis=0), axis=0), 'float32')
        # intersection = K.cast(K.all(K.stack([mask1, mask2], axis=0), axis=0), 'float32')
        iou = tf.reduce_mean(intersection) / (tf.reduce_mean(union) + 1e-6)
        acc_mask1 = tf.reduce_mean(intersection) / (tf.reduce_mean(mask1) + 1e-6)
        acc_mask2 = tf.reduce_mean(intersection) / (tf.reduce_mean(mask2) + 1e-6)
        return iou, acc_mask1, acc_mask2

    def MetricsIOU(self, y_true, y_pred):
        y_pred_mask = tf.where(y_pred > 0.5, tf.ones_like(y_pred), \
                                                tf.zeros_like(y_pred))
        # y_pred_mask = tf.cast(tf.greater(y_pred, 0.5 * tf.ones(tf.shape(y_pred))), tf.float32)
        overall_iou, precision, recall = self.MaskIoU(y_pred_mask, y_true)
        return overall_iou

    def MetricsP(self, y_true, y_pred):
        y_pred_mask = tf.cast(tf.where(y_pred > 0.5, tf.ones_like(y_pred), \
                                                tf.zeros_like(y_pred)), tf.float32)
        # y_pred_mask = tf.cast(tf.greater(y_pred, 0.5 * tf.ones(tf.shape(y_pred))), tf.float32)
        overall_iou, precision, recall = self.MaskIoU(y_pred_mask, y_true)
        return precision

    def MetricsR(self, y_true, y_pred):
        y_pred_mask = tf.cast(tf.where(y_pred > 0.5, tf.ones_like(y_pred), \
                                                tf.zeros_like(y_pred)), tf.float32)
        # y_pred_mask = tf.cast(tf.greater(y_pred, 0.5 * tf.ones(tf.shape(y_pred))), tf.float32)
        overall_iou, precision, recall = self.MaskIoU(y_pred_mask, y_true)
        return recall
    



        

