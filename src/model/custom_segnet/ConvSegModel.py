import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import keras.layers as L
import keras.backend as K
import keras.metrics as M
from keras.models import Model
import numpy as np

class ConvSegNet:
    def __init__(self, input_size, ):
        if (isinstance(input_size, tuple) is not True) or (len(input_size) != 2):
            raise ValueError("Wrong image input size, it should be a tuple with size 2.")
        else:
            self.w, self.h = input_size
        if (self.w % 2 != 0) or (self.h % 2 != 0):
            raise ValueError("Wrong image input size, the width and height should be even numbers.")
        self.X = L.Input(shape = (self.w, self.h, 1))

    def Conv2D_BN_ReLU(self, x, filters, kernel, strides, padding):
        x = L.Conv2D(filters, kernel, strides=strides, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        return x

    def GatingConv2D(self, x, filters, kernel, strides, padding):
        x = L.Conv2D(filters, kernel, strides=strides, padding=padding, activation='sigmoid')(x)
        return x
    
    def DeConv2D_BN_ReLU(self, x, filters, kernel, strides, padding):
        x = L.Conv2DTranspose(filters, kernel, strides=strides, padding=padding)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        # x = L.LeakyReLU(alpha = 0.1)(x)
        return x        

    def GatingConv2D(self, x, filters, kernel, strides, padding):
        x = L.Conv2D(filters, kernel, strides=strides, padding=padding, activation='sigmoid')(x)
        return x
    
    def ConvTwoBlock(self, x, filters):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(2,2), padding='same')
        return x
    
    def ConvThreeBlock(self, x, filters):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(2,2), padding='same')
        return x

    def DeConvThreeBlock(self, x ,filters, final_filters):
        x = self.DeConv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.DeConv2D_BN_ReLU(x, final_filters, [3,3], strides=(1,1), padding='same')
        x = self.DeConv2D_BN_ReLU(x, final_filters, [3,3], strides=(2,2), padding='same')
        return x

    def DeConvTwoBlock(self, x ,filters, final_filters):
        x = self.DeConv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.DeConv2D_BN_ReLU(x, final_filters, [3,3], strides=(2,2), padding='same')
        return x
    
    def LastLayer(self, x ,filters, final_filters):
        x = self.Conv2D_BN_ReLU(x, filters, [3,3], strides=(1,1), padding='same')
        x = self.GatingConv2D(x, final_filters, [1,1], strides=(1,1), padding='same')
        return x

    def ConvSegBody(self):
        x = self.ConvThreeBlock(self.X, 64)
        x = self.ConvThreeBlock(x, 128)
        x = self.ConvThreeBlock(x, 256)
        x = self.ConvThreeBlock(x, 512)
        # x = self.ConvThreeBlock(x, 512)

        # x = self.DeConvThreeBlock(x, 512, 512)
        x = self.DeConvThreeBlock(x, 512, 512)
        x = self.DeConvThreeBlock(x, 512, 256)
        x = self.DeConvThreeBlock(x, 256, 128)
        x = self.DeConvThreeBlock(x, 128, 64)
        x = self.LastLayer(x, 64, 1)
        return Model(self.X, x)
    
    def RegularLoss(self, y_true, y_pred):
        logits = K.reshape(y_pred, [-1,])
        labels = K.reshape(y_true, [-1,])
        loss = K.binary_crossentropy(target=labels, output=logits)
        loss = K.mean(loss)
        return loss

    def MSE(self, y_true, y_pred):
        loss = K.mean(K.square(y_true - y_pred))
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
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true + y_pred)
        return 1 - (numerator + 1) / (denominator + 1)

    def Jaccard(self, y_true, y_pred):
        numerator = K.sum(y_true * y_pred)
        denominator = K.sum(y_true + y_pred - y_true * y_pred)
        return (numerator + 1) / (denominator + 1)

    def TransferToMask(self, arr):
        arr_reshape = K.reshape(arr, [-1,])
        arr_bool = K.greater(arr_reshape, 0.5)
        arr_bool_float = K.cast(arr_bool, 'float32')
        return arr_bool_float

    def MaskIoU(self, mask1, mask2):
        intersection = mask1 * mask2
        union = mask2  + mask2 - intersection
        iou = tf.reduce_mean(intersection) / (tf.reduce_mean(union) + 1e-5)
        acc_mask1 = tf.reduce_mean(intersection) / (tf.reduce_mean(mask1) + 1e-5)
        acc_mask2 = tf.reduce_mean(intersection) / (tf.reduce_mean(mask2) + 1e-5)
        return iou, acc_mask1, acc_mask2

    def MetricsIOU(self, y_true, y_pred):
        iou_fg = self.Jaccard(y_true, y_pred)
        iou_bg = self.Jaccard(1.-y_true, 1.-y_pred)
        overall_iou = iou_fg * iou_bg
        return overall_iou

    def MetricsP(self, y_true, y_pred):
        y_true_mask = self.TransferToMask(y_true)
        y_pred_mask = self.TransferToMask(y_pred)
        overall_iou, precision, recall = self.MaskIoU(y_pred_mask, y_true_mask)
        return precision

    def MetricsR(self, y_true, y_pred):
        y_true_mask = self.TransferToMask(y_true)
        y_pred_mask = self.TransferToMask(y_pred)
        overall_iou, precision, recall = self.MaskIoU(y_pred_mask, y_true_mask)
        return recall