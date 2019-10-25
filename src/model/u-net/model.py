import os
import random
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.io import concatenate_images, imread, imshow
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout,
                                     GlobalMaxPool2D, Input, Lambda,
                                     MaxPooling2D, RepeatVector, Reshape, add,
                                     concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  array_to_img, img_to_array,
                                                  load_img)
from tqdm import tnrange, tqdm_notebook

plt.style.use("ggplot")


class Unet(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, BN=True):
        # First Layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer="he_normal", padding="same")(input_tensor)
        if BN:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Second Layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer="he_normal", padding="same")(input_tensor)
        if BN:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def get_unet(self, input_img, n_filters=12, dropout=0.5, BN=True):
        c1 = self.conv2d_block(input_img, n_filters*1, kernel_size=3, BN=BN)
        p1 = MaxPooling2D((2, 2))(c1)
        # p1 = Dropout(dropout)(p1)

        c2 = self.conv2d_block(p1, n_filters*2, kernel_size=3, BN=BN)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters*4, kernel_size=3, BN=BN)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters*8, kernel_size=3, BN=BN)
        p4 = MaxPooling2D((2, 2))(c4)
        # p4 = Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters*16, kernel_size=3, BN=BN)

        u6 = Conv2DTranspose(n_filters*8, (3, 3),
                             strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters *
                               8,  kernel_size=3, BN=BN)

        u7 = Conv2DTranspose(n_filters*4, (3, 3),
                             strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        # u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters *
                               4,  kernel_size=3, BN=BN)

        u8 = Conv2DTranspose(n_filters*2, (3, 3),
                             strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters *
                               2,  kernel_size=3, BN=BN)

        u9 = Conv2DTranspose(n_filters*1, (3, 3),
                             strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        # u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters,  kernel_size=3, BN=BN)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        self.model = model
        return model

    def compileModel(self):
        self.model.compile(optimizer=Adam(), loss="binary_crossentropy",
                           metrics=["accuracy"])
        self.model.summary() 