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


from model import Unet

im_width = 1280
im_height = 384
input_img = Input((im_height, im_width, 1), name='img')
unet = Unet()
model = unet.get_unet(input_img)
unet.compileModel()