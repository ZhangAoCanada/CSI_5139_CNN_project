import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np


model.add(L.Conv2D(256, (3, 3), strides=(1, 1), padding='same', \
                   activation='relu'))
model.add(L.Conv2D(128, (3, 3), strides=(1, 1), padding='same', \
                   activation='relu'))
model.add(L.Conv2D(64, (3, 3), strides=(1, 1), padding='same', \
                   activation='relu'))
model.add(L.Conv2D(2, (3, 3), strides=(1, 1), padding='same', \
                   activation='linear'))