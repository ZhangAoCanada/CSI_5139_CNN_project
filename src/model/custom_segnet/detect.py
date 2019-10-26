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
from SegNetModel import SegNet
# from ConvSegModel import ConvSegNet