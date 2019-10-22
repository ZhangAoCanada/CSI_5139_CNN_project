# import os
# ##### set specific gpu #####
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# import matplotlib
# matplotlib.use("tkagg")
# import matplotlib.pyplot as plt

# import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
from glob import glob

def FGandBG(mrcnn_dict):
    """
    Function:
        Get forground and background masks. Note that the forground masks consist
    of the following classes:
        car
        truck
    """
    # all classes included in the mask-rcnn
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
    # get wanted infomation
    mrcnn_mask = mrcnn_dict["masks"]
    class_ids = mrcnn_dict["class_ids"]
    mask_fg = np.zeros((mrcnn_mask.shape[0], mrcnn_mask.shape[1]))
    for i in range(len(class_ids)):
        single_class_id = class_ids[i]
        single_class_name = class_names[single_class_id]
        # feel free to get as many as you want
        if single_class_name == "car" or single_class_name == "truck":
            mask_fg += mrcnn_mask[..., i]
    mask_bg = 1 - mask_fg
    return mask_fg, mask_bg


def ReadInputOutput(input_dir, gt_dir, num):
    """
    Function:
        Read the disparity map from the dirctory.
    """
    # default image name
    name_len = 6
    # read all images to count the total number of images
    all_img_names = glob(input_dir + "/*.png")
    num_imgs = len(all_img_names)

    for  i in range(num_imgs):
        # get image name
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10"
        # read image 
        current_img = Image.open(input_dir + "/" + img_name + ".png")
        # read mask
        with open(gt_dir + "/" + img_name + ".pickle", "rb") as f:
            mrcnn_result = pickle.load(f)
        mask_fg, mask_bg = FGandBG(mrcnn_result)
        mask_fg = mask_fg > 0
        mask_fg = mask_fg.astype(np.float32)

        current_img.save("test_in/" + str(num) + ".png")
        mask_fg = np.save("test_out/" + str(num) + ".npy", mask_fg)
        num += 1
    return num



train_2015_input_dir = "train_2015_GANet"
train_2012_input_dir = "train_2012_GANet"

train_2015_gt_dir = "train_2015_mrcnn"
train_2012_gt_dir = "train_2012_mrcnn"

test_2015_input_dir = "test_2015_GANet"
test_2012_input_dir = "test_2012_GANet"

test_2015_gt_dir = "test_2015_mrcnn"
test_2012_gt_dir = "test_2012_mrcnn"

num_start = 0

num_middle = ReadInputOutput(test_2015_input_dir, test_2015_gt_dir, num_start)
num_final = ReadInputOutput(test_2012_input_dir, test_2012_gt_dir, num_middle)


