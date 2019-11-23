"""
Combine kitti stereo data with Yolov3 detection.
"""
import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(0, '../model/mrcnn')
import random
import math
import numpy as np
import skimage.io
import skimage.color
import matplotlib
matplotlib.use('tkagg')
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

from skimage.util import img_as_ubyte

from glob import glob
from tqdm import tqdm
import pickle

# Root directory of the project
ROOT_DIR = os.path.abspath("../model/mrcnn/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, fig=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

##################################################################################
    # # Show area outside image boundaries.
    # height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    # ax.set_title(title)
##################################################################################

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        class_id = class_ids[i]
        class_name = class_names[class_id]
        if class_name == "car" or class_name == "truck":
            color = colors[class_id]
            score = scores[i] if scores is not None else None
            caption = "{} {:.3f}".format(class_name, score) if score else class_name

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)

            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)
        else:
            continue

##################################################################################
        # # Mask Polygon
        # # Pad to ensure proper polygons for masks that touch image edges.
        # padded_mask = np.zeros(
        #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     p = Polygon(verts, facecolor="none", edgecolor=color)
        #     ax.add_patch(p)
##################################################################################
    return masked_image

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def ApplyMask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    colors = RandomColors(1)
    color = colors[0]
    masked_image = image.astype(np.uint32).copy()
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

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

def MRcnnModel():
    """
    Function:
        Build Mask RCNN model and load the pre-trained weights.
    """
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)


    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
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
                
    return model, class_names

def MRcnnPred(model, image):
    """
    Function:
        Use the mask RCNN model to predict the mask of the image
    """
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    return r    

def main(kitti_dir, prefix, mode = "test", if_2015 = True, if_MRCNN = False):
    """
    Function:
        main function, for reading, predicting, and plotting
    """
    left_img_dir = kitti_dir + "image_2/"
    right_img_dir = kitti_dir + "image_3/"
    GANet_disp_dir = mode + "_2015_GANet/"

    name_len = 6

    all_img_names = glob(left_img_dir + "*.png")
    total_disparity_num = len(all_img_names) // 2

    wield_number = 0

    # Pre-define all colors
    colors_all = random_colors(81)

    # load mrcnn model
    model, class_names = MRcnnModel()

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    i = 0
    img_count = str(i)
    zero_len = name_len - len(img_count)
    img_name = (zero_len * "0") + img_count + "_11"

    left_img = skimage.io.imread(left_img_dir + img_name + ".png")
    right_img = skimage.io.imread(right_img_dir + img_name + ".png")
    GANet_img = skimage.io.imread(GANet_disp_dir + img_name + ".png")
    GANet_img = skimage.color.gray2rgb(GANet_img)
    GANet_img = img_as_ubyte(GANet_img)

    if if_MRCNN:
        r = MRcnnPred(model, left_img)

        print("Number of objects: ", r['rois'].shape[0])
        print("Image shape: ",left_img.shape)
        print("Mask shape: ",r['masks'].shape)

        masked_img1 = display_instances(left_img, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, r['scores'], colors = colors_all, 
                                        ax = ax1, fig = fig)

        masked_img2 = display_instances(GANet_img, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, r['scores'], colors = colors_all, 
                                        ax = ax1, fig = fig)

        display_image = np.concatenate([masked_img1, masked_img2], axis = 0)
        ax1.clear()
        ax1.imshow(display_image.astype(np.uint8))
        ax1.axis("off")
        plt.savefig("mrcnn_presentation.png")

    else:
        with open(prefix + "/" + img_name + ".pickle", "rb") as f:
            mrcnn_result = pickle.load(f)
        mask_fg, mask_bg = FGandBG(mrcnn_result)

        masked_image_left = ApplyMask(left_img, mask_fg)
        masked_image_disp = ApplyMask(GANet_img, mask_fg)

        ax1.clear()
        ax1.imshow(left_img)
        # ax1.imshow(right_img)
        # ax1.imshow(masked_image_left)
        # ax1.imshow(masked_image_disp)
        ax1.axis("off")

        plt.savefig("forpresentation1.png")
        plt.show()

if __name__ == "__main__":
    
    kitti_dir_2015_train = "/home/azhang/Documents/kitti/2015/training/"
    prefix_2015_train = "train_2015_mrcnn/"

    main(kitti_dir_2015_train, prefix_2015_train, mode = "train", if_2015 = True)

