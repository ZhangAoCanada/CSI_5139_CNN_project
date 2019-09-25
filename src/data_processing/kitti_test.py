"""
Combine kitti stereo data with Yolov3 detection.
"""
import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(0, '../model/yolo_keras_customized_output')
# import matplotlib
# matplotlib.use("tkagg")
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from glob import glob
from output_test import get_anchors, get_class, Detection, ColorsForPIL, FontsAndThickness, DrawBox
from tqdm import tqdm

def YoloKerasPred(image, w_threshold = 0, h_threshold = 0):

    """
    Function:
        bounding box and class predictions by yolov3. copy right remains to git keras_yolov3.
        Additional code is added to the repo in order to get the customized results.
    Args:
        image               ->              input image of yolo model. (must be in the format of PIL.Image)
        w_threshold         ->              threshold for the size of the bounding box
        h_threshold         ->              threshold for the size of the bounding box
    """

    # get all the necessary parameter paths for the yolo model
    model_path = "../model/yolo_keras_customized_output/model_data/yolo.h5"
    classes_path = "../model/yolo_keras_customized_output/model_data/coco_classes.txt"
    anchors_path = "../model/yolo_keras_customized_output/model_data/yolo_anchors.txt"

    # read parameters
    anchors = get_anchors(anchors_path)
    class_names = get_class(classes_path)
    colors = ColorsForPIL(class_names)

    # since the model is trained under the fixed input size (416, 416)
    input_shape = (416, 416)

    # get the prediction from the yolov3 model (GPU involved !)
    pred_boxes, pred_scores, pred_classes = Detection(image, anchors, class_names, model_path, input_shape)

    # define the font format, box format to make the box more attractive
    font, thickness = FontsAndThickness(image)

    # check if there are boxes detected (normally, yes)
    if len(pred_boxes) == 0:
        print("No object detected.")
    else:
        all_boxes = []
        for index, class_ind in list(enumerate(pred_classes)):
            # interpret the prediction into boxes and classes
            pred_current_class = class_names[class_ind]
            box = pred_boxes[index]
            score = pred_scores[index]
            current_color = colors[class_ind]

            # define the label of the bounding boxes
            label = '{} {:.2f}'.format(pred_current_class, score)
            y, x, ymax, xmax = box
            w, h = xmax - x, ymax - y

            if w < w_threshold or h < h_threshold:
                continue
            else:
                # save all bounding boxes for further usage
                all_boxes.append([x, y, xmax, ymax, class_ind])
                # drawing bounding boxes
                img_box = DrawBox(image, x, y, xmax, ymax, label, font, current_color, thickness)
        all_boxes = np.array(all_boxes)
    return img_box, all_boxes

def MatchGTAndBox(obj_location, boxes):

    """
    Function:
        Use the object images from kitti as ground truth to find the corresponding bounding boxes
        from the prediction of yolov3.
    Args:
        obj_location            ->              the pixel locations of the objects in the obj_img
        boxes                   ->              all the bounding boxes output from yolo model
    """

    # read width and height pixel locations respectively
    obj_loc_x = obj_location[0]
    obj_loc_y = obj_location[1]
    
    """ 
    algorithm: 
        1. get the total number of pixel locations.
        2. iterate over all bounding boxes, then find the number of object pixels that located in that box.
        3. using the number of pixels inbox devided by number of total pixels as the score.
        4. the box with largest score should be the box we are looking for.
    """
    # read the sizes
    loc_mask_size = len(obj_loc_x)
    boxes_size = len(boxes)

    #initialize the score
    match_score = 0.
    for box_id in range(boxes_size):
        # get the info from the boxes
        current_box = boxes[box_id]
        x, y, xmax, ymax, class_id = current_box

        # attention: X and Y are flipped in the images.
        x_match = (obj_loc_x >= y) & (obj_loc_x <= ymax)
        y_match = (obj_loc_y >= x) & (obj_loc_y <= xmax)
        match_mask = x_match & y_match

        # num of pixels located in-box
        num_match_points = len(obj_loc_x[match_mask])
        # calculate score
        current_match_score = float(num_match_points) / loc_mask_size
        # find the best score
        if current_match_score >= match_score:
            match_score = current_match_score
            best_box = current_box
    return best_box

def FindObjBox(obj_img, left_boxes, right_boxes):

    """
    Function:
        Find the best fit box using the function MatchGTAndBox() with left and right 
        images respectively.
    Args:
        obj_img             ->              images contain only objects from kitti
        left_boxes          ->              predictions of left images from yolov3
        rigth_boxes          ->              predictions of right images from yolov3
    """

    # find the number of objects and their corresponding numbers
    all_obj_num = np.delete(np.unique(obj_img), 0)
    left_output_boxes = []
    right_output_boxes = []

    # find the best box for the objects and return 
    for obj_ind in all_obj_num:
        obj_location = np.where(obj_img == obj_ind)
        left_fit_box = MatchGTAndBox(obj_location, left_boxes)
        right_fit_box = MatchGTAndBox(obj_location, right_boxes)
        left_output_boxes.append(left_fit_box)
        right_output_boxes.append(right_fit_box)
    return left_output_boxes, right_output_boxes

def BooleanImageAndRemove(img, boxes):

    """
    Function:
        with the best fit box, remove all other unnecessary pixels and set them to 0
    Args:
        img             ->              the image to be processed
        boxes           ->              prediction of the image from yolov3
    """

    all_masks = []
    new_img = img
    w, h, channels = img.shape
    mask_shape = (w, h)

    # initialize the mask
    final_mask = np.full(mask_shape, False, dtype = bool)

    for box_id in range(len(boxes)):
        mask = np.full(mask_shape, False, dtype = bool)
        current_box = boxes[box_id]
        x, y, xmax, ymax, class_id = current_box

        # make the localization of the boxes as interger for further usage
        x, y, xmax, ymax = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), \
                            np.ceil(xmax).astype(np.int32), np.ceil(ymax).astype(np.int32)

        # find the mask for each box
        mask[y:ymax, x:xmax] = True
        all_masks.append(mask)

    # find the mask for all boxes
    for mask_id in range(len(all_masks)):
        final_mask = final_mask | all_masks[mask_id]

    # remove all unnecessary pixels and set to 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if final_mask[i, j]:
                new_img[i, j] = img[i, j]
            else:
                new_img[i, j] = 0.
    return new_img

def RemoveUndetectedArea(left_img, right_img, left_boxes, right_boxes):

    """
    Function:
        remove unnecessary pixels from left and right images respectively.
    """

    new_left_img = BooleanImageAndRemove(left_img, left_boxes)
    new_right_img = BooleanImageAndRemove(right_img, right_boxes)
    return new_left_img, new_right_img
    
def ReadAndPlotImages(kitti_dir):

    """
    Main function:
        for plot testing
    """

    disparity_occlude_dir = kitti_dir +  "disp_occ_0/"
    disparity_noc_dir = kitti_dir +  "disp_noc_0/"

    left_img_dir = kitti_dir + "image_2/"
    right_img_dir = kitti_dir + "image_3/"
    obj_dir = kitti_dir + "obj_map/"
    name_len = 6

    all_img_names = glob(disparity_occlude_dir + "*.png")
    total_disparity_num = len(all_img_names)

    wield_number = 97

    for i in tqdm(range(wield_number, total_disparity_num)):
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10.png"

        disp_img = np.array(Image.open(disparity_noc_dir + img_name))
        left_img = Image.open(left_img_dir + img_name)
        right_img = Image.open(right_img_dir + img_name)
        obj_img = np.array(Image.open(obj_dir + img_name))
        left_img_arr = np.array(left_img)
        right_img_arr = np.array(right_img)

        if left_img is None or right_img is None:
            raise ValueError("Wrong kitti dataset directory.")

        left_det, left_boxes = YoloKerasPred(left_img)
        right_det, right_boxes = YoloKerasPred(right_img)

        # left_fit_boxes, right_fit_boxes = FindObjBox(obj_img, left_boxes, right_boxes)

        # # remove all non-object pixels
        # left_remove, right_remove = RemoveUndetectedArea(left_img_arr, right_img_arr,
        #                                     left_fit_boxes, right_fit_boxes)

        # left_right_detection = np.concatenate([left_det, right_det], axis = 0)
        # left_right_remove = np.concatenate([left_remove, right_remove], axis = 0)
        # disp_obj = np.concatenate([disp_img, obj_img], axis = 0)

        # all_imgs_format = Image.fromarray(left_right_detection)

        # all_imgs_remove_format = Image.fromarray(left_right_remove)
        # all_disp_format = Image.fromarray(disp_obj)
        # all_imgs_format.save("kitti_obj_test/imgs" + str(i) + ".png", "PNG")
        # all_imgs_remove_format.save("kitti_obj_test/irem" + str(i) + ".png", "PNG")
        # all_disp_format.save("kitti_obj_test/disp" + str(i) + ".png", "PNG")

        left_det.save("kitti_obj_test/left_yolo" + str(i) + ".png", "PNG")
        right_det.save("kitti_obj_test/right_yolo" + str(i) + ".png", "PNG")
        np.save("kitti_obj_test/left_box" + str(i) + ".npy", left_boxes)
        np.save("kitti_obj_test/right_box" + str(i) + ".npy", right_boxes)


if __name__ == "__main__":
    
    kitti_dir = "/home/azhang/Documents/kitti/2015/training/"

    ReadAndPlotImages(kitti_dir)