import sys
sys.path.insert(0, '../model/yolo_keras_customized_output')
# import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from glob import glob
from output_test import get_anchors, get_class, Detection, ColorsForPIL, FontsAndThickness, DrawBox
from tqdm import tqdm
    
def YoloKerasPred(image, w_threshold = 0, h_threshold = 0):
    model_path = "../model/yolo_keras_customized_output/model_data/yolo.h5"
    classes_path = "../model/yolo_keras_customized_output/model_data/coco_classes.txt"
    anchors_path = "../model/yolo_keras_customized_output/model_data/yolo_anchors.txt"
    input_shape = (416, 416)

    anchors = get_anchors(anchors_path)
    class_names = get_class(classes_path)
    colors = ColorsForPIL(class_names)
    pred_boxes, pred_scores, pred_classes = Detection(image, anchors, class_names, model_path, input_shape)

    # drawing bounding boxes
    font, thickness = FontsAndThickness(image)

    if len(pred_boxes) == 0:
        print("No object detected.")
    else:
        all_boxes = []
        for index, class_ind in list(enumerate(pred_classes)):
            pred_current_class = class_names[class_ind]
            box = pred_boxes[index]
            score = pred_scores[index]
            current_color = colors[class_ind]

            label = '{} {:.2f}'.format(pred_current_class, score)
            y, x, ymax, xmax = box
            w, h = xmax - x, ymax - y

            if w < w_threshold or h < h_threshold:
                continue
            else:
                all_boxes.append([x, y, xmax, ymax, class_ind])
                
                # drawing bounding boxes
                img_box = DrawBox(image, x, y, xmax, ymax, label, font, current_color, thickness)

        all_boxes = np.array(all_boxes)
    return img_box, all_boxes

def MatchGTAndBox(obj_location, boxes):
    obj_loc_x = obj_location[0]
    obj_loc_y = obj_location[1]
    loc_mask_size = len(obj_loc_x)
    boxes_size = len(boxes)
    match_score = 0.
    for box_id in range(boxes_size):
        current_box = boxes[box_id]
        x, y, xmax, ymax, class_id = current_box
        x_match = (obj_loc_x >= y) & (obj_loc_x <= ymax)
        y_match = (obj_loc_y >= x) & (obj_loc_y <= xmax)
        match_mask = x_match & y_match
        num_match_points = len(obj_loc_x[match_mask])
        current_match_score = float(num_match_points) / loc_mask_size
        if current_match_score >= match_score:
            match_score = current_match_score
            best_box = current_box
    return best_box

def FindObjBox(obj_img, left_boxes, right_boxes):
    all_obj_num = np.delete(np.unique(obj_img), 0)
    left_output_boxes = []
    right_output_boxes = []
    for obj_ind in all_obj_num:
        obj_location = np.where(obj_img == obj_ind)
        left_fit_box = MatchGTAndBox(obj_location, left_boxes)
        right_fit_box = MatchGTAndBox(obj_location, right_boxes)
        left_output_boxes.append(left_fit_box)
        right_output_boxes.append(right_fit_box)
    return left_output_boxes, right_output_boxes

def BooleanImageAndRemove(img, boxes):
    all_masks = []
    new_img = img
    w, h, channels = img.shape
    mask_shape = (w, h)
    final_mask = np.full(mask_shape, False, dtype = bool)
    for box_id in range(len(boxes)):
        mask = np.full(mask_shape, False, dtype = bool)
        current_box = boxes[box_id]
        x, y, xmax, ymax, class_id = current_box
        x, y, xmax, ymax = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32), \
                            np.ceil(xmax).astype(np.int32), np.ceil(ymax).astype(np.int32)
        mask[y:ymax, x:xmax] = True
        all_masks.append(mask)
    for mask_id in range(len(all_masks)):
        final_mask = final_mask | all_masks[mask_id]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if final_mask[i, j]:
                new_img[i, j] = img[i, j]
            else:
                new_img[i, j] = 0.
    return new_img

def RemoveUndetectedArea(left_img, right_img, left_boxes, right_boxes):
    new_left_img = BooleanImageAndRemove(left_img, left_boxes)
    new_right_img = BooleanImageAndRemove(right_img, right_boxes)
    return new_left_img, new_right_img
    
def ReadAndPlotImages(kitti_dir):
    disparity_occlude_dir = kitti_dir +  "disp_occ_0/"
    disparity_noc_dir = kitti_dir +  "disp_noc_0/"

    left_img_dir = kitti_dir + "image_2/"
    right_img_dir = kitti_dir + "image_3/"
    obj_dir = kitti_dir + "obj_map/"
    name_len = 6

    all_img_names = glob(disparity_occlude_dir + "*.png")
    total_disparity_num = len(all_img_names)

    plt.ion()
    fig = plt.figure(figsize = (15, 6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    for i in tqdm(range(total_disparity_num)):
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10.png"

        disp_img = np.array(Image.open(disparity_noc_dir + img_name))
        left_img = Image.open(left_img_dir + img_name)
        right_img = Image.open(right_img_dir + img_name)
        obj_img = np.array(Image.open(obj_dir + img_name))

        if left_img is None:
            raise ValueError("Wrong kitti dataset directory.")

        left_det, left_boxes = YoloKerasPred(left_img)
        right_det, right_boxes = YoloKerasPred(right_img)

        left_img = np.array(left_img)
        right_img = np.array(right_img)

        left_fit_boxes, right_fit_boxes = FindObjBox(obj_img, left_boxes, right_boxes)

        # remove all non-object pixels
        left_remove, right_remove = RemoveUndetectedArea(left_img, right_img,
                                            left_fit_boxes, right_fit_boxes)

        plt.cla()
        ax1.clear()
        ax1.axis("off")
        ax1.imshow(left_det)
        ax2.clear()
        ax2.axis("off")
        ax2.imshow(right_det)
        ax3.clear()
        ax3.axis("off")
        ax3.imshow(disp_img)
        ax4.clear()
        ax4.axis("off")
        ax4.imshow(obj_img)
        plt.savefig("kitti_obj_test/" + str(i) + ".png")

if __name__ == "__main__":
    
    kitti_dir = "/home/aozhang2/Documents/kitti/training/"

    ReadAndPlotImages(kitti_dir)