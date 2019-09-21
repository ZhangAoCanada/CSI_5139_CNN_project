import sys
sys.path.insert(0, '../model/yolo_keras_customized_output')
import colorsys
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from glob import glob
from output_test import define_colors, get_anchors, get_class, define_colors, Detection, ColorsForPIL

def YoloKerasPred(image):
    model_path = "../model/yolo_keras_customized_output/model_data/yolo.h5"
    classes_path = "../model/yolo_keras_customized_output/model_data/coco_classes.txt"
    anchors_path = "../model/yolo_keras_customized_output/model_data/yolo_anchors.txt"
    input_shape = (416, 416)

    anchors = get_anchors(anchors_path)
    class_names = get_class(classes_path)
    colors = ColorsForPIL(class_names)
    pred_boxes, pred_scores, pred_classes = Detection(image, anchors, class_names, model_path, input_shape)

    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font='/usr/share/fonts/truetype/ttf-khmeros-core/KhmerOS.ttf', 
                            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    if len(pred_boxes) == 0:
        print("No object detected.")
    else:
        all_boxes = []
        for index, class_ind in list(enumerate(pred_classes)):
            pred_current_class = class_names[class_ind]
            box = pred_boxes[index]
            score = pred_scores[index]
            current_color = colors[class_ind]
            current_color = (current_color[0], current_color[1], current_color[2])

            label = '{} {:.2f}'.format(pred_current_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            y, x, ymax, xmax = box
            w = xmax - x
            h = ymax - y

            if y - label_size[1] >= 0:
                text_origin = np.array([x, y - label_size[1]])
            else:
                text_origin = np.array([x, y + 1])

            all_boxes.append([x, y, w, h, class_ind])

            for i in range(thickness):
                draw.rectangle(
                    [x + i, y + i, xmax - i, ymax - i],
                    outline=current_color)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=current_color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        all_boxes = np.array(all_boxes)

        return image, all_boxes


def ReadAndPlotImages(kitti_dir, if_plot = True):
    disparity_occlude_dir = kitti_dir +  "disp_occ_0/"
    disparity_noc_dir = kitti_dir +  "disp_noc_0/"

    left_img_dir = kitti_dir + "image_2/"
    right_img_dir = kitti_dir + "image_3/"
    obj_dir = kitti_dir + "obj_map/"

    name_len = 6

    if if_plot:
        plt.ion()
        fig = plt.figure(figsize = (16, 8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

    all_img_names = glob(disparity_occlude_dir + "*.png")
    total_disparity_num = len(all_img_names)

    for i in range(1):
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10.png"

        disp_img = np.array(Image.open(disparity_noc_dir + img_name))
        left_img = Image.open(left_img_dir + img_name)
        right_img = Image.open(right_img_dir + img_name)
        obj_img = np.array(Image.open(obj_dir + img_name))

        left_detect, left_boxes = YoloKerasPred(left_img)
        right_detect, right_boxes = YoloKerasPred(right_img)

        left_img = np.array(left_img)
        right_img = np.array(right_img)

        if left_img is None:
            raise ValueError("Wrong kitti dataset directory.")

        if if_plot:
            plt.cla()
            ax1.clear()
            ax1.axis("off")
            ax1.imshow(left_detect)
            ax2.clear()
            ax2.axis("off")
            ax2.imshow(right_detect)
            ax3.clear()
            ax3.axis("off")
            ax3.imshow(disp_img)
            ax4.clear()
            ax4.axis("off")
            ax4.imshow(obj_img)
            fig.canvas.draw()
            plt.savefig("detect_test.png")
            plt.pause(9999)


if __name__ == "__main__":
    
    kitti_dir = "/home/aozhang2/Documents/kitti/training/"

    ReadAndPlotImages(kitti_dir, if_plot = True)