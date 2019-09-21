import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body

def define_colors(class_names):
    num_classes = len(class_names)
    np.random.seed(11011)  # Fixed seed for consistent colors across runs.
    colors = np.random.randint(num_classes, size = (num_classes, 3)).astype(np.float32) / float(num_classes)
    return colors

def font():
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    return font

def resize_image(image, input_shape):
    '''
    Function: resize the image to the std yolo input.

    Args:
        image           ->      image read by opencv.
        input_shape     ->      std yolo input shape.

    Output:
        new_image       ->      resized image for detection.
    '''

    ##### read original image size #####
    h_original, w_original, channels = image.shape
    image_original_shape = (h_original, w_original)
    ##### read standard input size of yolo body #####
    h_input, w_input = input_shape

    ##### resize the image to the standard input size #####
    new_image = np.ones(input_shape + (3,), dtype = np.float32) * 0.01
    scale = np.minimum(h_input/h_original, w_input/w_original)
    h_new = int(h_original * scale)
    w_new = int(w_original * scale)
    h_gap = (h_input - h_new) // 2
    w_gap = (w_input - w_new) // 2
    image_resize = cv2.resize(image, (w_new, h_new))
    new_image[h_gap: h_gap+h_new, w_gap: w_gap+w_new, :] = image_resize / 255.

    return new_image, image_original_shape

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def main(image, anchors, class_names, model_path, input_shape, score_threshold = 0.3, iou_threshold = 0.45):
    num_classes = len(class_names)

    boxed_image = letterbox_image(image, tuple(reversed(input_shape)))
    image_data = np.array(boxed_image, dtype='float32')

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    sess = K.get_session()
    yolo_model = load_model(model_path, compile=False)
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors,
                len(class_names), input_image_shape,
                score_threshold=score_threshold, iou_threshold=iou_threshold)

    out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    return out_boxes, out_scores, out_classes

if __name__ == "__main__":
    model_path = "model_data/yolo.h5"
    classes_path = "model_data/coco_classes.txt"
    anchors_path = "model_data/yolo_anchors.txt"
    # img_filename = "/home/azhang/Documents/experiments_in_VIVA/testimage/IMG_2006.jpg"
    input_shape = (416, 416)

    # read left and right images
    img_num = 2661
    img_filename = "/home/azhang/Documents/Stereo_Radar_Test/stereo_radar_calibration_annotation/left/" + str(img_num) + ".jpg"
    # img_filename = "/home/azhang/Documents/Stereo_Radar_Test/stereo_radar_calibration_annotation/right/" + str(img_num) + ".jpg"

    image = Image.open(img_filename)
    anchors = get_anchors(anchors_path)
    class_names = get_class(classes_path)

    colors = define_colors(class_names)

    pred_boxes, pred_scores, pred_classes = main(image, anchors, class_names, model_path, input_shape)

    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_axes([0,0,1,1])
    if len(pred_boxes) == 0:
        print("No object detected.")
    else:
        ax1.imshow(image)
        all_boxes = []
        for index, class_ind in list(enumerate(pred_classes)):
            pred_current_class = class_names[class_ind]
            box = pred_boxes[index]
            score = pred_scores[index]
            current_color = colors[class_ind]
            y, x, ymax, xmax = box
            w = xmax - x
            h = ymax - y

            all_boxes.append([x, y, w, h, class_ind])

            label = "{} {:.2f}".format(pred_current_class, score)

            rect = patches.Rectangle((x, y), w, h,linewidth=1,edgecolor=current_color, facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x, y, label, fontsize = 5, bbox=dict(facecolor=current_color, alpha=0.5),
                    horizontalalignment='left',
                    verticalalignment='bottom')
            ax1.set_axis_off()
        plt.show()

        all_boxes = np.array(all_boxes)
        np.save("/home/azhang/Documents/Stereo_Radar_Test/stereo_radar_calibration_annotation/yolo_results/left_boxes.npy", all_boxes)
        # np.save("/home/azhang/Documents/Stereo_Radar_Test/stereo_radar_calibration_annotation/yolo_results/right_boxes.npy", all_boxes)

