import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob

def ObjImgTest(kitti_dir, if_plot = True):
    disparity_occlude_dir = kitti_dir +  "disp_occ_0/"
    left_img_dir = kitti_dir + "image_2/"
    right_img_dir = kitti_dir + "image_3/"
    obj_dir = kitti_dir + "obj_map/"

    name_len = 6

    all_img_names = glob(disparity_occlude_dir + "*.png")
    total_disparity_num = len(all_img_names)

    if if_plot:
        plt.ion()
        fig = plt.figure(figsize = (8, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

    for i in range(total_disparity_num):
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10.png"

        disp_img = np.array(Image.open(disparity_occlude_dir + img_name).resize((621, 188)))
        left_img = np.array(Image.open(left_img_dir + img_name).resize((621, 188)))
        right_img = np.array(Image.open(right_img_dir + img_name).resize((621, 188)))
        obj_img = np.array(Image.open(obj_dir + img_name).resize((621, 188)))

        obj_mask_num_repres = np.delete(np.unique(obj_img), 0)
        hm_obj = len(obj_mask_num_repres)
        first_obj = np.where(obj_img == obj_mask_num_repres[0], 1, 0)

        if if_plot:
            plt.cla()
            ax1.clear()
            ax1.axis("off")
            ax1.imshow(obj_img)
            ax2.clear()
            ax2.axis("off")
            ax2.imshow(first_obj)
            fig.canvas.draw()
            plt.pause(1)

if __name__ == "__main__":
    kitti_dir = "/home/azhang/Documents/kitti/2015/training/"

    ObjImgTest(kitti_dir)