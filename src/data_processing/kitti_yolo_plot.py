import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

def TransferDispAndObj(kitti_dir):

    """
    Function:
        read the disparity map and object map, normalize them, concatenate them,
        then save them.
    Args:
        kitti_dir               ->              directory path for kitti dataset
    """

    # paths for different images
    disparity_occlude_dir = kitti_dir +  "disp_occ_0/"
    disparity_noc_dir = kitti_dir +  "disp_noc_0/"
    obj_dir = kitti_dir + "obj_map/"

    name_len = 6
    # read only 50 images for test
    total_num_imgs = 50

    # find out how many images are there
    all_img_names = glob(disparity_occlude_dir + "*.png")
    total_disparity_num = len(all_img_names)

    # read and normalize the images
    for i in tqdm(range(total_num_imgs)):
        img_count = str(i)
        zero_len = name_len - len(img_count)
        img_name = (zero_len * "0") + img_count + "_10.png"
        disp_img = np.array(Image.open(disparity_noc_dir + img_name))
        obj_img = np.array(Image.open(obj_dir + img_name))

        plt.imsave("middel_transfer/disp" + str(i) + ".png", disp_img)
        plt.imsave("middel_transfer/obj" + str(i) + ".png", obj_img)

    # concatenate the images
    for i in tqdm(range(total_num_imgs)):
        disp_cur = np.array(Image.open("middel_transfer/disp" + str(i) + ".png"))
        obj_cur = np.array(Image.open("middel_transfer/obj" + str(i) + ".png"))
        disp_cur = disp_cur[..., :3]
        obj_cur = obj_cur[..., :3]
        dispobj_img = np.concatenate([disp_cur, obj_cur], axis = 0)

        plt.imsave("kitti_obj_test/disp" + str(i) + ".png", dispobj_img)

def ReadFromDir(dirct, if_plot = True):

    """
    Function:
        read all the results (yolo, pixels removal, and disparity) and plot together for visualize.

    Args:
        dirct               ->              directory of all the results
        if_plot             ->              whether to plot all results
    """

    # define paths
    yolo_imgs_pre = dirct + "imgs"
    removal_pre = dirct + "irem"
    dispobj_pre = dirct + "disp"

    # total number of images
    total_num = 50

    if if_plot:
        plt.ion()
        fig = plt.figure(figsize = (16,8))
        ax1 = fig.add_subplot(111)

    # read and plot
    for i in tqdm(range(total_num)):
        yolo_img = np.array(Image.open(yolo_imgs_pre + str(i) + ".png"))
        remov_img = np.array(Image.open(removal_pre + str(i) + ".png"))
        disp_img = np.array(Image.open(dispobj_pre + str(i) + ".png"))
        disp_img = disp_img[..., :3]

        if (yolo_img is None) or (remov_img is None) or (disp_img is None):
            continue

        display_yolo = np.concatenate([yolo_img, remov_img, disp_img], axis = 1)
        
        if if_plot:
            plt.cla()
            ax1.clear()
            ax1.axis("off")
            ax1.imshow(display_yolo)
            fig.canvas.draw()
            plt.pause(1)

if __name__ == "__main__":
    image_directory = "kitti_obj_test/"
    kitti_directory = "/home/azhang/Documents/kitti/2015/training/"

    disp_obj_num = len(glob(image_directory + "disp*.png"))

    if disp_obj_num == 0:
        TransferDispAndObj(kitti_directory)
    else:
        ReadFromDir(image_directory)


