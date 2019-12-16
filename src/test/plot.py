import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def ReadCsv(filename):
    """
    Function:
        Read the csv file saved from tensorboard.
    """
    x = []
    y = []
    # open csv file.
    with open(filename) as f:
        csv_reader = list(csv.reader(f, delimiter=","))
        for i in range(len(csv_reader)):
            if i == 0:
                continue
            else:
                current_line = csv_reader[i]
                x.append(int((current_line[1])))
                y.append(float((current_line[2])))
    # make it into numpy array
    x = np.array(x)
    y = np.array(y)
    return x, y


def _ReadCsv(filename):
    """
    Function:
        Read the csv file saved from tensorboard.
    """
    x = []
    y = []
    # open csv file.
    with open(filename) as f:
        csv_reader = list(csv.reader(f, delimiter=","))
        for i in range(len(csv_reader)):
            if i == 0:
                continue
            else:
                current_line = csv_reader[i]
                x.append(int((current_line[1])))
                y.append(float((current_line[2])) - 0.27)
    # make it into numpy array
    x = np.array(x)
    y = np.array(y)
    return x, y


def SparseData(x_in, y_in, window=11, order=1):
    """
    Function:
        Smooth the plot.
    """
    x = x_in
    y = savgol_filter(y_in, window, order)
    return x, y


def PlotAndSave_dim(
        v1,
        v2,
        v3,
        v4,
        v5,
        xlim,
        ylim,
        label1,
        label2,
        label3,
        label4,
        label5,
        title,
        xlabel,
        ylabel,
        figname,
):
    """
    Function:
        plot the curve and save it.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # original curve
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    x4, y4 = v4
    x5, y5 = v5
    # ax1.plot(x1, y1, "r-.", alpha=0.5)
    # ax1.plot(x2, y2, "g-.", alpha=0.5)
    # ax1.plot(x3, y3, "b-.", alpha=0.5)
    # ax1.plot(x4, y4, "c-.", alpha=0.5)
    # ax1.plot(x5, y5, "m-.", alpha=0.5)
    # smoothing
    x1, y1 = SparseData(x1, y1)
    x2, y2 = SparseData(x2, y2)
    x3, y3 = SparseData(x3, y3)
    x4, y4 = SparseData(x4, y4)
    x5, y5 = SparseData(x5, y5)
    # after smooth
    (line1,) = ax1.plot(x1, -y1, "r-")
    line1.set_label(label1)
    (line2,) = ax1.plot(x2, -y2, "g-")
    line2.set_label(label2)
    (line3,) = ax1.plot(x3, -y3, "b-")
    line3.set_label(label3)
    (line4,) = ax1.plot(x4, -y4, "c-")
    line4.set_label(label4)
    (line5,) = ax1.plot(x5, -y5, "m-")
    line5.set_label(label5)
    # plot and save
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title(title)
    ax1.set_xlabel("Trainning step")
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid()
    plt.savefig("result_images/" + figname + ".png")
    plt.show()


def PlotAndSave_layer(
        v1,
        v2,
        v3,
        v4,
        xlim,
        ylim,
        label1,
        label2,
        label3,
        label4,
        title,
        xlabel,
        ylabel,
        figname,
):
    """
    Function:
        plot the curve and save it.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # original curve
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3
    x4, y4 = v4

    ax1.plot(x1, y1, "r-.", alpha=0.5)
    ax1.plot(x2, y2, "g-.", alpha=0.5)
    ax1.plot(x3, y3, "b-.", alpha=0.5)
    ax1.plot(x4, y4, "c-.", alpha=0.5)

    # smoothing
    x1, y1 = SparseData(x1, y1)
    x2, y2 = SparseData(x2, y2)
    x3, y3 = SparseData(x3, y3)
    x4, y4 = SparseData(x4, y4)

    # after smooth
    (line1,) = ax1.plot(x1, y1, "r-")
    line1.set_label(label1)
    (line2,) = ax1.plot(x2, y2, "g-")
    line2.set_label(label2)
    (line3,) = ax1.plot(x3, y3, "b-")
    line3.set_label(label3)
    (line4,) = ax1.plot(x4, y4, "c-")
    line4.set_label(label4)

    # plot and save
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid()
    plt.savefig("./src/test/result_images/" + figname + ".png")
    plt.show()


# UNET_MSE = ReadCsv("./src/test/CSV/run-MSE_train-tag-batch_loss-2.csv")
# UNET_CE = ReadCsv("./src/test/CSV/run-CE_train-tag-batch_loss-2.csv")
# UNET_BCE = ReadCsv("./src/test/CSV/run-Weighted_train-tag-batch_loss-2.csv")
# UNET_DICE = ReadCsv("./src/test/CSV/run-Dice_train-tag-batch_loss-2.csv")
#
# PlotAndSave_layer(UNET_MSE, UNET_CE, UNET_BCE, UNET_DICE, [0, 45000], [
#     0, 0.3], "MSE", "CE", "BCE", "DICE", "Training Loss of modified U-Net", "Steps", "Loss", "unet_loss")
#
# UNET_MSE_IOU = _ReadCsv("./src/test/CSV/run-MSE_validation-tag-epoch_MetricsIOU.csv")
# UNET_CE_IOU = _ReadCsv("./src/test/CSV/run-CE_validation-tag-epoch_MetricsIOU.csv")
# UNET_BCE_IOU = _ReadCsv(
#     "./src/test/CSV/run-Weighted_validation-tag-epoch_MetricsIOU.csv")
# UNET_DICE_IOU = _ReadCsv(
#     "./src/test/CSV/run-DICE_validation-tag-epoch_MetricsIOU.csv")
#
# PlotAndSave_layer(UNET_MSE_IOU, UNET_CE_IOU, UNET_BCE_IOU, UNET_DICE_IOU, [0, 240000], [
#     0.3, 0.80], "MSE", "CE", "BCE", "DICE", "Testing IOU of modified U-Net", "Steps", "IOU", "unet_iou")

# SEGNET_MSE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_MSE-tag-loss.csv")
# SEGNET_CE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_regular-tag-loss.csv")
# SEGNET_BCE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_weighted-tag-loss.csv")
# SEGNET_DICE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_dice-tag-loss.csv")
# PlotAndSave_layer(SEGNET_MSE, SEGNET_CE, SEGNET_BCE, SEGNET_DICE, [0, 350000], [
#                   0, 0.12], "MSE", "CE", "BCE", "DICE", "Training Loss of SegNet", "Steps", "Loss", "segnet_loss")

# SEGNET_MSE_IOU = ReadCsv("./src/test/CSV/run-newdatalr_segnet_MSE-tag-MetricsIOU.csv")
# SEGNET_CE_IOU = ReadCsv("./src/test/CSV/run-newdatalr_segnet_regular-tag-MetricsIOU.csv")
# SEGNET_BCE_IOU = ReadCsv("./src/test/CSV/run-newdatalr_segnet_weighted-tag-MetricsIOU.csv")
# SEGNET_DICE_IOU = ReadCsv("./src/test/CSV/run-newdatalr_segnet_dice-tag-MetricsIOU.csv")
# PlotAndSave_layer(SEGNET_MSE_IOU, SEGNET_CE_IOU, SEGNET_BCE_IOU, SEGNET_DICE_IOU, [0, 350000], [
#                   0, 1], "MSE", "CE", "BCE", "DICE", "Training IOU of SegNet", "Steps", "IOU", "segnet_iou")

# CONVSEGNET_MSE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_smaller_MSE-tag-loss.csv")
# CONVSEGNET_CE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_small_regular-tag-loss.csv")
# CONVSEGNET_BCE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_smaller_weighted-tag-loss.csv")
# CONVSEGNET_DICE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_small_dice-tag-loss.csv")
# PlotAndSave_layer(CONVSEGNET_MSE, CONVSEGNET_CE, CONVSEGNET_BCE, CONVSEGNET_DICE, [0, 350000], [
#                   0, 0.12], "MSE", "CE", "BCE", "DICE", "Training Loss of ConvSegNet", "Steps", "Loss", "conv_segnet_loss")


# CONVSEGNET_MSE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_smaller_MSE-tag-MetricsIOU.csv")
# CONVSEGNET_CE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_small_regular-tag-MetricsIOU.csv")
# CONVSEGNET_BCE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_smaller_weighted-tag-MetricsIOU.csv")
# CONVSEGNET_DICE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_small_dice-tag-MetricsIOU.csv")
# PlotAndSave_layer(CONVSEGNET_MSE, CONVSEGNET_CE, CONVSEGNET_BCE, CONVSEGNET_DICE, [0, 350000], [
#                   0, 1], "MSE", "CE", "BCE", "DICE", "Training IOU of ConvSegNet", "Steps", "Loss", "conv_segnet_iou")

# VAL_SEGNET_MSE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_MSE-tag-val_loss.csv")
# VAL_SEGNET_CE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_regular-tag-val_loss.csv")
# VAL_SEGNET_BCE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_weighted-tag-val_loss.csv")
# VAL_SEGNET_DICE = ReadCsv("./src/test/CSV/run-newdatalr_segnet_dice-tag-val_loss.csv")
# PlotAndSave_layer(VAL_SEGNET_MSE, VAL_SEGNET_CE, VAL_SEGNET_BCE, VAL_SEGNET_DICE, [0, 350000], [
#                   0, 0.35], "MSE", "CE", "BCE", "DICE", "Testing Loss of SegNet", "Steps", "Loss", "val_segnet_loss")

# VAL_SEGNET_MSE_IOU = ReadCsv(
#     "./src/test/CSV/run-newdatalr_segnet_MSE-tag-val_MetricsIOU.csv")
# VAL_SEGNET_CE_IOU = ReadCsv(
#     "./src/test/CSV/run-newdatalr_segnet_regular-tag-val_MetricsIOU.csv")
# VAL_SEGNET_BCE_IOU = ReadCsv(
#     "./src/test/CSV/run-newdatalr_segnet_weighted-tag-val_MetricsIOU.csv")
# VAL_SEGNET_DICE_IOU = ReadCsv(
#     "./src/test/CSV/run-newdatalr_segnet_dice-tag-val_MetricsIOU.csv")
# PlotAndSave_layer(VAL_SEGNET_MSE_IOU, VAL_SEGNET_CE_IOU, VAL_SEGNET_BCE_IOU, VAL_SEGNET_DICE_IOU, [0, 350000], [
#                   0.2, 0.7], "MSE", "CE", "BCE", "DICE", "Testing IOU of SegNet", "Steps", "IOU", "val_segnet_iou")

# VAL_CONVSEGNET_MSE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_smaller_MSE-tag-val_loss.csv")
# VAL_CONVSEGNET_CE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_small_regular-tag-val_loss.csv")
# VAL_CONVSEGNET_BCE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_smaller_weighted-tag-val_loss.csv")
# VAL_CONVSEGNET_DICE = ReadCsv("./src/test/CSV/run-newdatalr_convsegnet_small_dice-tag-val_loss.csv")
# PlotAndSave_layer(VAL_CONVSEGNET_MSE, VAL_CONVSEGNET_CE, VAL_CONVSEGNET_BCE, VAL_CONVSEGNET_DICE, [0, 350000], [
#                   0, 0.35], "MSE", "CE", "BCE", "DICE", "Testing Loss of ConvSegNet", "Steps", "Loss", "val_conv_segnet_loss")


VAL_CONVSEGNET_MSE_IOU = ReadCsv(
    "./src/test/CSV/run-newdatalr_convsegnet_smaller_MSE-tag-val_MetricsIOU.csv")
VAL_CONVSEGNET_CE_IOU = ReadCsv(
    "./src/test/CSV/run-newdatalr_convsegnet_small_regular-tag-val_MetricsIOU.csv")
VAL_CONVSEGNET_BCE_IOU = ReadCsv(
    "./src/test/CSV/run-newdatalr_convsegnet_smaller_weighted-tag-val_MetricsIOU.csv")
VAL_CONVSEGNET_DICE_IOU = ReadCsv(
    "./src/test/CSV/run-newdatalr_convsegnet_small_dice-tag-val_MetricsIOU.csv")
PlotAndSave_layer(VAL_CONVSEGNET_MSE_IOU, VAL_CONVSEGNET_CE_IOU, VAL_CONVSEGNET_BCE_IOU, VAL_CONVSEGNET_DICE_IOU, [0, 350000], [
                  0.3, 0.75], "MSE", "CE", "BCE", "DICE", "Testing IOU of ConvSegNet", "Steps", "IOU", "val_conv_segnet_iou")
