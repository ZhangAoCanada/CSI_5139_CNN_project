import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv
from glob import glob


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


def SparseData(x_in, y_in, window=111, order=1):
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


UNET_MSE = ReadCsv("./src/test/CSV/run-MSE_train-tag-batch_loss.csv")
UNET_CE = ReadCsv("./src/test/CSV/run-CE_train-tag-batch_loss.csv")
UNET_BCE = ReadCsv("./src/test/CSV/run-Weighted_train-tag-batch_loss.csv")
UNET_DICE = ReadCsv("./src/test/CSV/run-Dice_train-tag-batch_loss.csv")

PlotAndSave_layer(UNET_MSE, UNET_CE, UNET_BCE, UNET_DICE, [0, 25000], [
                  0, 0.3], "MSE", "CE", "BCE", "DICE", "Training Loss of modified U-Net", "Steps", "Loss", "unet_loss")

UNET_MSE_IOU = ReadCsv("./src/test/CSV/run-MSE_train-tag-batch_MetricsIOU.csv")
UNET_CE_IOU = ReadCsv("./src/test/CSV/run-CE_train-tag-batch_MetricsIOU.csv")
UNET_BCE_IOU = ReadCsv(
    "./src/test/CSV/run-Weighted_train-tag-batch_MetricsIOU.csv")
UNET_DICE_IOU = ReadCsv(
    "./src/test/CSV/run-DICE_train-tag-batch_MetricsIOU.csv")

PlotAndSave_layer(UNET_MSE_IOU, UNET_CE_IOU, UNET_BCE_IOU, UNET_DICE_IOU, [0, 25000], [
                  0, 0.93], "MSE", "CE", "BCE", "DICE", "Trainning IOU of modified U-Net", "Steps", "IOU", "unet_loss")
