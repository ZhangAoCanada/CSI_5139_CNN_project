import sys
import numpy as np
from glob import glob
from PIL import Image

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=sys.maxsize)

# plt.ion()
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

all_files = glob("train_out/*.npy")

# for i in all_files:
#     a = np.load(i)
#     print(a.shape)
    # b = a == 1
    
    # plt.cla()
    # ax1.clear()
    # ax1.imshow(a)
    # ax2.clear()
    # ax2.imshow(b)
    # fig.canvas.draw()
    # plt.pause(0.3)

for i in range(len(all_files)):
    img = np.array(Image.open("train_in/" + str(i) + ".png"))
    gt = np.load("train_out/" + str(i) + ".npy")
    print("shape:\t{}, shape:\t{}".format(img.shape, gt.shape) )

