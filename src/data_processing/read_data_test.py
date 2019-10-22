import sys
import numpy as np
from glob import glob

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=sys.maxsize)

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

all_files = glob("train_out/*.npy")

for i in all_files:
    a = np.load(i)
    b = a == 1
    
    plt.cla()
    ax1.clear()
    ax1.imshow(a)
    ax2.clear()
    ax2.imshow(b)
    fig.canvas.draw()
    plt.pause(0.3)

