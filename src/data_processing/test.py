import sys
sys.path.insert(0, '../model/mrcnn')
import random
import math
import numpy as np
import skimage.io
import skimage.color
import matplotlib
matplotlib.use('tkagg')
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

from glob import glob
from tqdm import tqdm

a = np.arange(1000)

b = 2 * a

plt.plot(a, b)
plt.show()