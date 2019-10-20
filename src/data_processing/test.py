import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L


a = np.ones([1, 10, 10, 3])
# a = np.expand_dims(a, axis = 0)

innnn = tf.placeholder(tf.float32, [None, 10, 10, 3])

b = L.Conv2D(10, [3, 3])(innnn)

c = L.Conv2DTranspose(3, [3, 3])(b)

init = tf.initializers.global_variables()

sess = tf.Session()

sess.run(init)

bb, cc = sess.run([b, c], feed_dict = {innnn: a})

print(bb.shape)
print(cc.shape)


