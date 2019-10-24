import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np

x_train = np.array([[-1.551, -1.469], [1.022, 1.664]], dtype=np.float32)
y_train = np.array([1, 0], dtype=int)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.int32, shape=[None])

with tf.name_scope('network'):
    layer1 = tf.keras.layers.Dense(2, input_shape=(2, ))
    layer2 = tf.keras.layers.Dense(2, input_shape=(2, ))
    fc1 = layer1(x)
    logits = layer2(fc1)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_fn = tf.reduce_mean(xentropy)

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss_fn)

gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        loss_val = sess.run(loss_fn, feed_dict={x:x_train, y:y_train})
        _ = sess.run(train_op, feed_dict={x:x_train, y:y_train})

        print(loss_val)