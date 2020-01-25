import numpy as np
import tensorflow as tf
from reorg import reorg_tf_func, reorg_tf, reorg_numpy

x = np.asarray([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])

with tf.Session() as sess:
    print("------------Must Be Same--------------")
    print("---------------------------")
    print(sess.run(reorg_tf_func(x, 2)))
    print("---------------------------")
    print(sess.run(reorg_tf(x, 2)))
    print("---------------------------")
    print(reorg_numpy(x, 2))
    print("---------------------------")

