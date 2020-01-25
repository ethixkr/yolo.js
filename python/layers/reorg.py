import numpy as np
import tensorflow as tf

def reorg_numpy(x, block_size):
    batch, height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(batch, reduced_height, block_size, reduced_width, block_size, depth)
    z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
    return z
def reorg_tf(x, block_size):
    batch, height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = tf.reshape(x,[batch,reduced_height, block_size, reduced_width, block_size, depth])
    t = tf.transpose(y,[0,1,3,2,4,5])
    z = tf.reshape(t,[batch, reduced_height, reduced_width, -1])
    return z
def reorg_tf_func(x, bblock_size):
    return tf.space_to_depth(x, block_size=bblock_size)
    