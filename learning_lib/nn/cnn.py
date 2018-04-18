import tensorflow as tf
import numpy as np
from learning_lib.nn.nn_base_class import NN

"""
layer_type == conv:
    filter_size
    init_filter_mean
    init_filter_stddev
    stride_size
    ----
    filter

layer_type == pool:
    pool_type
    pool_size
    stride_size
}"""

class CNN(NN):
    """ An implementation of a 2D convolutional neural network class using our NN base class.

    Inputs
        layer_config <list>: A list containing dictionaries which describe the structure of the produced
            network.
    """

    def create_params(self):
        for i in range(len(self.lc)):
            lc = self.lc[i]
            if lc['layer_type'] == 'conv':
                lc['filter'] = tf.Variable(tf.random_normal(
                    shape=lc['filter_size'],
                    mean=lc['init_filter_mean'],
                    stddev=lc['init_filter_stddev'],
                    dtype=self.float_type
                ))

    def feed_forwards(self, input_vector):
        for i in range(len(self.lc)):
            lc = self.lc[i]

            if lc['layer_type'] == 'conv':
                input_vector = tf.nn.conv2d(
                    input=input_vector,
                    filter=lc['filter'],
                    strides=lc['stride_size'],
                    padding='SAME'
                )

            elif lc['layer_type'] == 'pool':
                if lc['pool_type'] == 'average':
                    input_vector = tf.nn.avg_pool(
                        value=input_vector,
                        ksize=lc['pool_size'],
                        strides=lc['stride_size'],
                        padding='SAME'
                    )
                elif lc['pool_type'] == 'max':
                    input_vector = tf.nn.max_pool(
                        value=input_vector,
                        ksize=lc['pool_size'],
                        strides=lc['stride_size'],
                        padding='SAME'
                    )

        return input_vector
