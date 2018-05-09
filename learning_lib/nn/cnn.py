import tensorflow as tf
import numpy as np
from learning_lib.nn.nn_base_class import NN

"""

}"""

class CNN(NN):
    """ An implementation of a 2D convolutional neural network class using our NN base class.

    Inputs
        layer_config <list>: A list containing dictionaries which describe the structure of the produced
            network. Each dictionary should contain the key 'layer_type' which maps to a string value of
            either 'conv' or 'pool' depending on if you would like the layer to be a convolutional layer
            or a pooling layer. The remaining parameters depend on which value this key takes:

            layer_type == reshape:
                new_shape

            layer_type == connected:
                input_dim
                output_dim
                activation
                init_weight_mean
                init_weight_stddev
                init_bias_mean
                init_bias_stddev
                ====================
                weights
                biases

            layer_type == conv:
                filter_size
                init_filter_mean
                init_filter_stddev
                stride_size
                activation
                ====================
                filter

            layer_type == conv_transpose:
                filter_size
                init_filter_mean
                init_filter_stddev
                stride_size
                activation
                output_size -- to be depreciated
                ====================
                filter

            layer_type == pool:
                pool_type
                pool_size
                stride_size
    """

    def create_params(self):
        for i in range(len(self.lc)):
            lc = self.lc[i]

            if lc['layer_type'] in ['conv', 'conv_transpose']:
                lc['filter'] = tf.Variable(tf.random_normal(
                    shape=lc['filter_size'],
                    mean=lc['init_filter_mean'],
                    stddev=lc['init_filter_stddev'],
                    dtype=self.float_type
                ))

            elif lc['layer_type'] == 'connected':
                lc['weights']  = tf.Variable(tf.random_normal(
                    shape=[lc['input_dim'], lc['output_dim']],
                    mean=lc['init_weight_mean'],
                    stddev=lc['init_weight_stddev'],
                    dtype=self.float_type
                ))
                lc['biases'] = tf.Variable(tf.random_normal(
                    shape=[lc['output_dim']],
                    mean=lc['init_bias_mean'],
                    stddev=lc['init_bias_stddev'],
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
                    padding='VALID'
                )
                input_vector = lc['activation'](input_vector)

            elif lc['layer_type'] == 'conv_transpose':
                input_vector = tf.nn.conv2d_transpose(
                    value=input_vector,
                    filter=lc['filter'],
                    output_shape=lc['output_size'],
                    strides=lc['stride_size']
                )
                input_vector = lc['activation'](input_vector)

            elif lc['layer_type'] == 'connected':
                a = lc['activation']
                w = lc['weights']
                b = lc['biases']
                input_vector = a(tf.matmul(input_vector, w) + b)

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

            elif lc['layer_type'] == 'reshape':
                input_vector = tf.reshape(input_vector, [-1] + lc['new_shape'])

        return input_vector
