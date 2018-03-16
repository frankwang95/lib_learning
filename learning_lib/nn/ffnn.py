import tensorflow as tf
import numpy as np
from learning_lib.nn.nn_base_class import NN


class FFNN(NN):
    """ An implementation of a simple feed-forward neural network using our NN base class.

    Inputs
        layer_config <list>: A list containing dictionaries which describe the structure of the produced
            network. The first element of the list should simply be a integer specifying the size of the input
            dimension. The remaining items should be dictionaries each containing the following:
            - n_nodes <int>: The number of nodes in the layer
            - activation <function(tf.Tensor -> tf.Tensor)>: A function mapping tensors to tensors
                which will be used as the activation function for the layer.
            - init_weight_lower <float>: The lower bound of the initialization for the weights.
            - init_weight_upper <float>: The upper bound of the initialization for the weights.
            - init_bias_lower <float>: The lower bound of the initialization for the bias terms.
            - init_bias_upper <float>: The upper bound of the initialization for the bias terms.
    """
    def create_params(self):
        self.activation = [lc['activation'] for lc in self.lc[1:]]
        self.weights = []
        self.biases = []

        n_nodes_prev_layer = self.lc[0]
        for i in range(len(self.lc) - 1):
            lc = self.lc[i + 1]

            weights = tf.Variable(tf.random_uniform(
                shape=[n_nodes_prev_layer, lc['n_nodes']],
                minval=lc['init_weight_lower'],
                maxval=lc['init_weight_upper'],
                dtype=self.float_type
            ))
            self.weights.append(weights)

            bias = tf.Variable(tf.random_uniform(
                shape=[lc['n_nodes']],
                minval=lc['init_bias_lower'],
                maxval=lc['init_bias_upper'],
                dtype=self.float_type
            ))
            self.biases.append(bias)

            n_nodes_prev_layer = lc['n_nodes']

    def feed_forwards(self, input_vector):
        for w, b, a in zip(self.weights, self.biases, self.activation):
            input_vector = a(tf.matmul(input_vector, w) + b)
        return input_vector
