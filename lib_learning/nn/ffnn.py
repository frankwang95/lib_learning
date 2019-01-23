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
            - init_weight_mean <float>: Weights will be initialized to have this mean.
            - init_weight_stddev <float>: Weights will be initialized to have this standard deviation.
            - init_bias_mean <float>: Biases will be initialized to have this mean.
            - init_bias_stddev <float>: Biases will be initialized to have this standard deviation.
    """

    def create_params(self):
        self.activation = [lc['activation'] for lc in self.lc[1:]]
        self.weights = []
        self.biases = []

        n_nodes_prev_layer = self.lc[0]
        for i in range(len(self.lc) - 1):
            lc = self.lc[i + 1]

            weights = tf.Variable(tf.random_normal(
                shape=[n_nodes_prev_layer, lc['n_nodes']],
                mean=lc['init_weight_mean'],
                stddev=lc['init_weight_stddev'],
                dtype=self.float_type
            ))
            self.weights.append(weights)

            bias = tf.Variable(tf.random_normal(
                shape=[lc['n_nodes']],
                mean=lc['init_bias_mean'],
                stddev=lc['init_bias_stddev'],
                dtype=self.float_type
            ))
            self.biases.append(bias)

            n_nodes_prev_layer = lc['n_nodes']

    def feed_forwards(self, input_vector, start_layer=None, end_layer=None):
        if start_layer is None:
            start_layer = 0
        if end_layer is None:
            end_layer = len(self.lc)

        iterator = zip(
            self.weights[start_layer:end_layer],
            self.biases[start_layer:end_layer],
            self.activation[start_layer:end_layer]
        )

        for w, b, a in iterator:
            input_vector = a(tf.matmul(input_vector, w) + b)
        return input_vector
