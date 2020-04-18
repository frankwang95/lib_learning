import tensorflow as tf
import numpy as np
import pandas as pd

import plotly.offline as plotly
import plotly.graph_objs as go

from lib_learning.model_monitoring.base_monitoring import BaseMonitor


class ActivationMonitor(BaseMonitor):
    def __init__(self, update_interval, layer):
        self.layer = layer
        init_dict = {'weights': None, 'bias': None}
        super().__init__(init_dict, update_interval)

    def link_to_network(self, network):
        self.network = network
        self.activation_shape = self.network.biases[self.layer].shape.as_list()
        self.values = np.array([]).reshape([0] + self.activation_shape)

        layers = zip(
            self.network.weights[:self.layer + 1],
            self.network.biases[:self.layer + 1],
            self.network.activation[:self.layer + 1]
        )

        self.activation = self.network.input[0:1]
        for w, b, a in layers:
            self.activation = a(tf.matmul(self.activation, w) + b)

        return self

    def evaluate(self):
        activations = self.network.session.run(self.activation)
        self.evaluate_core(activations)

    def evaluate_offline(self, train_in, train_out):
        activations = self.network.session.run(
            self.activation,
            feed_dict={self.network.input: train_in, self.network.train_targets: train_out}
        )
        self.evaluate_core(activations)

    def evaluate_core(self, activations):
        self.values = np.append(
            self.values,
            activations.reshape([1] + self.activation_shape),
            axis=0
        )

    def plot(self, mode='weights'):
        pass
