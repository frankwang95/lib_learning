import tensorflow as tf
import numpy as np

import plotly.offline as plotly
import plotly.graph_objs as go

from lib_learning.model_monitoring.base_monitoring import BaseMonitor


class WeightMonitor(BaseMonitor):
    def __init__(self, update_interval, layer):
        self.layer = layer
        init_dict = {'weights': None, 'bias': None}
        super().__init__(init_dict, update_interval)

    def link_to_network(self, network):
        self.network = network
        self.weight_shape = self.network.weights[self.layer].shape.as_list()
        self.bias_shape = self.network.biases[self.layer].shape.as_list()
        self.values['weights'] = np.array([]).reshape([0] + self.weight_shape)
        self.values['bias'] = np.array([]).reshape([0] + self.bias_shape)
        return self

    def evaluate(self):
        fetches = {
            'weights': self.network.weights[self.layer],
            'biases': self.network.biases[self.layer]
        }
        weights = self.network.session.run(fetches)
        self.evaluate_core(weights)

    def evaluate_offline(self, train_in, train_out):
        fetches = {
            'weights': self.network.weights[self.layer],
            'biases': self.network.biases[self.layer]
        }
        weights = self.network.session.run(
            fetches,
            feed_dict={self.network.input: train_in, self.network.train_targets: train_out}
        )
        self.evaluate_core(weights)

    def evaluate_core(self, weights):
        self.values['weights'] = np.append(
            self.values['weights'],
            weights['weights'].reshape([1] + self.weight_shape),
            axis=0
        )
        self.values['bias'] = np.append(
            self.values['bias'],
            weights['biases'].reshape([1] + self.bias_shape),
            axis=0
        )

    def plot(self, mode='weights'):
        plotly.init_notebook_mode(connected=True)
        if mode=='weights':
            self.plot_weights()
        elif mode=='bias':
            self.plot_bias()

    def plot_weights(self):
        pass

    def plot_bias(self):
        pass
