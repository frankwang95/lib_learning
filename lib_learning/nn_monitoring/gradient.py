import tensorflow as tf
import numpy as np
import pandas as pd

import plotly.offline as plotly
import plotly.graph_objs as go

from learning_lib.nn.monitoring.base_monitoring import BaseMonitor


class LossGradientMonitor(BaseMonitor):
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
        self.gradients = tf.gradients(
            self.network.loss_val,
            [self.network.weights[self.layer], self.network.biases[self.layer]]
        )
        return self

    def evaluate(self):
        loss_gradients = self.network.session.run(self.gradients)
        self.evaluate_core(loss_gradients)

    def evaluate_offline(self, train_in, train_out):
        loss_gradients = self.network.session.run(
            self.gradients,
            feed_dict={self.network.input: train_in, self.network.train_targets: train_out}
        )
        self.evaluate_core(loss_gradients)

    def evaluate_core(self, loss_gradients):
        self.values['weights'] = np.append(
            self.values['weights'],
            loss_gradients[0].reshape([1] + self.weight_shape),
            axis=0
        )
        self.values['bias'] = np.append(
            self.values['bias'],
            loss_gradients[1].reshape([1] + self.bias_shape),
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


class LossGradientNormMonitor(BaseMonitor):
    def __init__(self, update_interval, layer):
        self.layer = layer
        init_df = pd.DataFrame({
            'epochs': [],
            'weight_gradient': [],
            'bias_gradient': []
        })
        super().__init__(init_df, update_interval)
        self.n_weights = None
        self.n_biases = None

    def link_to_network(self, network):
        self.network = network
        self.n_weights = int(np.prod(self.network.weights[self.layer].shape))
        self.n_biases = int(np.prod(self.network.biases[self.layer].shape))
        self.gradients = tf.gradients(
            self.network.loss_val,
            [self.network.weights[self.layer], self.network.biases[self.layer]]
        )
        return self

    def evaluate(self):
        loss_gradients = self.network.session.run(self.gradients)
        self.evaluate_core(loss_gradients)

    def evaluate_offline(self, train_in, train_out):
        loss_gradients = self.network.session.run(
            self.gradients,
            feed_dict={self.network.input: train_in, self.network.train_targets: train_out}
        )
        self.evaluate_core(loss_gradients)

    def evaluate_core(self, loss_gradients):
        weights_gradient_norms = np.linalg.norm(loss_gradients[0]) / self.n_weights
        bias_gradient_norms = np.linalg.norm(loss_gradients[1]) / self.n_biases
        self.values = self.values.append(
            {
                'epochs': self.network.epochs,
                'weight_gradient': weights_gradient_norms,
                'bias_gradient': bias_gradient_norms
            },
            ignore_index=True
        )

    def plot(self):
        plotly.init_notebook_mode(connected=True)
        layout = go.Layout(
            title='Gradient Norm vs. Epochs',
            xaxis=dict(title='Epochs'),
            yaxis=dict(title='Gradient Norm')
        )
        data = [
            go.Scatter(
                x=self.values['epochs'],
                y=self.values['weight_gradient'],
                name='Weight Gradient Norms'
            ),
            go.Scatter(
                x=self.values['epochs'],
                y=self.values['bias_gradient'],
                name='Bias Gradient Norms'
            )
        ]
        return go.Figure(data=data, layout=layout)
