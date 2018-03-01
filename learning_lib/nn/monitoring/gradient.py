import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
import plotly.offline as plotly
import plotly.graph_objs as go

from learning_lib.nn.monitoring.base_monitoring import BaseMonitor


def visualize_grad(grad_monitor, kind, animated=True, scale_factor):
    '''
    '''
    scale_factor = grad_monitor.values[kind].max()
    image = (255 * grad_monitor.values[kind] / scale_factor).astype('uint8')
    image = np.repeat(image, scale_factor, axis=1)
    image = np.repeat(image, scale_factor, axis=2)

    if kind == 'bias':
        return Image
    else:
        # Generate GIF



class LossGradientMonitor(BaseMonitor):
    def __init__(self, update_interval, layer):
        self.layer = layer
        init_dict = {'weights': None, 'bias': None}
        super().__init__(init_dict, update_interval)

    def link_to_network(self, network):
        self.weight_shape = network.weights[self.layer].shape.as_list()
        self.bias_shape = network.biases[self.layer].shape.as_list()
        self.values['weights'] = np.array([]).reshape([0] + self.weight_shape)
        self.values['bias'] = np.array([]).reshape([0] + self.bias_shape)
        return super().link_to_network(network)

    def evaluate(self, train_in, train_out):
        loss_gradients = self.network.session.run(
            tf.gradients(
                self.network.loss_val,
                [self.network.weights[self.layer], self.network.biases[self.layer]]
            ),
            feed_dict={self.network.input: train_in, self.network.train_targets: train_out}
        )
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


class LossGradientNormMonitor(BaseMonitor):
    def __init__(self, update_interval, layer):
        self.layer = layer
        init_df = pd.DataFrame({
            'epochs': [],
            'weight_gradient': [],
            'bias_gradient': []
        })
        super().__init__(init_df, update_interval)

    def evaluate(self, train_in, train_out):
        loss_gradients = self.network.session.run(
            tf.gradients(
                self.network.loss_val,
                [self.network.weights[self.layer], self.network.biases[self.layer]]
            ),
            feed_dict={self.network.input: train_in, self.network.train_targets: train_out}
        )
        weights_gradient_norms = np.linalg.norm(loss_gradients[0])
        bias_gradient_norms = np.linalg.norm(loss_gradients[1])
        self.values = self.values.append(
            {
                'epochs': self.network.epochs,
                'weight_gradient': weights_gradient_norms,
                'bias_gradient': bias_gradient_norms
            },
            ignore_index=True
        )

    def plot(self):
        layout = go.Layout(
            title='Gradient Norm vs. Epochs',
            xaxis=dict(title='Epochs'),
            yaxis=dict(title='Gradient Norm')
        )
        data = [
            go.Scatter(
                x=self.values['epochs'],
                y=welf.values['weight_gradient'],
                name='Weight Gradient Norms'
            ),
            go.Scatter(
                x=self.values['epochs'],
                y=welf.values['bias_gradient'],
                name='Bias Gradient Norms'
            )
        ]
        fig = go.Figure(data=data, layout=layout)
        plotly.iplot(fig)
