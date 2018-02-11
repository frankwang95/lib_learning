import pandas as pd
from learning_lib.nn.monitoring.base_monitoring import BaseMonitor


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
            tf.gradient(
                self.network.loss_val,
                [self.network.weights[self.layer], self.network.biases[self.layer]]
            ),
            feed_dict={self.input: train_in, self.train_targets: train_out}
        )
        weights_gradient_norms = np.linalg.norm(loss_gradients[0])
        bias_gradient_norms = np.linalg.norm(loss_gradients[1])
        self.values.append(
            {
                'epochs': self.network.epochs,
                'weight_gradient': weights_gradient_norms,
                'bias_gradient': bias_gradient_norms
            },
            ignore_index=True
        )
