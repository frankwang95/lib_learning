import abc
import dill
import numpy as np



def load_saved_model(model_path):
    """ Loads a BasicModel from a serialized file.
    """
    with open(model_path, 'rb') as f:
        return dill.load(f)


class BasicModel(object):
    """ An implementation of a neural network framework using the low-level tensorflow API. Provides options for
        offline mode using `feed_dict` to load data or an online mode using the `tf.train.Dataset` API.

    Inputs
        layer_config <list>: A list containing the network configuration. Exact specifications will vary by the
            implementation and should be documented there.
        post_proc_function <function(np.array -> tf.Tensor)>: A function which is applied to the output layer at
            compute time but not traintime. One usecase is to apply softmax operations to the output layers in light of
            the fact that cross-entropy/softmax computations are done in tandem to address numerical overflow.
        loss_func <function(np.array, np.array -> np.array)>: A function that will compute a loss value from the
            train inputs after passing through the network and the train outputs. The default is the tensorflow
            implementation of mean squared loss.
        optimizer <Optimizer>: A function which performs a optimization step. The default is the autograd Gradient
            Descent Optimizer with step size 0.1.
        monitors <list(BaseMonitor)>: Gives a list of monitoring objects which will be tracked during training.
    """
    def __init__(model_config, post_proc_function, loss_func, optimizer, monitors):
        self.mc = model_config
        self.monitors = monitors
        self.loss_func = loss_func

        if post_proc_function is None:
            self.post_proc_function = np.identity
        else:
            self.post_proc_function = post_proc_function

        self.optimizer = optimizer


    @abc.abstractmethod
    def create_params(self):
        return


    @abc.abstractmethod
    def feed_forwards(self, input_vector):
        return


    def train(self, train_in, train_out, batch_size=10, steps=1):
        for _ in range(steps):



    def predict(self, in_vector):
        return self.feed_forwards(input_vector)


    def save(self, path):
        """ Serializes the model represented by an instance of the SimpleRecurrentModel class into a file which can be
            recovered using the `load_saved_model` function.
        """
        with open(file_path, 'wb') as f:
            dill.dump(self, f)
