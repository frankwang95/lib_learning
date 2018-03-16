import tensorflow as tf
import abc


class NN(object):
    def __init__(
        self, layer_config, post_proc_function=None, loss_func=None, optimizer=None, input_vector=None,
        train_targets_vector=None, session=None, float_type=None, monitors=[]
    ):
        """ An implementation of a neural network framework using the low-level tensorflow API. Provides options for
            offline mode using `feed_dict` to load data or an online mode using the `tf.train.Dataset` API.

        Inputs
            layer_config <list>: A list containing the network configuration. Exact specifications will vary by the
                implementation and should be documented there.
            post_proc_function <function(tf.Tensor -> tf.Tensor)>: A function which is applied to the output layer at
                compute time but not traintime. One usecase is to apply softmax operations to the output layers in
                light of the fact that cross-entropy/softmax computations are done in tandem to address issues with
                numerical overflow.
            loss_func <function(tf.Tensor, tf.Tensor -> tf.Tensor)>: A function that will compute a loss value from
                the train inputs after passing through the network and the train outputs. The default is the tensorflow
                implementation of mean squared loss.
            optimizer <tf.Optimizer>: A function which performs a optimization step. The default is the tensorflow
                Graident Descent Optimizer with step size 0.1.
            input_vector <tf.Tensor>: An (None, dim_in) shaped tensor used as the input layer to
                the network. This can be used to link the network to more complex structures or to attach the network
                to tf.DataSet objects. If not given, a tensorflow placeholder is initialized and used instead.
            train_targets_vector <tf.Tensor>: A tensor from a tf.DataSet object used to feed in training labels for
                online training in conjunction with `input_vector`.
            session <tf.Session>: A tensorflow session object. This argument is provided as an
                option so that to prevent graph ownership issuses when an FFNN is used as a
                component of a larger network. If not provided, a new interactive session will be
                initialized.
            float_type <tf.float32, tf.float64>: Gives the numerical type which internal values will be initalized to.
            monitors <list(BaseMonitor)>: Gives a list of monitoring objects which will be tracked during training.
        """
        self.lc = layer_config
        self.epochs = 0

        if post_proc_function is None:
            self.post_proc_function = tf.identity
        else:
            self.post_proc_function = post_proc_function
        if float_type is None:
            self.float_type = tf.float32
        else:
            self.float_type = float_type
        if session is None:
            self.session = tf.InteractiveSession()
        else:
            self.session = session
        if loss_func is None:
            self.loss_func = tf.losses.mean_squared_error
        else:
            self.loss_func = loss_func
        if optimizer is None:
            self.optimizer = tf.train.GradientDescentOptimizer(0.1)
        else:
            self.optimizer = optimizer

        # == Architecture == #
        self.create_params()

        if input_vector is None:
            self.input = tf.placeholder(self.float_type, [None, layer_config[0]])
        else:
            self.input = input_vector
        self.output = self.feed_forwards(self.input)

        # == Training == #
        if train_targets_vector is None:
            self.train_targets = tf.placeholder(self.float_type, [None, layer_config[-1]['n_nodes']])
        else:
            self.train_targets = train_targets_vector
        self.loss_val = self.loss_func(self.output, self.train_targets)
        self.train_step = self.optimizer.minimize(self.loss_val)

        # == Monitors == #
        self.monitors = [m.link_to_network(self) for m in monitors]

        # == Initialize Variables == #
        # TODO: Initialize only the variables initialized in this model
        self.session.run(tf.global_variables_initializer())

    @abc.abstractmethod
    def create_params(self):
        pass

    @abc.abstractmethod
    def feed_forwards(self, input_vector):
        pass

    def train_online(self):
        """ Train the network weights by running the defined train step under the assumption that the neural network
            input tensor is linked the batching output of a tf.data.DataSet iterator.
        """
        while True:
            try:
                self.session.run(self.train_step)

                # Generate reporting information
                for m in self.monitors:
                    if m.check_update():
                        m.evaluate()
            except tf.errors.OutOfRangeError:
                break

            self.epochs += 1

    def train_offline(self, train_in, train_out, batch_size=10, epochs=1):
        """ Train the network weights with provided numpy array data using `feed_dict` to send the data to the
            tensorflow runtime.

        Inputs
            train_in <np.ndarray>: A [n, d] shaped array where n is the number of datapoints and d is size of the
                input dimension.
            train_out <np.ndarray>: A [n, c] shaped array where n is the number of datapoints and c is size of the
                output dimension.
            batch_size <int>: The number of data points that will be passed through for each epoch. Default is 10.
            epochs <int>: The number of batches that will be passed through during the train job. Default is 1.
        """
        for i in range(epochs):
            in_batch = np.roll(train_in, -batch_size * i, 0)[:batch_size]
            out_batch = np.roll(train_out, -batch_size * i, 0)[:batch_size]
            self.session.run(
                self.train_step,
                feed_dict={self.input: in_batch, self.train_targets: out_batch}
            )

            # Generate reporting information
            for m in self.monitors:
                if m.check_update():
                    m.evaluate_offline(in_batch, out_batch)

            self.epochs += 1
            print("\rEpoch: {}".format(self.epochs), end='')
        print()


    def predict(self, in_vector):
        """ Runs the model on a numpy array representing a collection of input data using `feed_dict` to send the data
            to the tensorflow runtime.

        Inputs
            in_vector <np.ndarray>: A [n, d] array where n is the number of vectors and d is the size of the
                input dimension on which the neural network will be computed.
        Returns
            <np.ndarray>: An array returning the output of the neural network from the given input.
        """
        return self.session.run(self.post_proc_function(self.output), feed_dict={self.input: in_vector})
