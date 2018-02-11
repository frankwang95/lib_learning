import tensorflow as tf
import numpy as np


class FFNN(object):
    def __init__(self, layer_config, post_proc_function=None, loss_func=None, input_vector=None, session=None, float_type=None, monitors=[]):
        """ An implementation of a simple feed-forward neural network using the low-level
            tensorflow API.

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
            post_proc_function <function(tf.Tensor -> tf.Tensor)>: A function which is applied to the output layer at
                compute time but not traintime. One usecase is to apply softmax operations to the output layers in
                light of the fact that cross-entropy/softmax computations are done in tandem to address issues with
                numerical overflow.
            loss_func <function(tf.Tensor, tf.Tensor -> tf.Tensor)>: A function that will compute a loss value from
                the train inputs after passing through the network and the train outputs. The default is the tensorflow
                implementation of mean squared loss.
            input_vector <tf.Tensor>: An (None, dim_in) shaped tensor used as the input layer to
                the network. This can be used to link the network to more complex structures. If
                not given, a tensorflow placeholder is initialized and used instead.
            session <tf.Session>: A tensorflow session object. This argument is provided as an
                option so that to prevent graph ownership issuses when an FFNN is used as a
                component of a larger network. If not provided, a new interactive session will be
                initialized.
            float_type <tf.float32, tf.float64>: Gives the numerical type which internal values will be initalized to.
            monitors <list(BaseMonitor)>: Gives a list of monitoring objects which will be tracked during training.

        Attributes
            lc
            session
            epochs
            monitors

            input
            train_targets
            output
            loss_val

            float_type
            weights
            biases
            activation
            post_proc_function
        """

        # == Set basic attributes == #
        self.lc = lc
        if post_proc_function is None:
            self.post_proc_function = tf.identity
        else:
            self.post_proc_function = post_proc_function
        if self.float_type is None:
            self.float_type = tf.float32
        else:
            self.float_type = float_type
        self.monitors = [m.link_to_network(self) for m in monitors]
        self.epochs = 0

        # == Define session == #
        if session is None:
            self.session = tf.InteractiveSession()
        else:
            self.session = session

        # == Define input and output handles == #
        if input_vector is None:
            self.input = tf.placeholder(float_type, [None, layer_config[0]])
        else:
            self.input = input_vector
        self.output = self.input
        self.train_targets = tf.placeholder(float_type, [None, layer_config[-1]['n_nodes']])

        # == Create weights and Biases == #
        self.activation = [lc['activation'] for lc in layer_config[1:]]
        self.weights = []
        self.biases = []

        n_nodes_prev_layer = layer_config[0]
        for i in range(len(layer_config) - 1):
            lc = layer_config[i + 1]

            weights = tf.Variable(tf.random_uniform(
                shape=[n_nodes_prev_layer, lc['n_nodes']],
                minval=lc['init_weight_lower'],
                maxval=lc['init_weight_upper'],
                dtype=float_type
            ))
            self.weights.append(weights)

            bias = tf.Variable(tf.random_uniform(
                shape=[lc['n_nodes']],
                minval=lc['init_bias_lower'],
                maxval=lc['init_bias_upper'],
                dtype=float_type
            ))
            self.biases.append(bias)

            n_nodes_prev_layer = lc['n_nodes']

        # == Construct the model calculation in the graph == #
        for w, b, a in zip(self.weights, self.biases, self.activation):
            self.output = a(tf.matmul(self.output, w) + b)

        # == Define handles for accessing loss == #
        if loss_func is None:
            self.loss_func = tf.losses.mean_squared_error
        else:
            self.loss_func = loss_func
        self.loss_val = self.loss_func(self.output, self.train_targets)

        # == Initialize variables == #
        self.session.run(tf.global_variables_initializer())


    def train(self, train_in, train_out, optimizer=None, batch_size=10, epochs=1):
        """ Train the network weights with provided data. Trained weights can be accessed inside of
            the tensorflow session stored as `self.session`.

        Inputs
            train_in <np.ndarray>: A [n, d] shaped array where n is the number of datapoints and d is size of the
                input dimension.
            train_out <np.ndarray>: A [n, c] shaped array where n is the number of datapoints and c is size of the
                output dimension.
            optimizer: A function which performs a optimization step. The default is the tensorflow Graident Descent
                Optimizer with step size 0.1.
            batch_size <int>: The number of data points that will be passed through for each epoch. Default is 10.
            epochs <int>: The number of batches that will be passed through during the train job. Default is 1.
        """

        if optimizer is None: optimizer = tf.train.GradientDescentOptimizer(0.1)
        train_step = optimizer.minimize(self.loss_val)

        for i in range(epochs):
            in_batch = np.roll(train_in, -batch_size * i, 0)[:batch_size]
            out_batch = np.roll(train_out, -batch_size * i, 0)[:batch_size]
            self.session.run(train_step, feed_dict={self.input: in_batch, self.train_targets: out_batch})

            # Generate reporting information
            for m in self.monitors:
                if m.check_update():
                    m.evaluate()

            print("Reached epoch {}".format(self.epochs))
            self.epochs += 1


    def evaluate(self, in_vector):
        """ Runs the model on a numpy array representing a collection of input data

        Inputs
            in_vector <np.ndarray>: A [n, d] array where n is the number of vectors and d is the size of the
                input dimension on which the neural network will be computed.
        Returns
            <np.ndarray>: An array returning the output of the neural network from the given input.
        """
        return self.session.run(self.post_proc_function(self.output), feed_dict={self.input: in_vector})
