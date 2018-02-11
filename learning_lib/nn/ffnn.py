import tensorflow as tf
import numpy as np


class FFNN(object):
    def __init__(self, layer_config, post_proc_function=tf.identity, input_vector=None, session=None, float_type=tf.float32):
        """ An implementation of a simple feed-forward neural network using the low-level
            tensorflow API.

        Inputs
            layer_config <list(ints)>: A list containing dictionaries which describe the structure of the produced
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
            input_vector <tf.Tensor>: An (None, dim_in) shaped tensor used as the input layer to
                the network. This can be used to link the network to more complex structures. If
                not given, a tensorflow placeholder is initialized and used instead.
            session <tf.Session>: A tensorflow session object. This argument is provided as an
                option so that to prevent graph ownership issuses when an FFNN is used as a
                component of a larger network. If not provided, a new interactive session will be
                initialized.

        Attributes
            session
            post_proc_function
            learning_curve
            epochs
            input
            output
            train_targets
            activation
            weights
            biases
        """

        # == Set basic attributes == #
        self.post_proc_function = post_proc_function
        self.learning_curve = []
        self.loss_gradient = []
        self.property = []
        self.epochs = 0

        # == Define session == #
        if session is None:
            self.session = tf.InteractiveSession()
        else:
            self.session = session

        # == Define input and output points == #
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

        # == Initialize variables == #
        self.session.run(tf.global_variables_initializer())


    def train(self, train_in, train_out, loss_func=None, optimizer=None, batch_size=10, epochs=1, report_interval=10):
        """ Train the network weights with provided data. Trained weights can be accessed inside of
            the tensorflow session stored as `self.session`.

        Inputs
            train_in <np.ndarray>: A [n, d] shaped array where n is the number of datapoints and d is size of the
                input dimension.
            train_out <np.ndarray>: A [n, c] shaped array where n is the number of datapoints and c is size of the
                output dimension.
            loss_func <function(tf.Tensor, tf.Tensor -> tf.Tensor)>: A function that will compute a loss value from
                the train inputs after passing through the network and the train outputs. The default is the tensorflow
                implementation of mean squared loss.
            optimizer: A function which performs a optimization step. The default is the tensorflow Graident Descent
                Optimizer with step size 0.1.
            batch_size <int>: The number of data points that will be passed through for each epoch. Default is 10.
            epochs <int>: The number of batches that will be passed through during the train job. Default is 1.
            report_interval <int>: The number of epoches between which report metrics will be computed. Default is 10.
        """
        if loss_func is None: loss_func = tf.losses.mean_squared_error
        if optimizer is None: optimizer = tf.train.GradientDescentOptimizer(0.1)

        loss_val = loss_func(self.output, self.train_targets)
        train_step = optimizer.minimize(loss_val)

        for i in range(epochs):
            in_batch = np.roll(train_in, -batch_size * i, 0)[:batch_size]
            out_batch = np.roll(train_out, -batch_size * i, 0)[:batch_size]
            self.session.run(train_step, feed_dict={self.input: in_batch, self.train_targets: out_batch})

            # Writing report information
            if self.epochs % report_interval == 0:
                self.loss_gradient.append((
                    self.epochs,
                    self.session.run(
                        tf.gradients(loss_val, self.weights + self.biases),
                        feed_dict={self.input: train_in, self.train_targets: train_out}
                    )
                ))
                self.learning_curve.append((
                    self.epochs,
                    self.session.run(
                        loss_val,
                        feed_dict={self.input: train_in, self.train_targets: train_out}
                    )
                ))
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
