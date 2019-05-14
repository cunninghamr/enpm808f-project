import tensorflow as tf
import tflearn


class Actor:
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        """

        :param sess: tensorflow session
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param action_bound: action bound
        :param learning_rate: learning rate
        :param tau: target update factor
        :param batch_size: experience batch size
        """
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # actor network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # target network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # operation for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # this gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

        # combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # optimization operation
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        # state input
        inputs = tflearn.input_data(shape=[None, self.state_dim])

        # hidden layer 1
        net = tflearn.fully_connected(inputs, 400)
        # batch normalizing inputs
        net = tflearn.layers.normalization.batch_normalization(net)
        # relu activation
        net = tflearn.activations.relu(net)

        # hidden layer 2
        net = tflearn.fully_connected(net, 300)
        # batch normalizing inputs
        net = tflearn.layers.normalization.batch_normalization(net)
        # relu activiation
        net = tflearn.activations.relu(net)

        # action output with tanh activation
        out = tflearn.fully_connected(net, self.action_dim, activation='tanh',
                                      weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
        # scale output so action is within action bound
        scaled_out = tf.multiply(out, self.action_bound)

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
