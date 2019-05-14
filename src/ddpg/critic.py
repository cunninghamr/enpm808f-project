import tensorflow as tf
import tflearn


class Critic:
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        """

        :param sess: tensorflow session
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate: learning rate
        :param tau: target update factor
        :param gamma: discount factor
        :param num_actor_vars:
        """
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # target network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # operation for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # loss and optimization operations
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        # state input
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        # hidden layer 1
        net = tflearn.fully_connected(inputs, 400)
        # batch normalizing input
        net = tflearn.layers.normalization.batch_normalization(net)
        # relu activation
        net = tflearn.activations.relu(net)

        # action inputs
        action = tflearn.input_data(shape=[None, self.action_dim])

        # merge input and action
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # Q value output
        out = tflearn.fully_connected(net, 1, weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))

        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
