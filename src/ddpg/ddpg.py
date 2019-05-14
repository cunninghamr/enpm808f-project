import logging
import numpy as np
import os
import random
import tensorflow as tf

from collections import deque
from src.ddpg.actor import Actor
from src.ddpg.critic import Critic

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DDPG:
    def __init__(self, sess, state_dim, action_dim, action_bound, actor_lr=0.0001, critic_lr=0.001, gamma=0.99, tau=0.001, batch_size=120, demo_batch_size=8, is_training=False):
        """
        Create DDPG model.
        :param sess: tensorflow session
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param action_bound: action bound
        :param actor_lr: actor learning rate
        :param critic_lr: critic learning rate
        :param gamma: discount factor
        :param tau: target update factor
        :param batch_size: size of experience batch to sample
        :param demo_batch_size: size of demonstration experience batch to sample
        :param is_training: whether model is training
        """
        logger.info('Creating Model')

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.batch_size = batch_size
        self.demo_batch_size = demo_batch_size
        self.is_training = is_training

        # experience buffers
        self.replay_buffer = deque(maxlen=100000)
        self.demo_replay_buffer = deque(maxlen=5000)

        self.actor = Actor(sess, state_dim, action_dim, action_bound, actor_lr, tau, batch_size)
        self.critic = Critic(sess, state_dim, action_dim, critic_lr, tau, gamma, self.actor.get_num_trainable_vars())

        sess.run(tf.global_variables_initializer())

        # initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

    def seed(self, demo_experiences):
        logger.info('Seeding Demo Experiences')

        for experience in demo_experiences:
            self.remember(*experience, demo=True)

    def action(self, state):
        action = self.actor.predict(np.reshape(state, (1, self.state_dim)))

        # add gaussian noise to action for exploration during training
        if self.is_training:
            action += 0.1 * np.random.randn(self.action_dim)

        # restrict actions to action bound
        action = np.clip(action, -self.action_bound, self.action_bound)

        return action[0]

    def remember(self, state, action, reward, terminal, next_state, demo=False):
        if demo:
            self.demo_replay_buffer.append((np.reshape(state, (self.state_dim,)), np.reshape(action, (self.action_dim,)), reward,
                                            terminal, np.reshape(next_state, (self.state_dim,))))
        else:
            self.replay_buffer.append((np.reshape(state, (self.state_dim,)), np.reshape(action, (self.action_dim,)), reward,
                                       terminal, np.reshape(next_state, (self.state_dim,))))

    def train(self, use_demo=None):
        batch = None

        # sample a batch of experiences from the main and demonstration buffer
        if use_demo:
            if len(self.replay_buffer) >= self.batch_size and len(self.demo_replay_buffer) >= self.demo_batch_size:
                batch = np.concatenate([random.sample(self.replay_buffer, self.batch_size), random.sample(self.demo_replay_buffer, self.demo_batch_size)])
        # only sample a batch of experiences from the main replay buffer
        else:
            if len(self.replay_buffer) >= self.demo_batch_size + self.batch_size:
                batch = random.sample(self.replay_buffer, self.demo_batch_size + self.batch_size)

        # Keep adding experience to the memory until there are at least batch_size samples
        if batch is not None:
            state_batch = np.array([b[0] for b in batch])
            action_batch = np.array([b[1] for b in batch])
            reward_batch = np.array([b[2] for b in batch])
            terminal_batch = np.array([b[3] for b in batch])
            next_state_batch = np.array([b[4] for b in batch])

            # calculate targets
            target_q = self.critic.predict_target(
                next_state_batch, self.actor.predict_target(next_state_batch))

            y_i = []
            for k in range(self.batch_size + self.demo_batch_size):
                if terminal_batch[k]:
                    y_i.append(reward_batch[k])
                else:
                    y_i.append(reward_batch[k] + self.gamma * target_q[k])

            # update the critic given the targets
            predicted_q_value, _ = self.critic.train(
                state_batch, action_batch, np.reshape(y_i, (self.batch_size + self.demo_batch_size, 1)))

            # update the actor policy using the sampled gradient
            a_outs = self.actor.predict(state_batch)
            grads = self.critic.action_gradients(state_batch, a_outs)
            self.actor.train(state_batch, grads[0])

            # update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

            return np.average(predicted_q_value)

    def save(self, path, num, results_dir):
        logger.info('Saving {}'.format(num))

        # save the state of the model and the two replay buffers
        saver = tf.train.Saver(save_relative_paths=True)
        saver.save(self.sess, os.path.join(path, self.name.lower()))
        np.save(os.path.join('results', results_dir, 'replay_buffer_{}.npy'.format(num)), self.replay_buffer)
        np.save(os.path.join('results', results_dir, 'demo_replay_buffer_{}.npy'.format(num)), self.demo_replay_buffer)

    def load(self, ckpt_dir, ckpt_num=None):
        if ckpt_num is None:
            path = os.path.join('results', ckpt_dir)
            ckpt_num = int(max([ckpt for ckpt in os.listdir(path) if os.path.isdir(os.path.join(path, ckpt))]))

        model_path = os.path.join(path, str(ckpt_num), 'ddpg')
        logger.info('Loading {}'.format(model_path))

        # load the state of the model and the two replay buffers
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        replay_buffer_path = os.path.join('results', ckpt_dir, 'replay_buffer_{}.npy'.format(ckpt_num))
        if os.path.isfile(replay_buffer_path):
            logger.info('Loading {}'.format(replay_buffer_path))
            self.replay_buffer = deque(np.load(replay_buffer_path, allow_pickle=True), maxlen=100000)
        demo_replay_buffer_path = os.path.join('results', ckpt_dir, 'demo_replay_buffer_{}.npy'.format(ckpt_num))
        if os.path.isfile(demo_replay_buffer_path):
            logger.info('Loading {}'.format(demo_replay_buffer_path))
            self.demo_replay_buffer = deque(np.load(demo_replay_buffer_path, allow_pickle=True), maxlen=5000)
