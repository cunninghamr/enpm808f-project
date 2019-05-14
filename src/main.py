import logging
import numpy as np
import os
import tensorflow as tf
import time

from src.ddpg.ddpg import DDPG
from src.hexapod.env import Env
from src.hexapod.robot import Robot
from src.util.args import get_args
from src.util.metrics import Metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(raw_args=None):
    args = get_args(raw_args)

    with tf.Session() as session:
        # create the model
        model = DDPG(session, Robot.STATE_DIM, Robot.ACTION_DIM, Robot.ACTION_BOUND, is_training=args.train)

        # load a previously saved model
        if not args.demo and args.ckpt_dir is not None:
            model.load(args.ckpt_dir, args.load_ckpt_num)

        ckpt_num = args.load_ckpt_num + 1 if args.load_ckpt_num is not None else 1

        # only seed demonstrations for a new model
        if args.train and ckpt_num == 1:
            model.seed(np.load(os.path.join('results', args.ckpt_dir, 'demo_experiences.npy'), allow_pickle=True))

        demo_experiences = []

        num_batches = int(args.num_episodes / (100 + 1)) + 1

        for batch in range(num_batches):
            # load environment
            with Env(args.scene, args.render) as env:
                # create and load robot
                robot = Robot(env.client_id, args.demo_model if args.demo else args.model)

                metrics = Metrics(args.ckpt_dir)

                for episode in range(1 + (batch * 100), 1 + (batch * 100) + (100 if (batch + 1) < num_batches else args.num_episodes)):
                    start_time = time.time()

                    episode_reward = 0
                    episode_max_q = 0
                    num_steps = 0

                    for step in range(args.num_episode_steps):
                        state = robot.get_state()
                        position = robot.position

                        # take an action according to the model's policy
                        if not args.demo:
                            action = model.action(state)

                            robot.act(action)

                        env.step()

                        robot.sense()

                        next_state = robot.get_state()
                        next_position = robot.position

                        # calculate the taken action between states for the demo robot
                        if args.demo:
                            configuration_change = np.array(next_state) - np.array(state)

                            action = configuration_change[:robot.ACTION_DIM]

                        position_change = np.array(next_position) - np.array(position)

                        # reward for distance travelled in x direction
                        reward = -position_change[0]

                        # limit noise from V-REP
                        if args.demo:
                            if abs(reward) < 0.001:
                                reward = 0
                        if args.train:
                            if abs(reward) < 0.001:
                                reward = 0

                        # scale reward
                        reward *= 10

                        # subtract small time punishment so robot does not stand still
                        reward -= 0.001

                        # no terminal case (could add check for robot body hitting ground or tipping over)
                        terminal = False

                        # add to the demonstration experiences
                        if args.demo:
                            demo_experiences.append((state, action, reward, terminal, next_state))

                        # add experience and train model
                        if args.train:
                            model.remember(state, action, reward, terminal, next_state)
                            max_q = model.train(use_demo=False)

                            if max_q is not None:
                                episode_max_q += max_q

                        episode_reward += reward

                        num_steps += 1

                    metrics.num_episodes += 1
                    metrics.episode_reward.append(episode_reward)
                    metrics.episode_avg_max_q.append(episode_max_q / num_steps)

                    # reset environment and robot for next episode
                    if episode < args.num_episodes:
                        env.reset()

                        robot.reset()

                    logger.info('| Mode: {} | Episode {:d} | Steps: {:d} | Time/Step: {:.2f} | Reward: {:f} | Qmax: {:.4f}'
                                .format('Demo' if args.demo else 'Train' if args.train else 'Test' if args.test else '?',
                                        episode, num_steps, (time.time() - start_time) / num_steps, episode_reward,
                                        episode_max_q / num_steps))

                    # save model checkpoint
                    if args.train:
                        if episode % 50 == 0:
                            metrics.save(ckpt_num)
                            metrics = Metrics(args.results_dir)
                            model.save(os.path.join('results', metrics.name, str(ckpt_num)), ckpt_num, args.results_dir)
                            ckpt_num += 1

            try:
                env.stop()
            except:
                pass

        # save demo experiences
        if args.demo:
            if not os.path.isdir('results'):
                os.mkdir('results')
            if not os.path.isdir(os.path.join('results', args.results_dir)):
                os.mkdir(os.path.join('results', args.results_dir))
            np.save(os.path.join('results', args.results_dir, 'demo_experiences.npy'), np.asarray(demo_experiences))

        logger.info('Done')


if __name__ == '__main__':
    main()
