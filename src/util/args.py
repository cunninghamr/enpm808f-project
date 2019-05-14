import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description='Train and/or test a hexapod learning to walk.')

    # environment
    parser.add_argument('--scene', type=str, default='empty', help='name of scene to load')
    parser.add_argument('--render', action='store_true', help='show the environment')
    parser.add_argument('--model', type=str, default='hexapod', help='name of model to use for training and testing')
    parser.add_argument('--demo_model', type=str, default='hexapod_demo', help='name of model to use for demonstration learning')

    # common
    parser.add_argument('--ckpt_dir', type=str, default=None, help='name of checkpoint directory')
    parser.add_argument('--load_ckpt_num', type=int, default=None, help='checkpoint to load')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes to run')
    parser.add_argument('--num_episode_steps', type=int, default=200, help='maximum number of steps per episode')

    # training arguments
    parser.add_argument('--train', action='store_true', help='trains a model')

    # testing arguments
    parser.add_argument('--test', action='store_true', help='tests a model')

    # seeding arguments
    parser.add_argument('--demo', action='store_true', help='demonstrate a pre-programmed gait to seed the model')

    return parser.parse_args(raw_args)
