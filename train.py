import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from lightsaber.rl.replay_buffer import ReplayBuffer
from lightsaber.rl.trainer import Trainer
from lightsaber.rl.env_wrapper import EnvWrapper
from lightsaber.tensorflow.log import TfBoardLogger
from network import make_actor_network, make_critic_network, make_value_network
from agent import Agent
from datetime import datetime


def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--log', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-steps', type=int, default=10 ** 7)
    parser.add_argument('--episode-update', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/{}'.format(args.log))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logdir = os.path.join(os.path.dirname(__file__), 'logs/{}'.format(args.log))

    env = EnvWrapper(
        env=gym.make(args.env),
        r_preprocess=lambda r: r / 10.0
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    sess = tf.Session()
    sess.__enter__()

    actor = make_actor_network([128, 128])
    critic = make_critic_network([128, 128])
    value = make_value_network([128, 128])
    replay_buffer = ReplayBuffer(10 ** 6)

    agent = Agent(actor, critic, value, obs_dim, n_actions, replay_buffer)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    tflogger = TfBoardLogger(train_writer)
    tflogger.register('reward', dtype=tf.float32)

    end_episode = lambda r, s, e: tflogger.plot('reward', r, s)
    def after_action(state, reward, global_step, local_step):
        if global_step > 0 and global_step % 10 * 5 == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=global_step)

    trainer = Trainer(
        env=env,
        agent=agent,
        render=args.render,
        state_shape=[obs_dim],
        state_window=1,
        final_step=args.final_steps,
        end_episode=end_episode,
        after_action=after_action,
        training=not args.demo
    )

    trainer.start()

if __name__ == '__main__':
    main()
