import tensorflow as tf
import numpy as np
import argparse
import cv2
import gym, roboschool
import copy
import os
import constants

from rlsaber.replay_buffer import ReplayBuffer
from rlsaber.trainer import Trainer, Evaluator, Recorder
from rlsaber.env import EnvWrapper
from rlsaber.log import TfBoardLogger, JsonLogger
from rlsaber.log import dump_constants, restore_constants
from network import make_actor_network, make_critic_network, make_value_network
from agent import Agent
from datetime import datetime


def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--log', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--load-constants', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--reward_scale', type=float, default=0.1)
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/{}'.format(args.log))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logdir = os.path.join(os.path.dirname(__file__), 'logs/{}'.format(args.log))

    # load constant settings
    if args.load_constants is not None:
        restored_constants = restore_constants(args.load_constants)
        # save constant settings
        dump_constants(restored_constants, os.path.join(outdir, 'constants.json'))
    else:
        # this condition expression prevents python interpreter bug
        dump_constants(constants, os.path.join(outdir, 'constants.json'))

    env_ = gym.make(args.env)

    env_.seed(seed=args.seed)

    env = EnvWrapper(
        env=env_,
        r_preprocess=lambda r: r * args.reward_scale
    )

    np.random.seed(seed=1)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    sess = tf.Session()
    sess.__enter__()
    tf.set_random_seed(seed=args.seed)

    actor = make_actor_network(constants.ACTOR_HIDDENS)
    critic = make_critic_network(constants.CRITIC_HIDDENS)
    value = make_value_network(constants.VALUE_HIDDENS)
    replay_buffer = ReplayBuffer(constants.BUFFER_SIZE)

    agent = Agent(
        actor,
        critic,
        value,
        obs_dim,
        n_actions,
        replay_buffer,
        batch_size=constants.BATCH_SIZE,
        action_scale=env.action_space.high,
        gamma=constants.GAMMA,
        tau=constants.TAU,
        actor_lr=constants.ACTOR_LR,
        critic_lr=constants.CRITIC_LR,
        value_lr=constants.VALUE_LR,
        reg_factor=constants.REG_FACTOR
    )

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    # tensorboard logger
    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    tflogger = TfBoardLogger(train_writer)
    tflogger.register('reward', dtype=tf.float32)
    tflogger.register('eval_reward', dtype=tf.float32)
    # json logger
    jsonlogger = JsonLogger(os.path.join(outdir, 'reward.json'))

    def end_episode(reward, step, episode):
        tflogger.plot('reward', reward, step)
        jsonlogger.plot(reward=reward, step=step, episode=episode)

    def after_action(state, reward, global_step, local_step):
        if global_step > 0 and global_step % constants.MODEL_SAVE_INTERVAL == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=global_step)

    evaluator = Evaluator(
        env=copy.deepcopy(env),
        state_shape=[obs_dim],
        state_window=1,
        eval_episodes=10,
        recorder=Recorder(outdir) if args.record else None,
        record_episodes=3
    )
    should_eval = lambda step, episode: step > 0 and step % 10000 == 0
    end_eval = lambda s, e, r: tflogger.plot('eval_reward', np.mean(r), s)

    trainer = Trainer(
        env=env,
        agent=agent,
        render=args.render,
        state_shape=[obs_dim],
        state_window=1,
        final_step=constants.FINAL_STEP,
        end_episode=end_episode,
        after_action=after_action,
        training=not args.demo,
        evaluator=evaluator,
        should_eval=should_eval,
        end_eval=end_eval
    )

    trainer.start()

if __name__ == '__main__':
    main()
