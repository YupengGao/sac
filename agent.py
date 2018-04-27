import network
from build_graph import build_graph
import numpy as np
import tensorflow as tf


class Agent(object):
    def __init__(self,
                 actor,
                 critic,
                 value,
                 obs_dim,
                 num_actions,
                 replay_buffer,
                 batch_size=4,
                 action_scale=2.0,
                 gamma=0.9,
                 tau=0.01,
                 actor_lr=3*1e-3,
                 critic_lr=3*1e-3,
                 value_lr=3*1e-3,
                 reg_factor=1e-3):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_scale = action_scale
        self.last_obs = None
        self.t = 0
        self.replay_buffer = replay_buffer

        self._act,\
        self._train_actor,\
        self._train_critic,\
        self._train_value,\
        self._update_target = build_graph(
            actor=actor,
            critic=critic,
            value=value,
            obs_dim=obs_dim,
            num_actions=num_actions,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            value_lr=value_lr,
            reg_factor=reg_factor
        )

        self.actor_errors = []
        self.critic_errors = []
        self.value_errors = []

    def act(self, obs, reward, training=True):
        obs = obs[0]
        action, greedy_action = np.clip(self._act([obs]), -1, 1)
        action = action[0]
        greedy_action = greedy_action[0]

        if not training:
            action = greedy_action

        if training and self.t > 10 * 200:
            # sample experiences
            obs_t,\
            actions,\
            rewards,\
            obs_tp1,\
            dones = self.replay_buffer.sample(self.batch_size)

            # update networks
            value_error = self._train_value(obs_t, actions)
            critic_error = self._train_critic(
                obs_t, actions, rewards, obs_tp1, dones)
            actor_error = self._train_actor(obs_t, actions)

            # store errors through episode
            self.value_errors.append(value_error)
            self.critic_errors.append(critic_error)
            self.actor_errors.append(actor_error)

            # update target networks
            self._update_target()

        if training and self.last_obs is not None:
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=False
            )

        self.t += 1
        self.last_obs = obs
        self.last_action = action
        return action * self.action_scale

    def stop_episode(self, obs, reward, training=True):
        obs = obs[0]
        if training:
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=True
            )
            print('actor error: {}, critic error: {}, value error: {}'.format(
                sum(self.actor_errors), sum(self.critic_errors), sum(self.value_errors)))
        self.last_obs = None
        self.last_action = []
        self.value_errors = []
        self.critic_errors = []
        self.actor_errors = []
