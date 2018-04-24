import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


def build_graph(actor,
                critic,
                value,
                obs_dim,
                num_actions,
                batch_size,
                gamma,
                tau,
                actor_lr,
                critic_lr,
                value_lr,
                reg_factor,
                scope='ddpg',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # input placeholders
        obs_t_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_t')
        act_t_ph = tf.placeholder(tf.float32, [None, num_actions], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        # actor network
        policy_t, dist_t, regularizer = actor(
            obs_t_input, num_actions, reg_factor=reg_factor, scope='actor')
        actor_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/actor'.format(scope))

        # critic network
        q_t = critic(obs_t_input, act_t_ph, scope='critic')
        q_t_with_actor = critic(
            obs_t_input, policy_t, scope='critic', reuse=True)
        critic_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/critic'.format(scope))

        # value network
        v_t = value(obs_t_input, scope='value')
        value_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/value'.format(scope))

        # target value network
        v_tp1 = value(obs_tp1_input, scope='target_value')
        target_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/target_value'.format(scope))

        with tf.variable_scope('value_loss'):
            target = q_t - dist_t.log_prob(act_t_ph)
            value_loss = tf.reduce_mean(
                0.5 * tf.square(v_t - tf.stop_gradient(target)))

        with tf.variable_scope('critic_loss'):
            target = rew_t_ph + gamma * v_tp1 * (1.0 - done_mask_ph)
            critic_loss = tf.reduce_mean(
                0.5 * tf.square(q_t - tf.stop_gradient(target)))

        with tf.variable_scope('policy_loss'):
            target = q_t_with_actor - v_t
            log_pi = dist_t.log_prob(policy_t)
            actor_loss = 0.5 * tf.reduce_mean(
                log_pi * tf.stop_gradient(log_pi - target))
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = layers.apply_regularization(regularizer, reg_variables)
            actor_loss = actor_loss + l2_loss

        # optimize operations
        critic_optimizer = tf.train.AdamOptimizer(critic_lr)
        critic_optimize_expr = critic_optimizer.minimize(
            critic_loss, var_list=critic_func_vars)
        actor_optimizer = tf.train.AdamOptimizer(actor_lr)
        actor_optimize_expr = actor_optimizer.minimize(
            actor_loss, var_list=actor_func_vars)
        value_optimizer = tf.train.AdamOptimizer(value_lr)
        value_optimize_expr = value_optimizer.minimize(
            value_loss, var_list=value_func_vars)

        # update critic target operations
        with tf.variable_scope('update_value_target'):
            update_target_expr = []
            sorted_vars = sorted(value_func_vars, key=lambda v: v.name)
            sorted_target_vars = sorted(target_func_vars, key=lambda v: v.name)
            # assign value variables to target value variables
            for var, var_target in zip(sorted_vars, sorted_target_vars):
                new_var = tau * var + (1 - tau) * var_target
                update_target_expr.append(var_target.assign(new_var))
            update_target_expr = tf.group(*update_target_expr)

        def act(obs):
            feed_dict = {
                obs_t_input: obs
            }
            return tf.get_default_session().run(policy_t, feed_dict=feed_dict)

        def train_actor(obs, action):
            feed_dict = {
                obs_t_input: obs,
                act_t_ph: action
            }
            loss_val, _ = tf.get_default_session().run(
                [actor_loss, actor_optimize_expr], feed_dict=feed_dict)
            return loss_val

        def train_critic(obs_t, action, rew, obs_tp1, done):
            feed_dict = {
                obs_t_input: obs_t,
                act_t_ph: action,
                rew_t_ph: rew,
                obs_tp1_input: obs_tp1,
                done_mask_ph: done
            }
            loss_val, _ = tf.get_default_session().run(
                [critic_loss, critic_optimize_expr], feed_dict=feed_dict)
            return loss_val

        def train_value(obs_t, action):
            feed_dict = {
                obs_t_input: obs_t,
                act_t_ph: action
            }
            loss_val, _ = tf.get_default_session().run(
                [value_loss, value_optimize_expr], feed_dict=feed_dict)
            return loss_val

        def update_target():
            tf.get_default_session().run(update_target_expr)

        return act, train_actor, train_critic, train_value, update_target
