import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens,
                        inpt,
                        num_actions,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        reg_factor=1e-3,
                        scope='actor',
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # l2 regularizer
        regularizer = layers.l2_regularizer(scale=reg_factor)
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(
                out, hidden,
                bias_initializer=tf.constant_initializer(0.0),
                kernel_initializer=initializer,
                kernel_regularizer=regularizer)
            out = tf.nn.relu(out)

        # mean value of normal distribution
        mu = tf.layers.dense(
            out, num_actions, kernel_initializer=initializer, name='mu')

        # variance of normal distribution
        sigma = tf.layers.dense(
            out, num_actions, kernel_initializer=initializer, name='sigma')

        # sample actions from normal distribution
        dist = tf.distributions.Normal(mu, tf.exp(sigma))
        out = tf.reshape(dist.sample(num_actions), [-1, num_actions])
        out = tf.stop_gradient(out)
        action = tf.nn.tanh(out)
        log_prob = dist.log_prob(out) - tf.log(1 - action ** 2 + 1e-6)
    return action, dist, log_prob, regularizer

def _make_critic_network(hiddens,
                         inpt,
                         action,
                         initializer=tf.contrib.layers.xavier_initializer(),
                         scope='critic',
                         reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # concat action
        out = tf.concat([inpt, action], axis=1)
        for hidden in hiddens:
            out = tf.layers.dense(
                out, hidden,
                bias_initializer=tf.constant_initializer(0.0),
                kernel_initializer=initializer)
            out = tf.nn.relu(out)

        out = tf.layers.dense(out, 1, kernel_initializer=initializer)
    return out

def _make_value_network(hiddens,
                        inpt,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        scope='value',
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(
                out, hidden,
                bias_initializer=tf.constant_initializer(0.0),
                kernel_initializer=initializer)
            out = tf.nn.relu(out)

        out = tf.layers.dense(out, 1, kernel_initializer=initializer)
    return out

def make_actor_network(hiddens):
    return lambda *args, **kwargs: _make_actor_network(hiddens, *args, **kwargs)

def make_critic_network(hiddens):
    return lambda *args, **kwargs: _make_critic_network(hiddens, *args, **kwargs)

def make_value_network(hiddens):
    return lambda *args, **kwargs: _make_value_network(hiddens, *args, **kwargs)
