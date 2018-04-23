import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens,
                        inpt,
                        obs_dim,
                        num_actions,
                         initializer=tf.contrib.layers.xavier_initializer(),
                        scope='actor',
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(out, hidden,
                bias_initializer=tf.constant_initializer(0.0),
                kernel_initializer=initializer)
            out = tf.nn.relu(out)

        # mean value of normal distribution
        mu = tf.layers.dense(
            out, num_actions, kernel_initializer=initializer, name='mu')
        mu = tf.nn.tanh(mu + 1e-20)

        # variance of normal distribution
        sigma = tf.layers.dense(
            out, num_actions, kernel_initializer=initializer, name='sigma')
        sigma = tf.nn.softplus(sigma + 1e-20)

        # sample actions from normal distribution
        dist = tf.distributions.Normal(mu, sigma ** 2)
        out = tf.squeeze(dist.sample(num_actions), [0])
    return out, dist

def _make_critic_network(hiddens,
                         inpt,
                         action,
                         obs_dim,
                         initializer=tf.contrib.layers.xavier_initializer(),
                         scope='critic',
                         reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens[:-1]:
            out = tf.layers.dense(out, hidden,
                bias_initializer=tf.constant_initializer(0.0),
                kernel_initializer=initializer)
            out = tf.nn.relu(out)

        # concat action
        out = tf.concat([out, action], axis=1)
        out = tf.layers.dense(out, hiddens[-1],
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=initializer)
        out = tf.nn.relu(out)

        out = tf.layers.dense(out, 1, kernel_initializer=initializer)
    return out

def _make_value_network(hiddens,
                        inpt,
                        obs_dim,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        scope='value',
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(out, hidden,
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
