"""
GSDTM and LMDTM VAEs Implementation
This code provides an implementation of the LMDTM VAE Model.
"""

import tensorflow as tf
import numpy as np
import pickle
import sys
import os
from model import utils

slim = tf.contrib.slim

# Distribution over qy
def qy_graph(x, n_hidden, n_topic=50):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qy')) > 0

    with tf.variable_scope('qy'):
        net = slim.stack(x, slim.fully_connected, [n_hidden, n_hidden], reuse=reuse)
        qy_logit = slim.fully_connected(net, n_topic, activation_fn=None)
        qy = tf.nn.softmax(qy_logit, name='prob')

    return qy_logit, qy

# Distribution over qz
def qz_graph(x, y, n_hidden, flags, batch_size=64, n_topic=50, keep_prob=0.4):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qz')) > 0

    with tf.variable_scope('qz'):
        xy = tf.concat((x, y), 1, name='xy/concat')
        net = slim.stack(xy, slim.fully_connected, [n_hidden, n_hidden], scope='net', reuse=reuse)

        if flags.dropout:
            layer_do = tf.nn.dropout(net, keep_prob)
        else:
            layer_do = net

        net_mean = slim.fully_connected(layer_do, n_topic, activation_fn=None, scope='mean', reuse=reuse)
        net_log_sigma = slim.fully_connected(layer_do, n_topic, activation_fn=None, scope='log_sigma', reuse=reuse)

        if flags.batch_norm:
            mean = tf.contrib.layers.batch_norm(net_mean, scope='bn1', reuse=reuse)
            log_sigma = tf.contrib.layers.batch_norm(net_log_sigma, scope='bn2', reuse=reuse)
        else:
            mean = net_mean
            log_sigma = net_log_sigma

        sigma = tf.exp(log_sigma)

        eps = tf.random_normal((tf.shape(x)[0], flags.n_topic), 0, 1, dtype=tf.float32)

        z = tf.add(mean, tf.multiply(sigma, eps))
        z_softmax = tf.nn.softmax(z)

    return z_softmax, mean, sigma, log_sigma

# Distribution over px
def px_graph(z, y, batch_size, n_topic, vocabsize, hidden_size, flags, keep_prob=0.4):
    reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px')) > 0

    with tf.variable_scope('pz'):

        net_prior_mean = slim.fully_connected(y, n_topic, activation_fn=tf.nn.relu, scope='prior_mean', reuse=reuse)
        net_prior_log_sigma = slim.fully_connected(y, n_topic, activation_fn=tf.nn.relu, scope='prior_log_sq_sigma', reuse=reuse)

        if flags.batch_norm:
            mean_prior = tf.contrib.layers.batch_norm(net_prior_mean, scope='bn6',reuse=reuse)
            log_sigma_prior = tf.contrib.layers.batch_norm(net_prior_log_sigma, scope='bn7', reuse=reuse)
        else:
            mean_prior = net_prior_mean
            log_sigma_prior = net_prior_log_sigma

    with tf.variable_scope('px'):

        if flags.dropout:
            layer_do_decoder = tf.nn.dropout(z, keep_prob)
        else:
            layer_do_decoder = z

        output = slim.fully_connected(layer_do_decoder, vocabsize, activation_fn=None, scope='embedding', reuse=reuse)

        if flags.batch_norm:
            logits_x = tf.contrib.layers.batch_norm(output, scope='bn5', reuse=reuse)
        else:
            logits_x = output

    return mean_prior, log_sigma_prior, logits_x


class LMDTM(object):

    def __init__(self, vocabsize, flags):
        self.learning_rate = flags.learning_rate
        self.batch_size = flags.batch_size
        self.n_hidden = flags.n_hidden
        self.n_component = flags.n_component

        print('Network Architecture')
        print('Length of layers: {0}'.format(self.n_hidden))
        print('Learning Rate: {0}'.format(self.learning_rate))
        print('Batch size: {0}'.format(self.batch_size))

        with  tf.Session(config=tf.ConfigProto(
                    inter_op_parallelism_threads = flags.num_cores,
                    intra_op_parallelism_threads = flags.num_cores,
                    gpu_options=tf.GPUOptions(allow_growth=True))) as self.sess:

            # TF Graph input
            self.x = tf.placeholder(tf.float32, [None, vocabsize])
            self.keep_prob = tf.placeholder(tf.float32)

            with tf.name_scope('y_'):
                y_ = tf.fill(tf.stack([tf.shape(self.x)[0],
                                       flags.n_component]), 0.0)

            # Distribution over y
            qy_logit, qy = qy_graph(
                            self.x,
                            self.n_hidden,
                            n_topic=flags.n_topic)

            self.z, self.mean, self.sigma, self.log_sigma, \
            self.prior_mean, self.log_prior_sigma,  self.logits_x, \
            self.log_p_x = [[None] * flags.n_component for i in range(8)]

            # Create graph for each component
            for i in range(flags.n_component):
                with tf.name_scope('graphs/hot_at{:d}'.format(i)):

                    y = tf.add(y_, tf.constant(
                                    np.eye(flags.n_component)[i],
                                    name='hot_at_{:d}'.format(i),
                                    dtype=tf.float32))


                    self.z[i], self.mean[i], self.sigma[i], \
                    self.log_sigma[i] = qz_graph(self.x, y, \
                                                 self.n_hidden, flags,
                                                 batch_size=self.batch_size,
                                                 n_topic=flags.n_topic,
                                                 keep_prob=self.keep_prob)

                    self.prior_mean[i], self.log_prior_sigma[i], \
                    self.logits_x[i] = px_graph(self.z[i], y,
                                                flags.batch_size,
                                                flags.n_topic,
                                                vocabsize,
                                                self.n_hidden,
                                                flags)

                    self.log_p_x[i] = tf.nn.log_softmax(self.logits_x[i])
                    utils.variable_summaries(self.log_p_x[i], 'decoder/log_px{:d}'.format(i))

            # Multinomial KL
            self.KLD_discrete = tf.add_n([qy[:, i] * tf.log(flags.n_component * qy[:, i] + 1e-12) 
                    for i in range(flags.n_component)]) / flags.n_component

            # Gaussian KL
            self.KLD_gaussian = tf.add_n(
                    [
                        0.5 * tf.reduce_sum((2 * (self.log_prior_sigma[i] - self.log_sigma[i])) +\
                        tf.div(tf.square(self.sigma[i]) + tf.square(self.mean[i] - self.prior_mean[i]), \
                        # exp(2 * log(sigma)) == sigma^2
                        tf.exp(2 * self.log_prior_sigma[i])) - 1, 1)      

                        for i in range(flags.n_component)
                    ]) / flags.n_component

            self.reconstruction_loss = tf.add_n([tf.reduce_sum(tf.multiply(self.log_p_x[i], self.x), 1) 
                for i in range(flags.n_component)]) / flags.n_component

            self.elbo = self.reconstruction_loss - self.KLD_gaussian - self.KLD_discrete
            self.loss = tf.reduce_mean(-self.elbo)
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=slim.get_model_variables())

            # Extract word embedding space
            self.R = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px/embedding/weights:0')[0]
            init = tf.global_variables_initializer()

            self.embedding_var = tf.Variable(tf.transpose(self.R), name='word_embedding')
            utils.set_embedding_visualization(self.embedding_var, flags)

        self.sess.run(init)


    def vectors(self, batch_xs, keep_prob):
        return self.sess.run(self.z,
                             feed_dict={self.x: batch_xs,
                                        self.keep_prob: keep_prob}
                            )
