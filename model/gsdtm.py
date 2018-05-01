"""
GSDTM and LMDTM VAEs Implementation
This code provides an implementation of the GSDTM VAE Model.
"""

import tensorflow as tf
from model import utils
slim = tf.contrib.slim


class GSDTM(object):

    def __init__(self, vocabsize, flags):

        print(flags.learning_rate)
        self.learning_rate = flags.learning_rate
        self.batch_size = flags.batch_size
        self.n_hidden = flags.n_hidden
        self.init_temp = flags.temp

        print('Network Architecture')
        print('Length of layers: {0}'.format(self.n_hidden))
        print('Learning Rate: {0}'.format(self.learning_rate))
        print('Batch size: {0}'.format(self.batch_size))
        print('Initial temperature: {0}'.format(self.init_temp))

        with tf.Session(config=tf.ConfigProto(
                            inter_op_parallelism_threads=flags.num_cores,
                            intra_op_parallelism_threads=flags.num_cores,
                            gpu_options=tf.GPUOptions(
                                            allow_growth=True))) as self.sess:
            # tf Graph input
            self.x = tf.placeholder(tf.float32, [None, vocabsize])
            self.keep_prob = tf.placeholder(tf.float32)

            # temperature
            self.tau = tf.Variable(self.init_temp, name="temperature")

            net = slim.stack(self.x, slim.fully_connected,
                             [self.n_hidden, flags.n_topic])

            if flags.dropout:
                layer_do = tf.nn.dropout(net, self.keep_prob)
            else:
                layer_do = net

            if flags.batch_norm:
                logits_y = tf.contrib.layers.batch_norm(slim.fully_connected(
                                                           layer_do,
                                                           flags.n_topic,
                                                           activation_fn=None))
            else:
                logits_y = slim.fully_connected(layer_do,
                                                flags.n_topic,
                                                activation_fn=None)
            # Distribution over logits
            self.q_y = tf.nn.softmax(logits_y)
            utils.variable_summaries(self.q_y, 'encoder/qy')
            self.log_q_y = tf.log(self.q_y + 1e-20)

            # Samples 
            self.z = self.gumbel_softmax(logits_y, self.tau)
            utils.variable_summaries(self.z, 'encoder/z')

            if flags.dropout:
                layer_do_decoder = tf.nn.dropout(self.z, self.keep_prob)
            else:
                layer_do_decoder = self.z

            self.output = slim.fully_connected(layer_do_decoder,
                                               vocabsize,
                                               activation_fn=None,
                                               scope='embedding')

            if flags.batch_norm:
                logits_x = tf.contrib.layers.batch_norm(self.output)
            else:
                logits_x = self.output

            self.log_p_x = tf.nn.log_softmax(logits_x)
            utils.variable_summaries(self.log_p_x, 'decoder/log_px')

            kl_tmp = self.q_y * (self.log_q_y - tf.log(1.0 / flags.n_topic))
            self.KL = tf.reduce_sum(kl_tmp, 1)

            self.elbo = tf.reduce_sum(tf.multiply(self.log_p_x, self.x), 1) \
                - self.KL

            self.loss = tf.reduce_mean(-self.elbo)

            self.optimizer = \
                tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                    ).minimize(
                        self.loss,
                        var_list=slim.get_model_variables())

            self.R = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES,
                        scope='embedding/weights:0')[0]

            init = tf.global_variables_initializer()

            self.embedding_var = tf.Variable(
                                    tf.transpose(self.R),
                                    name='word_embedding')

            utils.set_embedding_visualization(self.embedding_var, flags)
        self.sess.run(init)

    def sample_gumbel(self, shape, epsilon=1e-20):
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + epsilon) + epsilon)

    def gumbel_softmax_sample(self, logits, temperature):
        z = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(z / temperature)

    def gumbel_softmax(self, logits, temperature):
        z = self.gumbel_softmax_sample(logits, temperature)
        return z

    def vectors(self, batch_xs, keep_prob):
        return self.sess.run(
                            [self.z],
                            feed_dict={self.x: batch_xs,
                                       self.keep_prob: keep_prob})
