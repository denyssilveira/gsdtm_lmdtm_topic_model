"""
GSDTM and LMDTM VAEs Implementation
Provides auxiliary functions for data visualization on tensorboard and parameters.
"""

from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import os
import json


# Tensorflow Embedding Visualization
def make_metadata_visualization(dataset, flags):
    fin = open(os.path.join(flags.summaries_dir, 'metadata.tsv'), 'w')

    fin.write('"Word"\t"Index"\n')

    for index, word in dataset.index2word.items():
        fin.write('{0}\t{1}\n'.format(index, word))

    fin.close()


def set_embedding_visualization(embedding_var, flags):

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()

    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = '../metadata.tsv'
    projector.visualize_embeddings(
                        tf.summary.FileWriter(
                            os.path.join(flags.summaries_dir, "train")),
                        config)


def variable_summaries(var, name):
    # Attach summaries to a tensor
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))

        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def save_params_json(flags, path):
        data = dict()

        data["data_dir"] = flags.data_dir
        data["model"] = flags.model
        data["learning_rate"] = flags.learning_rate
        data["batch_size"] = flags.batch_size
        data["n_hidden"] = flags.n_hidden
        data["n_topic"] = flags.n_topic
        data["n_epoch"] = flags.n_epoch
        data["summaries_dir"] = flags.summaries_dir
        data["seed"] = flags.seed
        data["num_cores"] = flags.num_cores
        data["temp"] = flags.temp
        data["n_component"] = flags.n_component
        data["batch_norm"] = flags.batch_norm
        data["dropout"] = flags.dropout

        json_file = open(path, 'w')
        json.dump(data, json_file)
