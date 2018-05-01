"""
GSDTM and LMDTM VAEs Implementation
Trains the models 
"""

import tensorflow as tf
import numpy as np
import sys
import os

import model.gsdtm as gsdtm
import model.lmdtm as lmdtm
import model.data as data
import model.evaluate as evaluate
from model import utils


def train(dataset, vocabsize, flags):
    tf.reset_default_graph()

    if flags.model == 'gsdtm':
        model = gsdtm.GSDTM(vocabsize, flags)
    else:
        model = lmdtm.LMDTM(vocabsize, flags)

    # Merge of Summaries
    merged = tf.summary.merge_all()

    # Summaries Writers
    train_summary_path = os.path.join(flags.summaries_dir, 'train')

    train_writer = tf.summary.FileWriter(
                            train_summary_path,
                            model.sess.graph)

    valid_writer = tf.summary.FileWriter(
                            os.path.join(flags.summaries_dir, 'valid'))

    saver = tf.train.Saver()

    best_perplexity_value = np.inf
    best_prec_value = 0.0
    emb = 0

    minibatches = dataset.create_minibatch(
                        dataset.train_data,
                        dataset.train_labels,
                        flags.batch_size)

    train_batch = dataset.create_minibatch(
                        dataset.train_data,
                        dataset.train_labels,
                        dataset.n_train_data)

    validation_batch = dataset.create_minibatch(dataset.validation_data,
                                                dataset.validation_labels,
                                                dataset.n_validation_data)

    utils.save_params_json(
                flags,
                os.path.join(flags.summaries_dir, "params.json"))

    for epoch in range(flags.n_epoch):
        avg_cost = 0.
        total_batch = int(dataset.n_train_data / flags.batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = minibatches.__next__()
            # Fit training using batch data
            _, cost, emb, summary = model.sess.run(
                            (model.optimizer, model.loss, model.R, merged),
                            feed_dict={model.x: batch_xs,
                                       model.keep_prob: 0.4})

            avg_cost += (cost / dataset.n_train_data) * flags.batch_size

            if np.isnan(avg_cost):
                print(epoch, i,
                      np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('NaN Error. Please, check learning rate \
                and momentum values.')
                sys.exit()

        s_cost = tf.Summary(value=[tf.Summary.Value(tag="loss/total_loss",
                                                    simple_value=avg_cost)])
        train_writer.add_summary(s_cost, epoch + 1)

        # Print Cost Value
        if ((epoch + 1) % 10 == 0) and (epoch != 0):
            print("Epoch:", '%04d' % (epoch + 1),
                  "Cost=", "{:.9f}".format(avg_cost))

            train_writer.add_summary(summary, epoch + 1)
            sys.stdout.flush()

        # Print topics from embedding space and validation
        if ((epoch + 1) % 10 == 0) and (epoch != 0):
            evaluate.print_top_words(
                            emb, list(zip(*sorted(dataset.index2word.items(),
                                                  key=lambda x: x[1])))[0])
            sys.stdout.flush()

            model.sess.run(model.embedding_var.initializer)

            # Get vectorial document representation from each fold
            train_xs, train_ys = train_batch.__next__()
            train_vectors = model.vectors(
                train_xs,
                0.4
            )

            valid_xs, valid_ys = validation_batch.__next__()
            valid_vectors = model.vectors(
                valid_xs,
                1.0
            )

            valid_retrieval_task_prec = evaluate.retrieval_evaluate(
                train_vectors,
                valid_vectors,
                train_ys,
                valid_ys
            )[0]

            s_prec_frac = tf.Summary(
                            value=[tf.Summary.Value(
                                   tag="precision/valid_precision_by_fraction",
                                   simple_value=valid_retrieval_task_prec)])
            valid_writer.add_summary(s_prec_frac, epoch + 1)

            if valid_retrieval_task_prec > best_prec_value:
                best_prec_value = valid_retrieval_task_prec
                print("Best precision@0.02 validation found: {0}".format(
                        valid_retrieval_task_prec))
                sys.stdout.flush()

                saver.save(model.sess,
                           os.path.join(flags.summaries_dir,
                                        'model_by_prec_frac.ckpt'),
                           global_step=epoch + 1)

            # Calculate the perplexity on valid dataset
            valid_perplexity_value = evaluate.get_perplexity(model, valid_xs)
            s_perplexity = tf.Summary(
                            value=[tf.Summary.Value(
                                    tag="perplexity/valid_perplexity",
                                    simple_value=valid_perplexity_value)])

            valid_writer.add_summary(s_perplexity, epoch + 1)

            print('Perplexity value found: {:.3f}'.format(
                valid_perplexity_value))
            sys.stdout.flush()

            if valid_perplexity_value < best_perplexity_value:
                best_perplexity_value = valid_perplexity_value
                print("Best perplexity on validation found: {0}".format(
                    valid_perplexity_value))

                saver.save(model.sess,
                           os.path.join(flags.summaries_dir,
                                        'model_by_perplexity.ckpt'),
                           global_step=epoch + 1)

    return model, emb


def main(flags):
    dataset = data.Dataset(flags.data_dir, flags.seed)
    vocabsize = len(dataset.index2word)

    # Clean log files
    if tf.gfile.Exists(flags.summaries_dir):
        tf.gfile.DeleteRecursively(flags.summaries_dir)
    tf.gfile.MakeDirs(flags.summaries_dir)

    # Make Tensorboard projector metafile
    utils.make_metadata_visualization(dataset, flags)

    model, emb = train(dataset, vocabsize, flags)
    # evaluate.print_top_words(emb,
    #                         list(zip(*sorted(dataset.index2word.items(),
    #                                          key=lambda x: x[1])))[0])


def parse_flags():
    flags = tf.app.flags
    flags.DEFINE_string('data_dir', 'data/20newsgroups', 'Data dir path.')
    flags.DEFINE_string('model', 'gsdtm', 'Define the model.')
    flags.DEFINE_integer('batch_size', 200, 'Batch size.')
    flags.DEFINE_integer('n_hidden', 100, 'Size of each hidden layer.')
    flags.DEFINE_integer('n_topic', 50, 'Size of stochastic vector.')
    flags.DEFINE_integer('n_epoch', 2000, 'Number of epochs performed.')
    flags.DEFINE_string('summaries_dir', 'log', 'Summaries directory.')
    flags.DEFINE_integer('seed', 1000, 'Seed.')
    flags.DEFINE_integer('num_cores', 4, 'Set number of cores.')
    flags.DEFINE_boolean('batch_norm', True, 'Use batch normalization?')
    flags.DEFINE_boolean('dropout', True, 'Use dropout?')
    flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate.')    
    flags.DEFINE_float('temp', 5.0, 'Temperature')     
    flags.DEFINE_integer('n_component', 50, 'Number of components')
    return(flags.FLAGS)


if __name__ == "__main__":
    main(parse_flags())
