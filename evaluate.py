"""
GSDTM and LMDTM VAEs Implementation
Uses model/evaluate.py to evaluate the trained model.
"""

import tensorflow as tf
import numpy as np
import json
import os
from collections import namedtuple
import pickle
import argparse

import model.gsdtm as gsdtm
import model.lmdtm as lmdtm
import model.data as data
import model.evaluate as ev


def get_recall_precision_perplexity(model, dataset, flags, vocabsize):

    recall_points = None
    precision_by_recall = None
    recall_points = [0.0002, 0.001, 0.004, 0.016, 0.064, 0.256, 1.0]

    train_batch = dataset.create_minibatch(dataset.train_data,
                                           dataset.train_labels,
                                           dataset.n_train_data)

    validation_batch = dataset.create_minibatch(dataset.validation_data,
                                                dataset.validation_labels,
                                                dataset.n_validation_data)

    test_batch = dataset.create_minibatch(dataset.test_data,
                                          dataset.test_labels,
                                          dataset.n_test_data)

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

    test_xs, test_ys = test_batch.__next__()
    test_vectors = model.vectors(
        test_xs,
        1.0
    )

    test_perplexity = ev.get_perplexity(model, test_xs)
    print('Perplexity: {:.6f}'.format(test_perplexity))
    
    if flags.model == 'gsdtm':
        for c in range(len(train_vectors)):
            train_vectors[c] = np.concatenate((train_vectors[c], valid_vectors[c]))
        train_ys = np.concatenate((train_ys, valid_ys))

        print('Get precision-recall data. Please, wait...')

        precision_by_recall = ev.retrieval_evaluate(
            train_vectors,
            test_vectors,
            train_ys,
            test_ys,
            recall_points
        )

        print("Recall x Precision: ")

        for r, p in zip(recall_points, precision_by_recall):
            print(r, p)

    return (recall_points, precision_by_recall, test_perplexity)


def save_results(path, **kargs):
    pickle.dump(kargs, open(path, 'wb'))


def evaluate(model, dataset, flags,  vocabsize):
    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.get_checkpoint_state(flags.summaries_dir)
    saver.restore(model.sess, checkpoint.model_checkpoint_path)

    x_recall, y_precision, perplexity = get_recall_precision_perplexity(
            model,
            dataset,
            flags,
            vocabsize)


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries_dir', type=str, required=True,
                        help='Path to summaries directory')

    args = parser.parse_args()

    fparams = open(os.path.join(args.summaries_dir, 'params.json'), 'r')
    flags_dict = json.load(fparams)

    flags = namedtuple('flags', flags_dict.keys())(*flags_dict.values())

    return flags


def main(flags):
    dataset = data.Dataset(flags.data_dir, flags.seed)
    vocabsize = len(dataset.index2word)
    if flags.model == 'gsdtm':
        model = gsdtm.GSDTM(vocabsize, flags)
    else:
        model = lmdtm.LMDTM(vocabsize, flags)

    evaluate(model, dataset, flags, vocabsize)


if __name__ == "__main__":
    main(parse_flags())
