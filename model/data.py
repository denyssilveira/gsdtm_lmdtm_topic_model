"""
GSDTM and LMDTM VAEs Implementation
This module defines the class Dataset in order to store and build batches of data
"""

import os
import numpy as np
import tensorflow as tf
import random


class Dataset(object):

    def __init__(self, data_dir, seed=42):
        self.data_dir = data_dir
        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.train_data, self.train_labels = self._load_data(
                                                os.path.join(data_dir,
                                                             "train.feat"))
        self.n_train_data = len(self.train_data)

        self.validation_data, self.validation_labels = self._load_data(
                                                        os.path.join(
                                                            data_dir,
                                                            "valid.feat"))
        self.n_validation_data = len(self.validation_data)

        self.test_data, self.test_labels = self._load_data(
                                                    os.path.join(
                                                            data_dir,
                                                            "test.feat"))
        self.n_test_data = len(self.test_data)

        self.word2index, self.index2word = self._get_dict_from_vocab(
                                                os.path.join(data_dir,
                                                             "vocab.new"))
        self.vocabsize = len(self.word2index)

    def _load_data(self, data_path):
        data = list()
        labels = list()

        try:
            fin = open(data_path, 'r')

            ID_WORD = 0
            FREQ_WORD = 1

            while True:
                line = fin.readline()

                if not line:
                    break

                id_freqs = line.split()
                doc = dict()
                count = 0

                label = id_freqs[0]

                for id_freq in id_freqs[1:]:
                    items = id_freq.split(':')

                    # index = ID_WORD - 1, because python starts list with
                    # index 0
                    id_word = int(items[ID_WORD]) - 1
                    freq_word = int(items[FREQ_WORD])
                    doc[id_word] = freq_word
                    count += freq_word

                if count > 0:
                    data.append(doc)
                    labels.append(label)

            fin.close()
            return data, labels

        except FileNotFoundError as e:
            print("Can not open {0}: file not found.".format(data_path))
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))

    def _get_dict_from_vocab(self, vocab_path):
        WORD = 0
        dictionary = dict()
        rev_dictionary = dict()

        try:
            fin = open(vocab_path)

            while True:
                line = fin.readline()
                if not line:
                    break

                word_freq = line.split()
                word = word_freq[WORD]

                id_x = len(dictionary)
                rev_dictionary[word] = id_x
                dictionary[id_x] = word

            fin.close()
            return dictionary, rev_dictionary

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))

    def _one_hot(self, batch):
        one_hot_batch = np.zeros((len(batch), self.vocabsize),
                                 dtype=np.float32)

        for i, item in enumerate(batch):
            data = np.zeros(self.vocabsize, dtype=np.float32)
            for word_id, freq in item.items():
                data[word_id] = freq
            one_hot_batch[i, :] = data

        return one_hot_batch

    def create_minibatch(self, data, labels, batch_size):
        rng = np.random.RandomState(self.seed)
        data = np.array(data)
        labels = np.array(labels)

        while True:
            # Return random data samples of a size 'minibatch_size' at
            # each iteration
            ixs = rng.randint(len(data), size=batch_size)
            yield self._one_hot(data[ixs]), labels[ixs]
