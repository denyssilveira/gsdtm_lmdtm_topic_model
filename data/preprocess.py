"""
GSDTM and LMDTM VAEs Implementation
file: data/preprocess
This implementation performs a preprocessing in the 20newsgroups Dataset.
"""

# Author: Denys Silveira <ddbs@icomp.ufam.edu.br>

from sklearn.datasets import fetch_20newsgroups
import nltk
import numpy as np
import string
import argparse
import os


def clean_data(data, labels):
    """ Clean the raw text, removing punctuations and performing tokenization.

    Parameters
    ----------
    data : array_like, shape (n_docs)
        List of n_docs lines of 20newsgroups dataset.

    Returns
    -------
    cleaned_docs : array_like, shape (?, n_docs)
        List of preprocessed words in each document.
    """

    cleaned_docs = list()
    cleaned_labels = list()

    # Translate each punctuation symbol with empty space
    translator = str.maketrans(string.punctuation,
                               ' ' * len(string.punctuation))

    for doc, label in zip(data, labels):
        # Apply the translator
        doc = doc.translate(translator)
        tokens = nltk.word_tokenize(doc)
        cleaned_doc = [word.lower() for word in tokens]

        if cleaned_doc != []:
            cleaned_docs.append(cleaned_doc)
            cleaned_labels.append(label)

    return cleaned_docs, cleaned_labels


def vocab_from_file(path):
    """ Read all words of vocabulary from a file.

    Parameters
    ----------
    path : string
        Path of vocabulary file.

    Returns
    -------
    index2word : dict_like, shape (vocabsize)
        Dictionary that contains indices as keys, and words as values.

    word2index : dict_like, shape (vocabsize)
        Dictionary that contains words as keys, and indices as values.
    """

    fin = open(path, 'r')

    WORD_INDEX = 0
    index2word = dict()
    word2index = dict()

    while True:

        line = fin.readline()

        if not line:
            break

        word_freq = line.split()
        word = word_freq[WORD_INDEX]
        word2index[word] = len(word2index) + 1
        index2word[len(index2word) + 1] = word

    fin.close()

    return (index2word, word2index)


def generate_data(data_docs, target_docs, word2index, path, npmi_av_path):
    """ Generate files with output of pre-processing.

    Parameters
    ----------
    data_docs : array_like, shape (?, n_docs)
        List of preprocessed words.

    target_docs : array_like, shape (n_docs)
        List of document labels.

    word2index : dict_like, shape (vocabsize)
        Dictionary that contains words as keys, and indices as values .

    path: string
        Output file path.

    npmi_av_path: string
        Path of file destined for NPMI avaliation.

    Returns
    -------
    """

    fout = open(path, 'w')

    if npmi_av_path is not None:
        fout_npmi_av = open(npmi_av_path, 'w')

    for data, target in zip(data_docs, target_docs):
        id_freq = dict()

        for word in data:
            if word in word2index:
                if npmi_av_path is not None:
                    fout_npmi_av.write("{0} ".format(word))

                if word not in id_freq:
                    id_freq[word] = 1
                else:
                    id_freq[word] += 1

        if npmi_av_path is not None:
            fout_npmi_av.write("\n")

        if len(id_freq) != 0:
            fout.write("{0} ".format(str(target + 1)))
            c = 0

            for word, freq in id_freq.items():
                fout.write("{0}:{1}".format(word2index[word], freq))
                c += 1

                if len(id_freq) != c:
                    fout.write(" ")

            fout.write('\n')
    fout.close()


def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    newsgroups_train = fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers',
                                                  'quotes'),
                                          random_state=args.seed,
                                          shuffle=True)

    newsgroups_test = fetch_20newsgroups(subset='test',
                                         remove=('headers', 'footers',
                                                 'quotes'),
                                         random_state=args.seed,
                                         shuffle=True)
    
    newsgroups_train_data = newsgroups_train.data[103:]
    newsgroups_train_labels = newsgroups_train.target[103:]

    newsgroups_valid_data = newsgroups_train.data[:103]
    newsgroups_valid_labels = newsgroups_train.target[:103]

    cleaned_train_data, cleaned_train_labels = clean_data(
                                                newsgroups_train_data,
                                                newsgroups_train_labels)

    cleaned_test_data, cleaned_test_labels = clean_data(newsgroups_test.data,
                                                        newsgroups_test.target)

    cleaned_valid_data, cleaned_valid_labels = clean_data(
                                                newsgroups_valid_data,
                                                newsgroups_valid_labels)

    index2word, word2index = vocab_from_file(args.vocab)

    generate_data(cleaned_train_data, cleaned_train_labels,
                  word2index, os.path.join(args.output, "train.feat"),
                  os.path.join(args.output, "dataset_train.txt"))

    generate_data(cleaned_valid_data, cleaned_valid_labels,
                  word2index, os.path.join(args.output, "valid.feat"),
                  None)

    generate_data(cleaned_test_data, cleaned_test_labels,
                  word2index, os.path.join(args.output, "test.feat"),
                  None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output dataset')

    parser.add_argument('--vocab', type=str, required=True,
                        help='Vocabulary file path')

    parser.add_argument('--seed', type=int, required=True,
                        help='Random Seed')
                        
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
