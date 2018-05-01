"""
GSDTM and LMDTM VAEs Implementation
This module computes the precison by fraction of documents, perplexity and rank of topics.

Code adapted from https://github.com/AYLIEN/docnade
"""

import numpy as np
import sklearn.metrics.pairwise as pw

def closest_docs_by_index(corpus_vectors, query_vectors, n_docs):

    docs = list()
    sim = pw.cosine_similarity(corpus_vectors, query_vectors)
    order = np.argsort(sim, axis=0)[::-1]

    for i in range(len(query_vectors)):
        docs.append(order[:, i][0:n_docs])

    return np.array(docs)


def precision(label, predictions):
    if len(predictions):
        return float(len([x for x in predictions if label in x])) \
            / len(predictions)
    else:
        return 0.0


def retrieval_evaluate(
    corpus_vectors,
    query_vectors,
    corpus_labels,
    query_labels,
    recall=[0.0002]
):

    corpus_size = len(corpus_labels)
    query_size = len(query_labels)

    results = list()
    results_overall = list()

    for c in range(len(corpus_vectors)):

        results = list()

        for r in recall:
            n_docs = int((corpus_size * r) + 0.5)

            if not n_docs:
                results.append(0.0)
                continue

            closest = closest_docs_by_index(corpus_vectors[c],
                                            query_vectors[c],
                                            n_docs)

            avg = 0.0

            for i in range(query_size):
                doc_labels = query_labels[i]
                doc_avg = 0.0
                for label in doc_labels:
                    doc_avg += precision(label, corpus_labels[closest[i]])
                doc_avg /= len(doc_labels)
                avg += doc_avg

            avg /= query_size
            results.append(avg)

        results_overall.append(results)
        results = list()

        prec_by_frac = np.mean(np.array(results_overall), axis=0)

    return prec_by_frac


def print_top_words(vectors, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(vectors)):
        print(" ".join([feature_names[j]
            for j in vectors[i].argsort()[:-n_top_words - 1:-1]]))
    print('---------------End of Topics------------------')

def get_cost(model, data):
    cost = model.sess.run((model.loss), feed_dict={model.x: np.expand_dims(data, axis=0),
                                                   model.keep_prob: 1.0})
    return cost


def get_perplexity(model, data):
    costs = list()

    for doc in data:
        doc = doc.astype(np.float32)
        n_words_in_doc = np.sum(doc)
        cost = get_cost(model, doc)
        costs.append(cost / n_words_in_doc)

    perplexity = np.exp(np.mean(np.array(costs)))

    return perplexity