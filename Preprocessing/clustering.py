from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from scipy import sparse
import networkx as nx
import numpy as np


def cluster_utterances(utterances_processed, args):
    n_gram_range = tuple(int(x) for x in args.n_gram.split(","))
    corpus = [elt[1] for elt in utterances_processed]
    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=n_gram_range, norm='l2')
    feature_matrix = vectorizer.fit_transform(corpus)

    svd = TruncatedSVD(args.lsa_num)
    normalizer = Normalizer(norm='l2', copy=False)
    lsa = make_pipeline(svd, normalizer)
    feature_matrix = lsa.fit_transform(feature_matrix)
    feature_matrix = sparse.csr_matrix(feature_matrix)

    if args.algorithm == 'ec':
        c, membership, f = equal_cluster(feature_matrix, [args.sent_num] * (feature_matrix.shape[0]//args.sent_num))
    if args.algorithm == 'kmeans':
        membership = KMeans(n_clusters=len(utterances_processed)//args.sent_num, init='k-means++', n_init=50, random_state=1)\
                        .fit_predict(feature_matrix)

    return membership


def equal_cluster(data, demand, maxiter=None, fixedprec=1e9):
    data = data.toarray()
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    C = min_ + np.random.random((len(demand), data.shape[1])) * (max_ - min_)
    M = np.array([-1] * len(data), dtype=np.int)

    itercnt = 0
    while True:
        itercnt += 1

        # memberships
        g = nx.DiGraph()
        g.add_nodes_from(range(0, data.shape[0]), demand=-1)  # points
        for i in range(0, len(C)):
            g.add_node(len(data) + i, demand=demand[i])

        # Calculating cost...
        cost = np.array([np.linalg.norm(
            np.tile(data.T, len(C)).T - np.tile(C, len(data)).reshape(len(C) * len(data), C.shape[1]), axis=1)])
        # Preparing data_to_C_edges...
        data_to_C_edges = np.concatenate((np.tile([range(0, data.shape[0])], len(C)).T,
                                          np.tile(np.array([range(data.shape[0], data.shape[0] + C.shape[0])]).T,
                                                  len(data)).reshape(len(C) * len(data), 1), cost.T * fixedprec),
                                         axis=1).astype(np.uint64)
        # Adding to graph
        g.add_weighted_edges_from(data_to_C_edges)

        a = len(data) + len(C)
        g.add_node(a, demand=len(data) - np.sum(demand))
        C_to_a_edges = np.concatenate((np.array([range(len(data), len(data) + len(C))]).T, np.tile([[a]], len(C)).T),
                                      axis=1)
        g.add_edges_from(C_to_a_edges)

        # Calculating min cost flow...
        f = nx.min_cost_flow(g)

        # assign
        M_new = np.ones(len(data), dtype=np.int) * -1
        for i in range(len(data)):
            p = sorted(f[i].items(), key=lambda x: x[1])[-1][0]
            M_new[i] = p - len(data)

        # stop condition
        if np.all(M_new == M):
            # Stop
            return C, M, f

        M = M_new

        # compute new centers
        for i in range(len(C)):
            C[i, :] = np.mean(data[M == i, :], axis=0)

        if maxiter is not None and itercnt >= maxiter:
            # Max iterations reached
            return C, M, f