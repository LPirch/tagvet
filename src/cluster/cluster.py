import os
import argparse
import pickle
import json

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, adjusted_rand_score, v_measure_score, pairwise_distances
from sklearn.metrics import completeness_score, homogeneity_score
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_file', help='feature file (pkl)')
    parser.add_argument('feat_cache', help='cache file for features after preprocessing and extraction')
    parser.add_argument('label_file', help='tsv file (tab-separated) with hashes and labels')
    parser.add_argument('out_results', help='output file with results (JSON)')
    parser.add_argument('out_prediction', help='output file for cluster labels (CSV)')
    parser.add_argument('jobs', type=int, default=1, help='number of parallel processes')
    parser.add_argument('--ngram_len', type=int, default=2, help='length of ngrams')
    parser.add_argument('--min_df', type=int, default=1, help='minimum absolute doc frequency')
    parser.add_argument('--max_df', type=float, default=1.0, help='maximum relative doc frequency')
    parser.add_argument('--binary', action='store_true', help='whether to extract binary vectors')
    return vars(parser.parse_args())


def preprocess(X):
    """
    Optional preprocessing for secondary data format.
    @param X: list of lists of lists (glogs, function calls, function call + attributes)
    """
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = ' '.join([str(x) for x in X[i][j]])
        X[i] = '\n'.join(X[i])
    return X


def eval_all(X, Y_true, jobs):
    """
    Perform a full clustering evaluation on different algorithms and configurations.
    @param X: feature matrix (scipy.sparse.csr_matrix, (n_reports x n_features))
    @param Y_true: reference labels (np.ndarray, (n_reports,), e.g. malware families
    @param jobs: number of parallel processes to use
    """
    results = {}
    cluster_types = {
        'KMeans': (KMeans, {
            'n_clusters': range(2, 501, 20),
            'use_precomputed': [False]
        }),
        'DBSCAN': (DBSCAN, {
            'eps': np.geomspace(0.01, 10.0, 100),
            'min_samples': [3],
            'metric': ['euclidean'],
            'n_jobs': [jobs],
            'use_precomputed': [True]
        }),
        'Agglomerative': (AgglomerativeClustering, {
            'n_clusters': range(2, 501, 20),
            'linkage': ['single', 'complete'],
            'affinity': ['euclidean'],
            'use_precomputed': [True]
        })
    }

    # pre-compute distances
    Xdist_dict = {
        'euclidean': pairwise_distances(X, metric='euclidean', n_jobs=jobs),
    }
    """
    # alternative distance metrics
    'cosine': pairwise_distances(X, metric='cosine', n_jobs=jobs),
    'l1': pairwise_distances(X, metric='l1', n_jobs=jobs)
    """

    best_pred = -np.ones(Y_true.shape)  # worst prediction is outlier only
    best_score = -1
    best_ari = -1
    for name, (cluster_type, params) in cluster_types.items():
        print("Evaluating ", name)
        Y_pred, score, ari, res = eval_clustering(X, Xdist_dict, Y_true, cluster_type, params)
        results[name] = res
        if score > best_score:
            best_score = score
            best_ari = ari
            best_pred = Y_pred

    # assign all outliers to trash class
    max_class = best_pred.max()
    best_pred[np.argwhere(best_pred == -1)[:, 0]] = max_class + 1
    return best_pred, best_score, best_ari, results


def eval_clustering(X, Xdist_dict, Y_true, cluster_type, params, target_metric='silhouette'):
    """
    Evaluate a single clustering configuration (algorithm + hyperparameters).
    @param X: feature matrix (scipy.sparse.csr_matrix, (n_reports x n_features))
    @param Xdist_dict: dict of distance matrices (n_reports x n_reports)
    @param Y_true: reference labels (np.ndarray (n_reports, ))
    @param cluster_type: sklearn clustering estimator
    @param params: hyperparameters for the estimator
    @param target_metric: which metric to optimize (default: silhouette score)
    """
    results = []
    best_pred = None
    best_score = -1
    best_ari = -1
    use_precomputed = params.pop('use_precomputed')[0]
    for p in tqdm(ParameterGrid(params)):
        metric = p.get('metric') or 'euclidean'
        Xdist = None
        if use_precomputed:
            Xdist = Xdist_dict[metric]
            p_dict = {k: v for k, v in p.items() if k != 'metric'}
            if cluster_type == AgglomerativeClustering:
                p_dict['affinity'] = 'precomputed'
            else:
                p_dict['metric'] = 'precomputed'
            Y_pred = cluster_type(**p_dict).fit_predict(Xdist)
        else:
            Y_pred = cluster_type(**p).fit_predict(X)
        results.append((p, eval_metrics(X, Y_true, Y_pred, Xdist=Xdist)))
        if results[-1][-1][target_metric] > best_score:
            best_score = results[-1][-1][target_metric]
            best_ari = results[-1][-1]['ARI']
            best_pred = Y_pred
    return best_pred, best_score, best_ari, results


def eval_metrics(X, Y_true, Y_pred, Xdist=None):
    """
    Compute clustering metrics on a given cluster assignment.
    @param X: feature matrix (scipy.sparse.csr_matrix, (n_reports x n_features))
    @param Y_true: reference labels (np.ndarray (n_reports, ))
    @param Y_pred: assigned cluster IDs (np.ndarray (n_reports, ))
    @param Xdist: optional distance matrix (n_reports x n_reports)
    """
    res = {}
    metrics = {
        'silhouette': (lambda X, Y_pred: silhouette_score(X, Y_pred).item(), False, True),
        'ARI': (adjusted_rand_score, True, False),
        'cluster_f1': (v_measure_score, True, False),
        'homogeneity': (homogeneity_score, True, False),
        'completeness': (completeness_score, True, False),
        'n_clusters': (lambda _, Y: len(np.unique(Y)), False, False),
        'n_outliers': (lambda _, Y: len(np.argwhere(Y < 0)), False, False)
    }

    error_val = -1
    for name, (func, is_supervised, use_precomputed) in metrics.items():
        if set(Y_pred) == set([0, -1]) or len(set(Y_pred)) == 1:
            res[name] = error_val
            continue
        try:
            if is_supervised:
                res[name] = func(Y_true, Y_pred)
            else:
                if use_precomputed and Xdist is not None:
                    res[name] = func(Xdist, Y_pred)
                else:
                    res[name] = func(X, Y_pred)
        except ValueError:
            # metric couldn't be evaluated, e.g. when only one cluster is predicted
            res[name] = error_val

    return res


def read_labels(label_file, get_types=False):
    """
    Read malware hashes and labels from Virustotal label file.
    @param label_file: input file with labels (semicolon separated)
    @get_types: whether to extract families of generic malware types as labels
    """
    hashes, labels = [], []
    with open(label_file, 'r') as f:
        for line in f:
            sha256, md5, fam, typ = line.strip().split(';')
            # hashes.append(sha256)
            hashes.append(md5)
            labels.append(typ if get_types else fam)
    return hashes, labels


def load_features(feat_file):
    """
    Load a feature matrix from pickle.
    @param feat_file: input file
    """
    with open(feat_file, 'rb') as pkl:
        X = pickle.load(pkl)
    return X


def load_labels(label_file):
    """
    Load hashes labels from csv file.
    @param label_file: input file
    """
    hashes = []
    Y = []
    with open(label_file, 'r') as f:
        for line in f.read().splitlines():
            h, lab = line.strip().split(',')
            hashes.append(h)
            Y.append(lab)
    return hashes, np.array(Y)


def dump_results(results, best_pred, best_score, best_ari, out_file):
    """
    Save clustering results in a JSON file.
    @param results: dictionary with clustering results
    @param best_pred: cluster assignment of best result
    @param best_score: best unsupervised score (target metric 'silhouette')
    @param best_ari: best ARI score
    @param out_file: name of the output JSON file
    """
    data_dict = {
        'cluster_labels': [str(y) for y in best_pred],
        'best_silhouette': best_score,
        'best_ari': best_ari,
        'metrics': results
    }
    with open(out_file, 'w') as f:
        json.dump(data_dict, f, indent=4)


def dump_prediction(hashes, Y_pred, out_file):
    """
    Save a clustering in a csv file.
    @param hashes: list of malware hashes
    @param Y_pred: assigned cluster IDs
    @param out_file: output file name
    """
    with open(out_file, 'w') as f:
        for h, lab in zip(hashes, Y_pred):
            f.write('{},{}\n'.format(h, lab))


def analyzer(x):
    """
    Yield tokens from list of lists.
    @param x: list of lists containing tokens
    """
    for row in x:
        for token in row:
            yield token


def main(feat_file, feat_cache, label_file, out_results, out_prediction, jobs, ngram_len, min_df, max_df, binary):
    """
    Extract features if not cached already, perform all clustering experiments and save results and the best cluster
    assignment in respective files.
    @param feat_file: feature file (pkl)
    @param feat_cache: cache file for features after preprocessing and extraction
    @param label_file: tsv file (tab-separated) with hashes and labels
    @param out_results: output file with results (JSON)
    @param out_prediction: output file for cluster labels (CSV)
    @param jobs: number of parallel processes
    @param ngram_len: length of ngrams
    @param min_df: minimum absolute doc frequency
    @param max_df: maximum relative doc frequency
    @param binary: whether to extract binary vectors
    """
    if os.path.exists(feat_cache):
        with open(feat_cache, 'rb') as pkl:
            X = pickle.load(pkl)
    else:
        print('[-] cache file not found, starting extraction.')
        X = load_features(feat_file)
        vec = TfidfVectorizer(ngram_range=(ngram_len, ngram_len), analyzer=analyzer)
        X = vec.fit_transform(X)
        svd = TruncatedSVD(n_components=100)
        X = svd.fit_transform(X)
        print('[-] finished. explained variance of SVD: ', sum(svd.explained_variance_ratio_),
              ', dumping features to ', feat_cache)

        with open(feat_cache, 'wb') as pkl:
            pickle.dump(X, pkl)
    hashes, Y_true = load_labels(label_file)
    best_pred, best_score, best_ari, results = eval_all(X, Y_true, jobs)
    dump_results(results, best_pred, best_score, best_ari, out_results)
    dump_prediction(hashes, best_pred, out_prediction)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
