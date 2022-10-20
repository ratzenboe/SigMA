import numpy as np
import copy
from scipy.stats import mode
from itertools import product
from sklearn.metrics.cluster import contingency_matrix


def compute_jaccard_matrix(labels_1, labels_2, minor=False):
    """
    Compute (minor) jaccard matrix where unique clusters of labels_1 are represented as rows
    and clusters of labels_2 as columns.
    The stored data is the (minor) jaccard index between a combination of these
    """
    cm = contingency_matrix(labels_1, labels_2)
    n, m = cm.shape
    jacc_matrix = np.zeros_like(cm, dtype=np.float16)
    for row, col in product(range(n), range(m)):
        intersect_size = cm[row, col]
        if intersect_size > 1e-5:
            if minor:
                jacc_matrix[row, col] = intersect_size / np.min(np.sum(cm[row, :]), np.sum(cm[:, col]))
            else:
                jacc_matrix[row, col] = intersect_size / (np.sum(cm[row, :]) + np.sum(cm[:, col]) - intersect_size)
    return jacc_matrix


def match_labels_many2one(clusters_labels, alt_clusters_labels):
    """Matching
    :return dict:
        key is cluster label in 'alt' solution
        value is cluster label in 'min' solution with highest jaccard score
    """
    jacc_matrix = compute_jaccard_matrix(alt_clusters_labels, clusters_labels)
    # lists of cluster labels in each cluster solution
    alt_labels = np.unique(alt_clusters_labels)
    labels = np.unique(clusters_labels)
    # Dict: key is cluster label in 'alt' solution; value is cluster label in 'min' solution with highest jaccard index
    label_key = {val: labels[np.argmax(jacc_matrix[i, :])] for i, val in enumerate(alt_labels)}
    return label_key


def match_labels_one2one(cluster_labels, alt_cluster_labels):
    """Matching
    :return dict:
        key is cluster label in 'alt' solution
        value is cluster label in cluster solution with highest jaccard score
    """
    jacc_matrix = compute_jaccard_matrix(alt_cluster_labels, cluster_labels)
    # lists of cluster labels in each cluster solution
    alt_labels = np.unique(alt_cluster_labels)
    labels = np.unique(cluster_labels)
    # Loop through maximal entries, assign matches, and remove match from arrays
    label_key = {}
    while (len(alt_labels) > 0) & (len(labels) > 0):
        # Maximum entry in
        i, j = np.unravel_index(np.argmax(jacc_matrix, axis=None), jacc_matrix.shape)
        # Save match
        label_key[alt_labels[i]] = labels[j]
        # remove entries
        alt_labels = np.delete(alt_labels, i)
        labels = np.delete(labels, j)
        jacc_matrix = np.delete(jacc_matrix, i, 0)
        jacc_matrix = np.delete(jacc_matrix, j, 1)
    # If there are still un-matched clusters left give them a new value
    max_val = max(label_key.values())
    if len(alt_labels) > 0:
        for i, alt_i in enumerate(alt_labels):
            # New target value is a value not already stored in the data
            label_key[alt_i] = max_val + 1 + i
    return label_key


def majority_vote(labels, base_labels=None, return_counts=False):
    """Refactored code from: https://github.com/MattHodgman/celluster"""
    labels_copy = copy.deepcopy(labels)
    if base_labels is None:
        # ---- Use kxn np array as input for function; k...number of cluster solutions, n...number of points ----
        # ---- Baseline: Clustering solution with the minimum number of clusters ----
        nb_clusters = [np.unique(l).size-1 for l in labels_copy]
        argmin_nbclusters = np.argmin(nb_clusters)
        # ---- Clusters ids without baseline solution ----
        cluster_solutions_by_idx = [i for i in range(labels_copy.shape[0]) if i != argmin_nbclusters]
        # ---- Loop through the clustering solutions ----
    else:
        cluster_solutions_by_idx = range(labels.shape[0])

    for m in cluster_solutions_by_idx:
        # ---- Get mapping from m'th labels to base_labels ----
        label_key = match_labels_many2one(base_labels, labels_copy[m])
        # ---- Relabel m'th cluster solutions
        labels_copy[m] = np.vectorize(label_key.get)(labels_copy[m])
    # vote
    labels_final, mode_count = mode(labels_copy, axis=0)
    if return_counts:
        return labels_final[0], mode_count[0]
    labels_final[mode_count < labels_copy.shape[0]/2] = -1
    return labels_final.flatten()
