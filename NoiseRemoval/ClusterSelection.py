import copy
import numpy as np
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor_distribution(data, k_neighbors=7):
    nn = NearestNeighbors().fit(data)
    # find the distance to the k'th nn
    neigh_dist, _ = nn.kneighbors(X=None, n_neighbors=k_neighbors, return_distance=True)
    return np.max(neigh_dist, axis=1)


def maximize_ch_score(nearest_neighbors_arr):
    """
    Iteratively create clustering by merging i densest and N-i lowest density clusters
    and check if clustering "is good". Goodness is defined by maximizing
    the calinski_harabasz_score (tried others, this works best) --> looks for two prominent well separated peaks.
    We suppose that the density of clusters in a region should approximately follow a singal modal distribution;
    Thus, if we find 2 well separated peaks we remove the low density peak
    """
    # Compute median distances & sort them
    nna_medians = np.array([np.median(nn) for nn in nearest_neighbors_arr])
    nna_argsort = np.argsort(nna_medians)
    # --- loop through clusters by decreasing median density (increasing distance) ---
    scores = []
    index = np.arange(3, nna_argsort.size)  # Start with 3 clusters minimum
    for i in index:
        group_good = np.concatenate([nearest_neighbors_arr[j] for j in nna_argsort[:i]])
        group_bad = np.concatenate([nearest_neighbors_arr[j] for j in nna_argsort[i:]])
        dat = np.concatenate([group_good, group_bad]).reshape(-1, 1)
        label_dat = np.concatenate([np.zeros(group_good.size), np.ones(group_bad.size)])
        scores.append(calinski_harabasz_score(dat, label_dat))
    return index, scores, nna_medians, nna_argsort


def remove_outlier_clusters(clustering_res, nearest_neighbors_arr, save_as_new_cluster=False):
    """Remove clusters based on density histogram comparisons with other clusters"""
    cr = copy.deepcopy(clustering_res)
    # Get maximal CH score
    index, scores, nna_medians, nna_argsort = maximize_ch_score(nearest_neighbors_arr)
    # Remove "bad" clusters
    i = index[np.argmax(scores)]  # best separation at maximum calinski_harabasz_score
    bad_cluster_ids = nna_argsort[i:]
    # Replace or remove "bad" clusters
    max_clusternb = np.max(clustering_res)
    if save_as_new_cluster:
        new_cluster_nb = max_clusternb + 1
    else:
        new_cluster_nb = -1
    cr[np.isin(clustering_res, bad_cluster_ids)] = new_cluster_nb
    return cr, nna_medians[nna_argsort[i]]


def remove_outlier_by_nn_threshold(clustering_res, nearest_neighbors_arr, nn_median_th, save_as_new_cluster=False):
    cr = copy.deepcopy(clustering_res)
    # Compute median distances & sort them
    nna_medians = np.array([np.median(nn) for nn in nearest_neighbors_arr])
    # Check which median is closest to the threshold
    nn = NearestNeighbors().fit(nna_medians.reshape(-1, 1))
    neigh_idx = nn.kneighbors(X=np.array([[nn_median_th]]), n_neighbors=1, return_distance=False)
    # we still want to include the "closest" neighbor even if it's value is smaller than the threshold
    # plus small epsilon to account for inaccuracies
    final_th = np.max([nna_medians[neigh_idx.flatten()[0]], nn_median_th]) + 1e-4
    # Remove clusters with median distances larger than final_th
    bad_cluster_ids = np.argwhere(nna_medians > final_th).flatten()
    # Replace or remove "bad" clusters
    max_clusternb = np.max(clustering_res)
    if save_as_new_cluster:  # Start with 4 clusters minimum
        new_cluster_nb = max_clusternb + 1
    else:
        new_cluster_nb = -1
    cr[np.isin(clustering_res, bad_cluster_ids)] = new_cluster_nb
    return cr

