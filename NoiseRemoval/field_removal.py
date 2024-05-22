import numpy as np
from NoiseRemoval.RemoveNoiseTransformed import remove_noise_gmm
from scipy.sparse.csgraph import connected_components
from NoiseRemoval.BulkVelocityClassic import ClassicBV


def get_dense(clusterer, cluster_id):
    data_idx = np.arange(clusterer.X.shape[0])
    cluster_bool_array = remove_noise_gmm(clusterer.labels_==cluster_id, te_obj=clusterer)
    if cluster_bool_array is None:
        rho = clusterer.weights_[clusterer.labels_ == cluster_id]
        mad = np.median(np.abs(rho - np.median(rho)))
        threshold = np.median(rho)*0.995 + 3 * mad * 1.01
        # Statistisch fundierterer cut
        # threshold = np.median(rho) + 3 * mad
        cluster_bool_array = rho > threshold
        idx_cluster = data_idx[clusterer.labels_ == cluster_id][cluster_bool_array]
        if len(idx_cluster) > 30:
            _, cc_idx = connected_components(clusterer.A[idx_cluster, :][:, idx_cluster])
            # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
            cluster_indices = data_idx[idx_cluster][cc_idx == np.argmax(np.bincount(cc_idx))]
            cluster_bool_array = np.isin(data_idx, cluster_indices)
        else:
            cluster_bool_array = np.isin(data_idx, idx_cluster)
    return cluster_bool_array


def remove_field(clusterer, data):
    data_idx = np.arange(clusterer.X.shape[0])
    final_labels = -np.ones(clusterer.X.shape[0], dtype=int)
    # Create ClassicBV object
    c_bv = ClassicBV(data)
    for i, label_i in enumerate(np.unique(clusterer.labels_)):
        bool_arr_l_i = get_dense(clusterer, label_i)
        if np.sum(bool_arr_l_i) > 20:
            mu_i, C_i = c_bv.estimate_normal_params(cluster_subset=bool_arr_l_i)
            # Estimate rv of potential cluster members
            dist_maha = c_bv.mahalanobis_distance(uvw_mean=mu_i, cov=C_i, cluster_subset=clusterer.labels_==label_i)
            subset_idx = data_idx[clusterer.labels_==label_i][dist_maha < 5]
            if len(subset_idx) > 30:
                bool_arr_final = remove_noise_gmm(subset_idx, te_obj=clusterer)
            else:
                bool_arr_final = subset_idx
            final_labels[bool_arr_final] = i
        else:
            final_labels[clusterer.labels_==label_i] = -1
    return final_labels
