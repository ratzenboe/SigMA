from NoiseRemoval.RemoveNoiseTransformed import remove_noise_quick_n_dirty, remove_noise_simple
from NoiseRemoval.ClusterSelection import nearest_neighbor_distribution, remove_outlier_clusters
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import LabelEncoder
from ConcensusClustering.consensus import ClusterConsensus
from PlotlyResults import plot
import pandas as pd
import numpy as np


def extract_signal_remove_spurious(df_fit: pd.DataFrame, labels: np.array, density: np.array, X_matrix: np.array,
                                   output_matrix: np.array, array_index: int):
    """
    Older function that I built by merging the old extract_signal and remove_spurious_clusters functions from the
    jupyter-notebook I started out with.

    :param df_fit: Input dataframe that was provided to the SigMA clusterer instance
    :param labels: Label array
    :param density: clusterer.weights attribute
    :param X_matrix: clusterer.X attribute
    :param output_matrix: matrix or 1,N array to write the new labels to. Matrix style was coded for loops over
    multiple sf or KNN
    :param array_index: row index of the output matrix specifying the actual position in the loop
    :return: The numbers of found clusters after a) removing field stars and b) removing spurious clusters. Labels are
    written to the output matrix.
    """

    # very similar to extract_signal, except for the use of a different noise removal function
    labels_with_noise = -np.ones(df_fit.shape[0], dtype=int)
    data_idx = np.arange(df_fit.shape[0])
    for ii, u_cl in enumerate(np.unique(labels[labels > -1])):
        cluster_bool_array, is_good_cluster = remove_noise_quick_n_dirty(density, labels == u_cl)
        if is_good_cluster:
            idx_cluster = data_idx[labels == u_cl][cluster_bool_array]
            labels_with_noise[idx_cluster] = ii

    nb_extracted_signal = np.unique(labels_with_noise[labels_with_noise > -1]).size

    # Compute nearest neighbors distances
    nn_data = nearest_neighbor_distribution(X_matrix)
    # Transform labels to start from 0 - (N-1)
    labels_traf = LabelEncoder().fit_transform(labels_with_noise) - 1
    # Compute NN distances of cluster members
    nn_arr = []
    for u_cl in np.unique(labels_traf[labels_traf > -1]):
        nn_arr.append(nn_data[labels_traf == u_cl].copy())

    cs, _ = remove_outlier_clusters(labels_traf, nn_arr, save_as_new_cluster=False)
    labels_clean = LabelEncoder().fit_transform(cs) - 1
    nb_remove_spurious_clusters = np.unique(labels_clean[labels_clean > -1]).size

    output_matrix[array_index, :] = labels_clean

    return nb_extracted_signal, nb_remove_spurious_clusters


def signal_spurious_simple(df_fit: pd.DataFrame, labels: np.array, te_obj: np.array, X_matrix: np.array,
                           output_matrix: np.array, array_index: int):
    """
    Older function that I built by merging the old extract_signal and remove_spurious_clusters functions from the
    jupyter-notebook I started out with.

    :param df_fit: Input dataframe that was provided to the SigMA clusterer instance
    :param labels: Label array
    :param density: clusterer.weights attribute
    :param X_matrix: clusterer.X attribute
    :param output_matrix: matrix or 1,N array to write the new labels to. Matrix style was coded for loops over
    multiple sf or KNN
    :param array_index: row index of the output matrix specifying the actual position in the loop
    :return: The numbers of found clusters after a) removing field stars and b) removing spurious clusters. Labels are
    written to the output matrix.
    """

    # very similar to extract_signal, except for the use of a different noise removal function
    labels_with_noise = -np.ones(df_fit.shape[0], dtype=int)
    data_idx = np.arange(df_fit.shape[0])
    for ii, u_cl in enumerate(np.unique(labels[labels > -1])):
        cluster_bool_array = remove_noise_simple(labels == u_cl, te_obj=te_obj)
    # if is_good_cluster:
    #     idx_cluster = data_idx[labels == u_cl][cluster_bool_array]
    #     labels_with_noise[idx_cluster] = ii

    nb_extracted_signal = np.unique(labels_with_noise[labels_with_noise > -1]).size

    # Compute nearest neighbors distances
    nn_data = nearest_neighbor_distribution(X_matrix)
    # Transform labels to start from 0 - (N-1)
    labels_traf = LabelEncoder().fit_transform(labels_with_noise) - 1
    # Compute NN distances of cluster members
    nn_arr = []
    for u_cl in np.unique(labels_traf[labels_traf > -1]):
        nn_arr.append(nn_data[labels_traf == u_cl].copy())

    cs, _ = remove_outlier_clusters(labels_traf, nn_arr, save_as_new_cluster=False)
    labels_clean = LabelEncoder().fit_transform(cs) - 1
    nb_remove_spurious_clusters = np.unique(labels_clean[labels_clean > -1]).size

    output_matrix[array_index, :] = labels_clean

    return nb_extracted_signal, nb_remove_spurious_clusters


def remove_field_stars(labels: np.array, density: np.array, output_matrix: np.array, array_index: int):
    """
    Function removing field stars from the clustering solution. Very lenient.

    :param labels: Label array
    :param density: clusterer.weights attribute
    :param output_matrix: matrix or 1,N array to write the new labels to. Matrix style was coded for loops over
    multiple sf or KNN
    :param array_index: row index of the output matrix specifying the actual position in the loop
    :return: The numbers of found clusters after a) removing field stars
    """

    lab = -np.ones_like(labels, dtype=np.int32)
    indices = np.arange(lab.size)
    for unique_cluster in np.unique(labels):
        indices_cluster = indices[labels == unique_cluster]
        lab[indices_cluster[remove_noise_quick_n_dirty(density, labels == unique_cluster)[0]]] = unique_cluster
    # Encode labels to range from 0 -- (N-1), for N clusters with field stars having "-1"
    labels_rfs = LabelEncoder().fit_transform(lab) - 1
    nb_remove_field_stars = np.unique(labels_rfs[labels_rfs > -1]).size

    # write row in the joint label array
    output_matrix[array_index, :] = labels_rfs

    return nb_remove_field_stars


def extract_signal(labels: np.ndarray, clusterer: object, output_matrix: np.array, array_index: int):
    """
    New function for extracting the signal (= clusters) from the clustering solution that Sebastian sent me
    in mid-October. Unlike the previous functions, it relies on the NoiseRemoval function remove_noise_simple.

    :param array_index: current row index of the output matrix to which labels are written
    :param output_matrix: Matrix holding the labels
    :param labels: array holding the labels determined by clusterer.fit()
    :param clusterer: SigMA instance applied to the dataset at hand
    :return: label array where field stars are denoted by -1, and the other groups by integers. ADDENDUM - I use the
    Label-Encoder to create cluster labels between 0 and N-1 for N found clusters
    """

    # initialize label array as all -1
    labels_with_noise = -np.ones(clusterer.X.shape[0], dtype=int)
    data_idx = np.arange(clusterer.X.shape[0])

    # iterate through all labels of cluster stars and remove noise with the custom function
    for i, u_cl in enumerate(np.unique(labels[labels > -1])):
        cluster_bool_array = remove_noise_simple(labels == u_cl, te_obj=clusterer)

        # make a distinction in case there is no noise to be removed (?)
        if cluster_bool_array is not None:
            labels_with_noise[cluster_bool_array] = i
        else:
            rho = clusterer.weights_[labels == u_cl]
            mad = np.median(np.abs(rho - np.median(rho)))
            threshold = np.median(rho) * 0.99 + 3 * mad * 1.2
            # Statistisch fundierterer cut
            # threshold = np.median(rho) + 3 * mad
            cluster_bool_array = rho > threshold
            idx_cluster = data_idx[labels == u_cl][cluster_bool_array]

            # I think this is invoked if more than 30 clusters are found
            if len(idx_cluster) > 30:
                # labels_with_noise[idx_cluster] = i
                # Only graph connected points allowed
                _, cc_idx = connected_components(clusterer.A[idx_cluster, :][:, idx_cluster])
                # Combine CCs data points with originally defined dense core
                # (to not miss out on potentially dropped points)
                cluster_indices = data_idx[idx_cluster][cc_idx == np.argmax(np.bincount(cc_idx))]
                labels_with_noise[np.isin(data_idx, cluster_indices)] = i

    labels_simple = LabelEncoder().fit_transform(labels_with_noise) - 1
    nb_simple = np.unique(labels_simple[labels_simple > -1]).size

    # write row in the joint label array
    output_matrix[array_index, :] = labels_simple

    return nb_simple


def consensus_function(label_matrix: np.array, density_sum: np.array, df_fit: pd.DataFrame, file: str = None,
                       path: str = None, plotting: bool = True):
    """
    Function that takes the different labels created in a loop over either KNNs or scaling factors and makes a consensus
    solution.

    :param label_matrix: matrix holding the different results of the loop
    :param density_sum: the sum of the 1D density calculated in each step of the loop
    :param df_fit: input dataframe
    :param file: filename for the plot
    :param path: output path of the plot
    :param plotting: bool
    :return: consensus-labels and number of clusters found in the consensus solution
    """
    cc = ClusterConsensus(*label_matrix)
    labels_cc = cc.fit(density=density_sum, min_cluster_size=15)
    labels_cc_clean = LabelEncoder().fit_transform(labels_cc) - 1
    if plotting:
        plot(labels=labels_cc_clean, df=df_fit, filename=file, output_pathname=path)
    nb_consensus = np.unique(labels_cc_clean[labels_cc_clean > -1]).size

    return labels_cc_clean, nb_consensus