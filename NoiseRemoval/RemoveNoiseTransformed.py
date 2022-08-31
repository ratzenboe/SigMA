import numpy as np
from scipy.optimize import minimize
import networkx as nx
from miscellaneous.utils import flatten_listlist
from scipy.sparse.csgraph import connected_components
from Modality.DensityEstKNN import DensityEstKNN
from NoiseRemoval.ClusterGMM import gmm_cut
from Graph.extract_neighbors import neighboring_modes
from NoiseRemoval.OptimalVelocity import optimize_velocity, transform_velocity, transform_velocity_diff, vr_solver


def rn_obtain_data(
        data,
        cluster_bool_arr,
        te_obj,
        pos_cols,
        nb_neigh_density,
        data_full,
        ra_col, dec_col, plx_col, pmra_col, pmdec_col, rv_col, rv_err_col,
        adjacency_mtrx,
        uvw_cols=None,
        radius=8,
        verbose=False,
        min_cluster_size=10
):
    # ----- Part-I: extract dense core -------
    data_idx = np.arange(data.shape[0])
    # --- (a) Get densest components of overall mixture of cluster and BG ---
    _, cluster_labels, _, _, _, is_good_clustering = gmm_cut(te_obj.weights_[cluster_bool_arr], n_components=2)
    if not is_good_clustering:
        return None, None, None, False
    # --- (a) get labels of local cluster mode containing the peak
    cluster_modes_dense = np.unique(te_obj.leaf_labels_[data_idx[cluster_bool_arr][cluster_labels]])
    # Add neighbors of densest region that is in the cluster:
    search_for_peak = cluster_modes_dense
    for dense_cl in cluster_modes_dense:
        peak_neighbors = np.intersect1d(te_obj.leaf_labels_[cluster_bool_arr],
                                        te_obj.mode_neighbors(dense_cl))
        search_for_peak = np.union1d(search_for_peak, peak_neighbors)
    # Get dense component of peak
    cut_dense_neighs = np.isin(te_obj.leaf_labels_, search_for_peak)  # filtered points: modal and surrounding regions
    _, cluster_labels_filter, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(te_obj.weights_[cut_dense_neighs],
                                                                                n_components=2)
    if not is_good_clustering:
        return None, None, None, False
    # Dense core points
    cut_dense_core = data_idx[cut_dense_neighs][cluster_labels_filter]  # translate bool arr to data index
    if cut_dense_core.size < min_cluster_size:
        return None, None, None, False

    # ---- Part-II: Compute "optimal" cartesian velocity ----
    # Prepare data
    cols = [ra_col, dec_col, plx_col, pmra_col, pmdec_col, rv_col, rv_err_col]
    ra, dec, plx, pmra, pmdec, rv, rv_err = data_full.iloc[cut_dense_core][cols].values.T
    # Prepare initial guess
    mean_uvw = np.zeros(3)
    if uvw_cols is not None:
        mean_uvw = np.mean(data_full.iloc[cut_dense_core][uvw_cols], axis=0)
    # Compute optimal velocity
    sol = optimize_velocity(ra, dec, plx, pmra, pmdec, rv, rv_err, init_guess=mean_uvw, do_minimize=True)
    optimal_vel = sol.x
    # calculate rv for cases without rv estimations or very large errors
    # Minimize distance to optimal velocity for the following data points (cut_dense_neighs)
    ra, dec, plx, pmra, pmdec, rv, rv_err = data_full.iloc[cut_dense_neighs][cols].values.T
    idx_arr = np.arange(rv.size)
    rv_isnan_or_large_err = np.isnan(rv)  # | (np.abs(rv / rv_err) < 2)  # for large errors find better suited rvs
    # Estimating rv for ~rv_isnan_or_large_err sources
    rv_computed = np.copy(rv)
    rv_computed[rv_isnan_or_large_err] = vr_solver(U=optimal_vel[0], V=optimal_vel[1], W=optimal_vel[2],
                                                   ra=ra[rv_isnan_or_large_err],
                                                   dec=dec[rv_isnan_or_large_err],
                                                   plx=plx[rv_isnan_or_large_err],
                                                   pmra=pmra[rv_isnan_or_large_err],
                                                   pmdec=pmdec[rv_isnan_or_large_err]
                                                   )
    # Prepare bool array for data
    cluster_member_arr = np.zeros(data_full.shape[0], dtype=int)

    # Transform to uvw
    uvw_computed = transform_velocity(ra, dec, plx, pmra, pmdec, rv_computed)
    # only care about velocities near the optimal velocity -> others have too different space velocity
    uvw_calc_diff = np.linalg.norm(uvw_computed - optimal_vel, axis=1)
    # differences larger than radius (default=20) are very likely not part of stellar system
    cut_uvw_diff = uvw_calc_diff < radius
    if np.sum(cut_uvw_diff) < min_cluster_size:
        # if only very few stars are found that share the same velocity, focus on the dense core
        return np.isin(data_idx, data_idx[cut_dense_neighs][cut_uvw_diff]), fp_rate, fn_rate, False
    if verbose:
        print(f':: 3rd filter step: {np.sum(cut_uvw_diff)} in proximity to optimal velocity')

    # Scale XYZ:
    scale = 5

    xyzuvw = np.c_[data_full.iloc[cut_dense_neighs][pos_cols].values / scale, uvw_computed]
    # Compute densities
    duvw = DensityEstKNN(xyzuvw, nb_neigh_density)
    rho_uvw = duvw.knn_density(nb_neigh_density)
    # Predict membership via GMM with 2 components
    _, cut_gmm_xyzuvw, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(rho_uvw[cut_uvw_diff], n_components=2)

    # Extract connected component from dense component
    _, cc_idx = connected_components(
        adjacency_mtrx[cut_dense_neighs, :][:, cut_dense_neighs][cut_uvw_diff, :][:, cut_uvw_diff][cut_gmm_xyzuvw,
        :][:,
        cut_gmm_xyzuvw])
    # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
    cluster_indices = data_idx[cut_dense_neighs][cut_uvw_diff][cut_gmm_xyzuvw][
        cc_idx == np.argmax(np.bincount(cc_idx))]
    cluster_member_arr[cluster_indices] += 1

    return cluster_member_arr == 1, rho_uvw, {'rv_computed': rv_computed,
                                              'data_idx': data_idx[cut_dense_neighs]}, True



def remove_noise_sigma(data, cluster_bool_arr, te_obj,
                       pos_cols, nb_neigh_density,
                       data_full, ra_col, dec_col, plx_col, pmra_col, pmdec_col, rv_col, rv_err_col,
                       adjacency_mtrx, uvw_cols=None, radius=8, verbose=False, min_cluster_size=10
                       ):
    # ----- Part-I: extract dense core -------
    data_idx = np.arange(data.shape[0])
    # --- (a) Get densest components of overall mixture of cluster and BG ---
    _, cluster_labels, _, _, _, is_good_clustering = gmm_cut(te_obj.weights_[cluster_bool_arr], n_components=2)
    if not is_good_clustering:
        return None, None, None, False
    # --- (a) get labels of local cluster mode containing the peak
    cluster_modes_dense = np.unique(te_obj.leaf_labels_[data_idx[cluster_bool_arr][cluster_labels]])
    # Add neighbors of densest region that is in the cluster:
    search_for_peak = cluster_modes_dense
    for dense_cl in cluster_modes_dense:
        peak_neighbors = np.intersect1d(te_obj.leaf_labels_[cluster_bool_arr],
                                        te_obj.mode_neighbors(dense_cl))
        search_for_peak = np.union1d(search_for_peak, peak_neighbors)
    # Get dense component of peak
    cut_dense_neighs = np.isin(te_obj.leaf_labels_, search_for_peak)  # filtered points: modal and surrounding regions
    _, cluster_labels_filter, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(te_obj.weights_[cut_dense_neighs],
                                                                                n_components=2)
    if not is_good_clustering:
        return None, None, None, False
    # Dense core points
    cut_dense_core = data_idx[cut_dense_neighs][cluster_labels_filter]  # translate bool arr to data index
    if cut_dense_core.size < min_cluster_size:
        return None, None, None, False

    # ---- Part-II: Compute "optimal" cartesian velocity ----
    # Prepare data
    cols = [ra_col, dec_col, plx_col, pmra_col, pmdec_col, rv_col, rv_err_col]
    ra, dec, plx, pmra, pmdec, rv, rv_err = data_full.iloc[cut_dense_core][cols].values.T
    # Prepare initial guess
    mean_uvw = np.zeros(3)
    if uvw_cols is not None:
        mean_uvw = np.mean(data_full.iloc[cut_dense_core][uvw_cols], axis=0)
    # Compute optimal velocity
    sol = optimize_velocity(ra, dec, plx, pmra, pmdec, rv, rv_err, init_guess=mean_uvw, do_minimize=True)
    optimal_vel = sol.x
    # calculate rv for cases without rv estimations or very large errors
    # Minimize distance to optimal velocity for the following data points (cut_dense_neighs)
    ra, dec, plx, pmra, pmdec, rv, rv_err = data_full.iloc[cut_dense_neighs][cols].values.T
    idx_arr = np.arange(rv.size)
    rv_isnan_or_large_err = np.isnan(rv)  # | (np.abs(rv / rv_err) < 2)  # for large errors find better suited rvs
    # Estimating rv for ~rv_isnan_or_large_err sources
    rv_computed = np.copy(rv)
    rv_computed[rv_isnan_or_large_err] = vr_solver(U=optimal_vel[0], V=optimal_vel[1], W=optimal_vel[2],
                                                   ra=ra[rv_isnan_or_large_err],
                                                   dec=dec[rv_isnan_or_large_err],
                                                   plx=plx[rv_isnan_or_large_err],
                                                   pmra=pmra[rv_isnan_or_large_err],
                                                   pmdec=pmdec[rv_isnan_or_large_err]
                                                   )
    # Prepare bool array for data
    cluster_member_arr = np.zeros(data_full.shape[0], dtype=int)

    # Transform to uvw
    uvw_computed = transform_velocity(ra, dec, plx, pmra, pmdec, rv_computed)
    # only care about velocities near the optimal velocity -> others have too different space velocity
    uvw_calc_diff = np.linalg.norm(uvw_computed - optimal_vel, axis=1)
    # differences larger than radius (default=20) are very likely not part of stellar system
    cut_uvw_diff = uvw_calc_diff < radius
    if np.sum(cut_uvw_diff) < min_cluster_size:
        # if only very few stars are found that share the same velocity, focus on the dense core
        return np.isin(data_idx, data_idx[cut_dense_neighs][cut_uvw_diff]), fp_rate, fn_rate, False
    if verbose:
        print(f':: 3rd filter step: {np.sum(cut_uvw_diff)} in proximity to optimal velocity')

    # Scale XYZ:
    # scales range from ~2-10 assuming the density in velocity is constant
    # while the space density can vary from a dense core to a less dense corona
    contamination_fraction = []
    completeness_fraction = []
    for scale in np.linspace(2, 10, 20):
        xyzuvw = np.c_[data_full.iloc[cut_dense_neighs][pos_cols].values / scale, uvw_computed]
        # Compute densities
        duvw = DensityEstKNN(xyzuvw, nb_neigh_density)
        rho_uvw = duvw.knn_density(nb_neigh_density)
        # Predict membership via GMM with 2 components
        _, cut_gmm_xyzuvw, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(rho_uvw[cut_uvw_diff], n_components=2)
        contamination_fraction.append(fp_rate)
        completeness_fraction.append(fn_rate)

        if not is_good_clustering:
            return None, None, None, False

        # Extract connected component from dense component
        _, cc_idx = connected_components(
            adjacency_mtrx[cut_dense_neighs, :][:, cut_dense_neighs][cut_uvw_diff, :][:, cut_uvw_diff][cut_gmm_xyzuvw,
            :][:,
            cut_gmm_xyzuvw])
        # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
        cluster_indices = data_idx[cut_dense_neighs][cut_uvw_diff][cut_gmm_xyzuvw][
            cc_idx == np.argmax(np.bincount(cc_idx))]
        cluster_member_arr[cluster_indices] += 1

    # Mean contamination fraction
    mean_contamination_fraction = np.mean(contamination_fraction)
    mean_completeness_fraction = np.mean(completeness_fraction)

    # Prepare output
    final_clustering_strict = cluster_member_arr >= 10  # More than 75% of hits
    if np.sum(final_clustering_strict) >= min_cluster_size:
        # Keep connected components
        _, cc_idx = connected_components(adjacency_mtrx[final_clustering_strict, :][:, final_clustering_strict])
        # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
        cluster_indices = data_idx[final_clustering_strict][cc_idx == np.argmax(np.bincount(cc_idx))]
        cluster_final_xyzuvw = np.isin(data_idx, cluster_indices)
        return cluster_final_xyzuvw, mean_contamination_fraction, mean_completeness_fraction, True
    else:
        return final_clustering_strict, mean_contamination_fraction, mean_completeness_fraction, False


def remove_noise(data, cluster_bool_arr, G, pos_cols, labels, density, nb_neigh_denstiy,
                 data_full, ra_col, dec_col, plx_col, pmra_col, pmdec_col, rv_col, rv_err_col,
                 adjacency_mtrx, uvw_cols=None, radius=8, isin_points=None, verbose=False, min_cluster_size=10
                 ):
    """Remove noise for a given cluster
    :param data: full data set
    :param cluster_bool_arr: bool array highlighting the cluster
    :param G: the MST graph describing the modes and their connection via saddle points
    :param pos_cols: poition columns (needed for combination of new feature space)
    :param labels: labels for each initial mode appearing in the data set
    :param density: point density estimate, usually via KNN density estimation
    :param nb_neigh_denstiy: number of neighbors to use for denstiy estimation
    """
    data_idx = np.arange(data.shape[0])
    # Get densest components in the given cluster
    _, cluster_labels, _, is_good_clustering = gmm_cut(density[cluster_bool_arr], n_components=2)
    if not is_good_clustering:
        return None, False
    if verbose:
        print(f':: 1st filter step: {np.sum(cluster_labels)} in densest component')
    # get labels of local cluster mode containing the peak
    cluster_modes_dense = np.unique(labels[data_idx[cluster_bool_arr][cluster_labels]])
    # extract connected components from cluster_modes_dense (via G)
    nbs_saddle = np.array(flatten_listlist([list(int(n) for n in G.neighbors(cmd)) for cmd in cluster_modes_dense]))
    nodes_to_search = np.union1d(cluster_modes_dense, nbs_saddle)
    dense_subgraph = G.subgraph(nodes_to_search)
    largest_cc = np.array(list(max(nx.connected_components(dense_subgraph), key=len)), dtype=int)
    cluster_modes_dense = np.intersect1d(largest_cc, labels)

    # Get modes surrounding the dense cluster core
    nbs_modes = neighboring_modes(cluster_modes_dense, G, nb_neighbors=1)
    # Remove neighboring nodes that are not in the cluster
    nbs_modes = np.intersect1d(nbs_modes, np.unique(labels[cluster_bool_arr]))
    cut_filter = np.isin(labels, nbs_modes)  # filtered points: modal and surrounding regions
    rho_fitlered = density[cut_filter]  # get density of filtered points
    _, cluster_labels_filter, _, is_good_clustering = gmm_cut(rho_fitlered, n_components=2)  # dense core points of this region
    if not is_good_clustering:
        return None, False
    if verbose:
        print(f':: 2nd filter step: {np.sum(cluster_labels_filter)} in dense core with surrounding modes')
    cut_dense_core = data_idx[cut_filter][cluster_labels_filter]  # translate bool arr to data index

    # Test if the dense part overlaps with
    if isin_points is not None:
        if np.intersect1d(cut_dense_core, data_idx[isin_points]).size < min_cluster_size:
            # If cluster contains too few unrelated points we are not interested in it
            return None, False

    # ---- Compute "optimal" cartesian velocity ----
    # Prepare data
    cols = [ra_col, dec_col, plx_col, pmra_col, pmdec_col, rv_col, rv_err_col]
    ra, dec, plx, pmra, pmdec, rv, rv_err = data_full.loc[cut_dense_core, cols].values.T
    # Prepare initial guess
    mean_uvw = np.zeros(3)
    if uvw_cols is not None:
        mean_uvw = np.mean(data_full.loc[cut_dense_core, uvw_cols], axis=0)
    # Compute optimal velocity
    sol = optimize_velocity(ra, dec, plx, pmra, pmdec, rv, rv_err, init_guess=mean_uvw, do_minimize=True)
    optimal_vel = sol.x
    # Compute propermotions under given optimal 3D velocity of full sample
    ra, dec, plx, pmra, pmdec, rv, rv_err = data_full.loc[
        cut_filter, [ra_col, dec_col, plx_col, pmra_col, pmdec_col, rv_col, rv_err_col]].values.T
    # Find best fitting rvs for given data
    # calculate rv for cases without rv estimations or very large errors
    idx_arr = np.arange(rv.size)
    rv_isnan_or_large_err = np.isnan(rv) | (np.abs(rv / rv_err) < 2)  # for large errors find better suited rvs
    list_op_rvs = []
    for i in idx_arr[rv_isnan_or_large_err]:
        opt_rv = minimize(fun=transform_velocity_diff, x0=0.,
                          args=(ra[i], dec[i], plx[i], pmra[i], pmdec[i], optimal_vel))
        list_op_rvs.append(opt_rv.x[0])
    # Set optimal rv's
    rv_computed = np.copy(rv)
    rv_computed[rv_isnan_or_large_err] = np.array(list_op_rvs)

    # Prepare bool array for data
    cluster_member_arr = np.zeros(data_full.shape[0], dtype=int)

    # Transform to uvw
    uvw_computed = transform_velocity(ra, dec, plx, pmra, pmdec, rv_computed)
    # only care about velocities near the optimal velocity -> others have too different space velocity
    uvw_calc_diff = np.linalg.norm(uvw_computed - optimal_vel, axis=1)
    # differences larger than radius (default=20) are very likely not part of stellar system
    cut_uvw_diff = uvw_calc_diff < radius
    if np.sum(cut_uvw_diff) < min_cluster_size:
        # if only very few stars are found that share the same velocity, focus on the dense core
        return np.isin(data_idx, data_idx[cut_filter][cut_uvw_diff]), False
    if verbose:
        print(f':: 3rd filter step: {np.sum(cut_uvw_diff)} in proximity to optimal velocity')

    # Scale XYZ:
    # scales range from ~2-10 assuming the density in velocity is constant
    # while the space density can vary from a dense core to a less dense corona
    for scale in np.linspace(2, 10, 20):
        xyzuvw = np.c_[data_full.loc[cut_filter, pos_cols].values / scale, uvw_computed]
        # Compute densities
        duvw = DensityEstKNN(xyzuvw, nb_neigh_denstiy)
        rho_uvw = duvw.knn_density(nb_neigh_denstiy)
        # Predict membership via GMM with 2 components
        _, cut_gmm_xyzuvw, _, is_good_clustering = gmm_cut(rho_uvw[cut_uvw_diff], n_components=2)
        if not is_good_clustering:
            return None, False

            # Extract connected component from dense component
        _, cc_idx = connected_components(
            adjacency_mtrx[cut_filter, :][:, cut_filter][cut_uvw_diff, :][:, cut_uvw_diff][cut_gmm_xyzuvw, :][:,
            cut_gmm_xyzuvw])
        # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
        cluster_indices = data_idx[cut_filter][cut_uvw_diff][cut_gmm_xyzuvw][cc_idx == np.argmax(np.bincount(cc_idx))]
        cluster_member_arr[cluster_indices] += 1

    final_clustering_strict = cluster_member_arr >= 15  # More than 75% of hits
    if np.sum(final_clustering_strict) >= min_cluster_size:
        # Keep connected components
        _, cc_idx = connected_components(adjacency_mtrx[final_clustering_strict, :][:, final_clustering_strict])
        # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
        cluster_indices = data_idx[final_clustering_strict][cc_idx == np.argmax(np.bincount(cc_idx))]
        cluster_final_xyzuvw = np.isin(data_idx, cluster_indices)
        return cluster_final_xyzuvw, True
    else:
        return final_clustering_strict, False


def remove_noise_simple(cluster_bool_arr, te_obj, adjacency_mtrx=None):
    """Remove noise with only gmms"""
    # ----- Part-I: extract dense core -------
    data_idx = np.arange(te_obj.X.shape[0])
    # Get densest components in the given cluster
    _, cluster_labels, _, _, _, is_good_clustering = gmm_cut(te_obj.weights_[cluster_bool_arr], n_components=2)
    if not is_good_clustering:
        return None
    # get labels of local cluster mode containing the peak
    cluster_modes_dense = np.unique(te_obj.leaf_labels_[data_idx[cluster_bool_arr][cluster_labels]])
    # Add neighbors of densest region that is in the cluster:
    search_for_peak = cluster_modes_dense
    for dense_cl in cluster_modes_dense:
        peak_neighbors = np.intersect1d(te_obj.leaf_labels_[cluster_bool_arr],
                                        te_obj.mode_neighbors(dense_cl))
        search_for_peak = np.union1d(search_for_peak, peak_neighbors)
    # Get dense component of peak
    cut_dense_neighs = np.isin(te_obj.leaf_labels_, search_for_peak)  # filtered points: modal and surrounding regions
    _, cluster_labels_filter, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(te_obj.weights_[cut_dense_neighs],
                                                                                n_components=2)
    if not is_good_clustering:
        return None
    # Dense core points
    cut_dense_core = np.isin(data_idx, data_idx[cut_dense_neighs][cluster_labels_filter])

    if adjacency_mtrx is None:
        adjacency_mtrx = te_obj.A

    _, cc_idx = connected_components(adjacency_mtrx[cut_dense_core, :][:, cut_dense_core])
    # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
    cluster_indices = data_idx[cut_dense_neighs][cluster_labels_filter][cc_idx == np.argmax(np.bincount(cc_idx))]
    return np.isin(data_idx, cluster_indices)
