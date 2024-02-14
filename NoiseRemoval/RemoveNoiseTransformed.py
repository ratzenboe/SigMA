import numpy as np
from scipy.sparse.csgraph import connected_components
from SigMA.DensityEstimator import DensityEstKNN
from NoiseRemoval.ClusterGMM import gmm_cut
from NoiseRemoval.OptimalVelocity import (
    optimize_velocity,
    transform_velocity,
    vr_solver,
)


def rn_obtain_data(
    data_full,
    cluster_bool_arr,
    te_obj,
    pos_cols,
    nb_neigh_density,
    ra_col,
    dec_col,
    plx_col,
    pmra_col,
    pmdec_col,
    rv_col,
    pmra_err_col,
    pmdec_err_col,
    rv_err_col,
    adjacency_mtrx,
    uvw_cols=None,
    radius=8,
    verbose=False,
    min_cluster_size=10,
):
    # ----- Part-I: extract dense core -------
    data_idx = np.arange(data_full.shape[0])
    # --- (a) Get densest components of overall mixture of cluster and BG ---
    _, cluster_labels, _, _, _, is_good_clustering = gmm_cut(
        te_obj.weights_[cluster_bool_arr], n_components=2
    )
    if not is_good_clustering:
        return None, None, None, False
    # --- (a) get labels of local cluster mode containing the peak
    cluster_modes_dense = np.unique(
        te_obj.leaf_labels_[data_idx[cluster_bool_arr][cluster_labels]]
    )
    # Add neighbors of densest region that is in the cluster:
    search_for_peak = cluster_modes_dense
    for dense_cl in cluster_modes_dense:
        peak_neighbors = np.intersect1d(
            te_obj.leaf_labels_[cluster_bool_arr], te_obj.mode_neighbors(dense_cl)
        )
        search_for_peak = np.union1d(search_for_peak, peak_neighbors)
    # Get dense component of peak
    cut_dense_neighs = np.isin(
        te_obj.leaf_labels_, search_for_peak
    )  # filtered points: modal and surrounding regions
    _, cluster_labels_filter, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(
        te_obj.weights_[cut_dense_neighs], n_components=2
    )
    if not is_good_clustering:
        return None, None, None, False
    # Dense core points
    cut_dense_core = data_idx[cut_dense_neighs][
        cluster_labels_filter
    ]  # translate bool arr to data index
    if cut_dense_core.size < min_cluster_size:
        return None, None, None, False

    # ---- Part-II: Compute "optimal" cartesian velocity ----
    # Prepare data
    cols = [
        ra_col,
        dec_col,
        plx_col,
        pmra_col,
        pmdec_col,
        rv_col,
        pmra_err_col,
        pmdec_err_col,
        rv_err_col,
    ]
    ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err = data_full.iloc[
        cut_dense_core
    ][cols].values.T
    # Prepare initial guess
    mean_uvw = np.zeros(3)
    if uvw_cols is not None:
        mean_uvw = np.mean(data_full.iloc[cut_dense_core][uvw_cols], axis=0)
    # Compute optimal velocity
    sol = optimize_velocity(
        ra,
        dec,
        plx,
        pmra,
        pmdec,
        rv,
        pmra_err,
        pmdec_err,
        rv_err,
        init_guess=mean_uvw,
        do_minimize=True,
    )
    optimal_vel = sol.x
    # calculate rv for cases without rv estimations or very large errors
    # Minimize distance to optimal velocity for the following data points (cut_dense_neighs)
    ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err = data_full.iloc[
        cut_dense_neighs
    ][cols].values.T
    idx_arr = np.arange(rv.size)
    rv_isnan_or_large_err = np.isnan(rv)
    # Estimating rv for ~rv_isnan_or_large_err sources
    rv_computed = np.copy(rv)
    rv_computed[rv_isnan_or_large_err] = vr_solver(
        U=optimal_vel[0],
        V=optimal_vel[1],
        W=optimal_vel[2],
        ra=ra[rv_isnan_or_large_err],
        dec=dec[rv_isnan_or_large_err],
        plx=plx[rv_isnan_or_large_err],
        pmra=pmra[rv_isnan_or_large_err],
        pmdec=pmdec[rv_isnan_or_large_err],
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
        return (
            np.isin(data_idx, data_idx[cut_dense_neighs][cut_uvw_diff]),
            fp_rate,
            fn_rate,
            False,
        )
    if verbose:
        print(
            f":: 3rd filter step: {np.sum(cut_uvw_diff)} in proximity to optimal velocity"
        )

    # Scale XYZ:
    scale = 5

    xyzuvw = np.c_[
        data_full.iloc[cut_dense_neighs][pos_cols].values / scale, uvw_computed
    ]
    # Compute densities
    duvw = DensityEstKNN(xyzuvw, nb_neigh_density)
    rho_uvw = duvw.knn_density(nb_neigh_density)
    # Predict membership via GMM with 2 components
    _, cut_gmm_xyzuvw, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(
        rho_uvw[cut_uvw_diff], n_components=2
    )

    # Extract connected component from dense component
    _, cc_idx = connected_components(
        adjacency_mtrx[cut_dense_neighs, :][:, cut_dense_neighs][cut_uvw_diff, :][
            :, cut_uvw_diff
        ][cut_gmm_xyzuvw, :][:, cut_gmm_xyzuvw]
    )
    # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
    cluster_indices = data_idx[cut_dense_neighs][cut_uvw_diff][cut_gmm_xyzuvw][
        cc_idx == np.argmax(np.bincount(cc_idx))
    ]
    cluster_member_arr[cluster_indices] += 1

    return (
        cluster_member_arr == 1,
        rho_uvw,
        {"rv_computed": rv_computed, "data_idx": data_idx[cut_dense_neighs]},
        True,
    )


def remove_noise_quick_n_dirty(data_density, cluster_bool_arr, n_components=2):
    # --- (a) Get densest components of overall mixture of cluster and BG ---
    _, cluster_labels, _, _, _, is_good_clustering = gmm_cut(
        data_density[cluster_bool_arr], n_components=n_components
    )
    if not is_good_clustering:
        return None, False
    return cluster_labels, True


def remove_noise_sigma(
    data_full,
    cluster_bool_arr,
    rho,
    pos_cols,
    nb_neigh_density,
    rv_min,
    rv_max,
    ra_col,
    dec_col,
    plx_col,
    pmra_col,
    pmdec_col,
    rv_col,
    pmra_err_col,
    pmdec_err_col,
    rv_err_col,
    adjacency_mtrx,
    uvw_cols=None,
    radius=8,
    verbose=False,
    min_cluster_size=10,
):
    # ----- Part-I: extract dense core -------
    data_idx = np.arange(data_full.shape[0])
    # --- (a) Get densest components of overall mixture of cluster and BG ---
    _, cluster_labels, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(
        rho[cluster_bool_arr], n_components=2
    )
    if not is_good_clustering:
        return None, None, None, False
    # # --- (b) get labels of local cluster mode containing the peak
    # cluster_modes_dense = np.unique(te_obj.leaf_labels_[data_idx[cluster_bool_arr][cluster_labels]])
    # # Add neighbors of most dense region that is in the cluster:
    # search_for_peak = cluster_modes_dense
    # for dense_cl in cluster_modes_dense:
    #     peak_neighbors = np.intersect1d(te_obj.leaf_labels_[cluster_bool_arr],
    #                                     te_obj.mode_neighbors(dense_cl))
    #     search_for_peak = np.union1d(search_for_peak, peak_neighbors)
    # # Get dense component of peak
    # cut_dense_neighs = np.isin(te_obj.leaf_labels_, search_for_peak)  # filtered points: modal and surrounding regions
    # _, cluster_labels_filter, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(
    #     te_obj.weights_[cut_dense_neighs],
    #     n_components=2
    # )
    if not is_good_clustering:
        return None, None, None, False
    # Dense core points
    cut_dense_core = data_idx[cluster_bool_arr][
        cluster_labels
    ]  # translate bool arr to data index
    if cut_dense_core.size < min_cluster_size:
        return None, None, None, False

    # ---- Part-II: Compute "optimal" cartesian velocity ----
    # Prepare data
    cols = [
        ra_col,
        dec_col,
        plx_col,
        pmra_col,
        pmdec_col,
        rv_col,
        pmra_err_col,
        pmdec_err_col,
        rv_err_col,
    ]
    ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err = data_full.iloc[
        cut_dense_core
    ][cols].values.T
    # Prepare initial guess
    mean_uvw = np.zeros(3)
    if uvw_cols is not None:
        mean_uvw = np.mean(data_full.iloc[cut_dense_core][uvw_cols], axis=0)
    # Compute optimal velocity
    sol = optimize_velocity(
        ra,
        dec,
        plx,
        pmra,
        pmdec,
        rv,
        pmra_err,
        pmdec_err,
        rv_err,
        init_guess=mean_uvw,
        do_minimize=True,
    )
    optimal_vel = sol.x
    # calculate rv for cases without rv estimations or very large errors
    # Minimize distance to optimal velocity for the following data points (cut_dense_neighs)
    ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err = data_full.iloc[
        cluster_bool_arr
    ][cols].values.T
    rv_isnan_or_large_err = np.isnan(
        rv
    )  # | (np.abs(rv / rv_err) < 2)  # for large errors find better suited rvs
    # Estimating rv for ~rv_isnan_or_large_err sources
    rv_computed = np.copy(rv)
    rv_computed[rv_isnan_or_large_err] = vr_solver(
        U=optimal_vel[0],
        V=optimal_vel[1],
        W=optimal_vel[2],
        ra=ra[rv_isnan_or_large_err],
        dec=dec[rv_isnan_or_large_err],
        plx=plx[rv_isnan_or_large_err],
        pmra=pmra[rv_isnan_or_large_err],
        pmdec=pmdec[rv_isnan_or_large_err],
    )
    # Bound radial velocity by physical bounds
    rv_computed[rv_isnan_or_large_err & (rv_computed > rv_max)] = rv_max
    rv_computed[rv_isnan_or_large_err & (rv_computed < rv_min)] = rv_min
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
        return (
            np.isin(data_idx, data_idx[cluster_bool_arr][cut_uvw_diff]),
            fp_rate,
            fn_rate,
            False,
        )
    if verbose:
        print(
            f":: 3rd filter step: {np.sum(cut_uvw_diff)} in proximity to optimal velocity"
        )

    # Scale XYZ:
    # scales range from ~2-10 assuming the density in velocity is constant
    # while the space density can vary from a dense core to a less dense corona
    contamination_fraction = []
    completeness_fraction = []
    rho2fit = []
    for scale in np.linspace(2, 10, 6):
        xyzuvw = np.c_[
            data_full.iloc[cluster_bool_arr][pos_cols].values / scale, uvw_computed
        ]
        # Compute densities
        duvw = DensityEstKNN(xyzuvw, nb_neigh_density)
        rho_uvw = duvw.knn_density(nb_neigh_density)
        # Predict membership via GMM with 2 components
        _, cut_gmm_xyzuvw, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(
            rho_uvw[cut_uvw_diff], n_components="auto"
        )
        contamination_fraction.append(fp_rate)
        completeness_fraction.append(fn_rate)
        rho2fit.append(rho_uvw[cut_uvw_diff])

        if not is_good_clustering:
            return None, None, None, False

        # Extract connected component from dense component
        _, cc_idx = connected_components(
            adjacency_mtrx[cluster_bool_arr, :][:, cluster_bool_arr][cut_uvw_diff, :][
                :, cut_uvw_diff
            ][cut_gmm_xyzuvw, :][:, cut_gmm_xyzuvw]
        )

        if cc_idx.size < min_cluster_size:
            return None, None, None, False

        # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
        cluster_indices = data_idx[cluster_bool_arr][cut_uvw_diff][cut_gmm_xyzuvw][
            cc_idx == np.argmax(np.bincount(cc_idx))
        ]
        cluster_member_arr[cluster_indices] += 1

    # Mean contamination fraction
    mean_contamination_fraction = np.median(contamination_fraction)
    mean_completeness_fraction = np.median(completeness_fraction)
    contamination_completeness = {
        "contamination": mean_contamination_fraction,
        "completeness": mean_completeness_fraction,
        "rho2fit": rho2fit,
        "data2fit": data_idx[cluster_bool_arr][cut_uvw_diff],
    }
    rv_info = {"rv_computed": rv_computed, "data_idx": data_idx[cluster_bool_arr]}

    # Prepare output
    final_clustering_strict = cluster_member_arr >= 3  # More than 50% of hits
    if np.sum(final_clustering_strict) >= min_cluster_size:
        # Keep connected components
        _, cc_idx = connected_components(
            adjacency_mtrx[final_clustering_strict, :][:, final_clustering_strict]
        )
        # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
        cluster_indices = data_idx[final_clustering_strict][
            cc_idx == np.argmax(np.bincount(cc_idx))
        ]
        cluster_final_xyzuvw = np.isin(data_idx, cluster_indices)
        return cluster_final_xyzuvw, contamination_completeness, rv_info, True
    else:
        return final_clustering_strict, contamination_completeness, rv_info, False


def remove_noise_simple(cluster_bool_arr, te_obj):
    """Remove noise with only gmms"""
    adjacency_mtrx = te_obj.A
    # ----- Part-I: extract dense core -------
    data_idx = np.arange(te_obj.X.shape[0])
    # Get most dense components in the given cluster
    _, cluster_labels, _, _, _, is_good_clustering = gmm_cut(
        te_obj.weights_[cluster_bool_arr], n_components=2
    )
    if not is_good_clustering:
        return None
    # get labels of local cluster mode containing the peak
    cluster_modes_dense = np.unique(
        te_obj.leaf_labels_[data_idx[cluster_bool_arr][cluster_labels]]
    )
    # Add neighbors of most dense regions that is in the cluster:
    search_for_peak = cluster_modes_dense
    for dense_cl in cluster_modes_dense:
        peak_neighbors = np.intersect1d(
            te_obj.leaf_labels_[cluster_bool_arr], te_obj.mode_neighbors(dense_cl)
        )
        search_for_peak = np.union1d(search_for_peak, peak_neighbors)
    # Get dense component of peak
    cut_dense_neighs = np.isin(
        te_obj.leaf_labels_, search_for_peak
    )  # filtered points: modal and surrounding regions
    _, cluster_labels_filter, _, fp_rate, fn_rate, is_good_clustering = gmm_cut(
        te_obj.weights_[cut_dense_neighs], n_components=2
    )
    if not is_good_clustering:
        return None
    # Dense core points
    cut_dense_core = np.isin(
        data_idx, data_idx[cut_dense_neighs][cluster_labels_filter]
    )

    if adjacency_mtrx is None:
        adjacency_mtrx = te_obj.A

    _, cc_idx = connected_components(
        adjacency_mtrx[cut_dense_core, :][:, cut_dense_core]
    )
    # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
    try:
        cluster_indices = data_idx[cut_dense_neighs][cluster_labels_filter][
            cc_idx == np.argmax(np.bincount(cc_idx))
        ]
        return np.isin(data_idx, cluster_indices)
    except ValueError:
        return None


def remove_noise_gmm(
    cluster_bool_arr, te_obj, adjacency_mtrx=None, n_components="auto"
):
    """Remove noise with only gmms"""
    # ----- Part-I: extract dense core -------
    data_idx = np.arange(te_obj.X.shape[0])
    # Get most dense components in the given cluster
    _, cluster_labels, _, _, _, is_good_clustering = gmm_cut(
        te_obj.weights_[cluster_bool_arr], n_components=n_components
    )
    if not is_good_clustering:
        return None
    # Dense core points
    cut_dense_core = np.isin(data_idx, data_idx[cluster_bool_arr][cluster_labels])
    # Get dense component of peak
    if adjacency_mtrx is None:
        adjacency_mtrx = te_obj.A
    _, cc_idx = connected_components(
        adjacency_mtrx[cut_dense_core, :][:, cut_dense_core]
    )
    # Combine CCs data points with originally defined dense core (to not miss out on potentially dropped points)
    try:
        cluster_indices = data_idx[cluster_bool_arr][cluster_labels][
            cc_idx == np.argmax(np.bincount(cc_idx))
        ]
        return np.isin(data_idx, cluster_indices)
    except ValueError:
        return None
