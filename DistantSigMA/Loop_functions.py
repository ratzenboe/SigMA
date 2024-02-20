
import numpy as np
import pandas as pd

from PlotlyResults import plot
from SigMA.SigMA import SigMA


# BROKEN AND NOT USED FOR NOW
def run_loop(step, df_fit, feature, setup_kwargs, knn_list, alpha, output_path, output_file=None, num: int = None,
             plotting: bool = True):
    # ---------------------------------------------------------

    # setup kwargs
    sigma_kwargs = setup_kwargs["sigma_kwargs"]
    scale_factor_list = setup_kwargs["scale_factor_list"]

    # ---------------------------------------------------------

    # initialize SigMA with sf_mean
    clusterer = SigMA(data=df_fit, **sigma_kwargs)
    # save X_mean
    X_mean_sf = clusterer.X
    # initialize array for density values (collect the rho_sums)
    rhosum_list = []

    # Initialize array for the outer cc (occ) results (remove field stars / rfs, remove spurious clusters / rsc)
    results_rfs = np.empty(shape=(len(knn_list), len(df_fit)))
    results_rsc = np.empty(shape=(len(knn_list), len(df_fit)))
    results_simple = np.empty(shape=(len(knn_list), len(df_fit)))

    # ---------------------------------------------------------

    # Outer loop: KNN
    for kid, knn in enumerate(knn_list):

        # print(f"-- Current run with KNN = {knn} -- \n")

        label_matrix_rfs = np.empty(shape=(len(scale_factor_list), len(df_fit)))
        label_matrix_rsc = np.empty(shape=(len(scale_factor_list), len(df_fit)))
        label_matrix_simple = np.empty(shape=(len(scale_factor_list), len(df_fit)))

        # initialize density-sum over all scaling factors
        rho_sum = np.zeros(df_fit.shape[0], dtype=np.float32)

        # ---------------------------------------------------------

        # Inner loop: Scale factors
        for sf_id, sf in enumerate(scale_factor_list):
            # Set current scale factor
            scale_factors = {'pos': {'features': [feature], 'factor': sf}}
            clusterer_coarse.set_scaling_factors(scale_factors)
            print(f"Performing clustering for scale factor {clusterer_coarse.scale_factors['vel']['factor']}...")
            # Fit
            # Fit
            # st = time.time()
            label_array = clusterer.fit(alpha=alpha, knn=knn, bh_correction=True)
            # delta_t = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]
            # print(f'Done! [took {delta_t}]. Found {np.unique(label_array).size} clusters')

            # density and X
            rho, X = clusterer.weights_, clusterer.X
            rho_sum += rho

            # a) remove field stars
            nb_rfs = remove_field_stars(label_array, rho, label_matrix_rfs, sf_id)
            # b) remove spurious clusters
            nb_es, nb_rsc = extract_signal_remove_spurious(df_fit, label_array, rho, X, label_matrix_rsc, sf_id)
            # c) do new method
            nb_simple = extract_signal(label_array, clusterer, label_matrix_simple, sf_id)
            # Write the output to the hyperparameter file:
            if output_file:
                save_output_summary(summary_str={"knn": knn, "sf": sf, "n_rfs": nb_rfs, "n_rsc": nb_rsc,
                                                 "n_simple": nb_simple},
                                    file=output_file)

        # append the density sum to the list over all KNN
        rhosum_list.append(rho_sum)

        # Perform consensus clustering on the a) and b) arrays (automatically generates and saves a html-plot)
        labels_icc_rfs, n_icc_rfs = consensus_function(label_matrix_rfs, rho_sum, df_fit,
                                                       f"Step{step}_C_{num}_rfs_KNN_{knn}_ICC", output_path,
                                                       plotting=plotting)
        labels_icc_rsc, n_icc_rsc = consensus_function(label_matrix_rsc, rho_sum, df_fit,
                                                       f"Step{step}_C_{num}_rsc_KNN_{knn}_ICC", output_path,
                                                       plotting=plotting)

        labels_icc_simple, n_icc_simple = consensus_function(label_matrix_simple, rho_sum, df_fit,
                                                             f"Step{step}_C_{num}_simple_KNN_{knn}_ICC", output_path,
                                                             plotting=plotting)
        results_rfs[kid, :] = labels_icc_rfs
        results_rsc[kid, :] = labels_icc_rsc
        results_simple[kid, :] = labels_icc_simple

        print(f":: Finished run for KNN={knn}! \n. Found {n_icc_rfs} / {n_icc_rsc} / {n_icc_simple} final clusters.")

    return [results_rfs, results_rsc, results_simple], X_mean_sf, rhosum_list


def outer_consensus(step: float, num, df_fit: pd.DataFrame, knn_list: np.array, output_path: str, loop_dict: dict,
                    output_file=None):
    knn_mid = int(len(knn_list) / 2 - 1)
    df_save = df_fit

    label_lists, X_mean_sf, rhosum_list = run_loop(step=step, df_fit=df_fit, knn_list=knn_list, num=num, **loop_dict)

    # Perform consensus clustering on the c) and d) steps
    labels_occ, n_occ = zip(*(consensus_function(jl, rhosum_list[knn_mid], df_fit, f"Step{step}_C_{num}_{name}_OCC",
                                                 output_path) for jl, name in zip(label_lists, ["rfs", "rsc"])))
    n_occ = list(n_occ)
    labels_occ = list(labels_occ)

    # bring labels to the same system
    rfs_changed = rewrite_labels(labels_occ[1], labels_occ[0])
    labels_changed = rfs_changed.reshape(len(rfs_changed, ))
    plot(labels=labels_changed, df=df_fit, filename=f"Step{step}_C_{num}_rfs_OCC", output_pathname=output_path)

    n_occ.append(len(np.unique(labels_changed)))
    labels_occ.append(labels_changed)

    # clean the rfs selection again
    rfs_cleaned = np.empty(shape=(1, len(labels_changed)))
    nb_es, nb_rsc = extract_signal_remove_spurious(df_fit, labels_occ[2], rhosum_list[knn_mid], X_mean_sf,
                                                   rfs_cleaned, 0)
    results_rfs_cleaned = rfs_cleaned.reshape(len(rfs_cleaned[0, :], ))
    plot(labels=results_rfs_cleaned, df=df_fit, filename=f"Step{step}_C_{num}_rfs_cleaned_OCC",
         output_pathname=output_path)
    n_occ.append(nb_rsc)
    labels_occ.append(results_rfs_cleaned)

    if output_file:
        save_output_summary(summary_str={"knn": "occ", "n_rfs": n_occ[2], "n_rsc": n_occ[1],
                                         "n_rfs_cleaned": n_occ[3]}, file=output_file)

    # save the labels in a csv file and plot the result
    df_save["rsc"] = labels_occ[1]
    df_save["rfs"] = labels_occ[2]
    df_save["rfs_cleaned"] = labels_occ[3]
    df_save.to_csv(output_path + f"Step{step}_C_{num}_results_CC.csv")

    return df_save
