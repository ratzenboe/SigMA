"""
20-02-24
Update 15-1-24: addition to existing routine and overall check of routine.

Routine that uses the 5D parameter space (X, Y, Z, v_a_lsr, v_d_lsr) and a high KNN number to segment the
 initial field into large chunks on which a secondary clustering with other features (RA, DEC, parallax, pmra, pmdec)
 is undertaken. The information on the chunks is retained via training RF for each scale factor and combining the
 resulting predictions.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from SigMA.SigMA import SigMA

from DistantSigMA.IsochroneArchive.myTools import my_utility
from DistantSigMA.DistantSigMA.clustering_setup import setup_Cartesian_ps, read_in_file
from DistantSigMA.DistantSigMA.coarse import get_segments, merge_subsets
from DistantSigMA.DistantSigMA.PlotlyResults import plot

# 1.) Paths
# ---------------------------------------------------------
# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
script_name = my_utility.get_calling_script_name(__file__)
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/', script_name=script_name)

# 2.) Data
# ---------------------------------------------------------
# load the dataframe
cols = ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pm', 'pmra', 'pmra_error',
        'pmdec',
        'pmdec_error', 'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr',
        'dec_pmra_corr',
        'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 'astrometric_sigma5d_max',
        'ruwe',
        'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_g_mean_mag', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
        'phot_bp_mean_mag', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'phot_rp_mean_mag',
        'phot_bp_rp_excess_factor',
        'radial_velocity', 'radial_velocity_error', 'rv_method_used', 'X', 'Y', 'Z', 'v_a_lsr', 'v_d_lsr']

df_focus = pd.read_csv('../Data/Gaia/Orion_Taurus_full_galCoords_19-2-24.csv',
                       usecols=cols, na_values="nan")

# df_focus = pd.read_csv('../Data/Gaia/data_orion_focus.csv')

# 3.) Parameters
# ---------------------------------------------------------
# Define all FIXED SigMA parameters
beta = 0.99
knn_initcluster_graph = 35
KNN_list = [300]

# coarse clustering
alpha = 0.01
scaling = "bayesian"
bh_correction = True

# setup kwargs
setup_kwargs = setup_Cartesian_ps(df_fit=df_focus, KNN_list=KNN_list, beta=beta,
                                  knn_initcluster_graph=knn_initcluster_graph)

sigma_kwargs = setup_kwargs["sigma_kwargs"]
scale_factor_list = setup_kwargs["scale_factor_list"][4:6]


# 4.) Clustering
# ---------------------------------------------------------
# initialize SigMA with sf_mean
clusterer_coarse = SigMA(data=df_focus, **sigma_kwargs)

for knn in KNN_list:

    # initialize the density sum
    rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)

    # initialize the label matrix for the different scaling factors
    label_matrix_coarse = np.empty(shape=(len(scale_factor_list), len(df_focus)))

    # df_labels = pd.DataFrame()

    # only sf loop
    for sf_id, sf in enumerate(scale_factor_list[:]):
        # Set current scale factor
        scale_factors = {'vel': {'features': ['v_a_lsr', 'v_d_lsr'], 'factor': sf}}
        clusterer_coarse.set_scaling_factors(scale_factors)
        print(f"Performing clustering for scale factor {clusterer_coarse.scale_factors}...")

        # Fit
        clusterer_coarse.fit(alpha=alpha, knn=knn, bh_correction=True)
        label_array = clusterer_coarse.labels_

        # raw labels
        labels_real = LabelEncoder().fit_transform(label_array)
        label_matrix_coarse[sf_id, :] = labels_real
        # df_labels[f"sf_{sf_id}"] = labels_real

        # density
        rho = clusterer_coarse.weights_
        rho_sum += rho

    combined_pred = get_segments(df_focus, sigma_kwargs["cluster_features"], label_matrix_coarse)
    print(np.unique(combined_pred, return_counts=True))
    # re-merge smallest regions if their size is smaller than KNN value
    df_save = merge_subsets(df_focus, combined_pred, knn)

    # 5.) Output
    # ---------------------------------------------------------
    # output path extension for runs
    run = f"Orion_Taurus_box_KNN_{knn}_2sf"

    if not os.path.exists(output_path + f"Run_{run}/"):
        os.makedirs(output_path + f"Run_{run}/")

    result_path = output_path + f"Run_{run}/"

    # plotting
    # plot(labels=df_save["labels"], df=df_save, filename=f"RF_run_{run}",
    #      output_pathname=result_path)

    # save dataframe for subsequent runs on the separate regions
    df_save.to_csv(result_path + f"RF_run_{run}.csv")
    # df_labels.to_csv(result_path+f"RF_run_{run}_all_sf_labels.csv")

    # Output log-file
    all_fixed = {"mode": "Cartesian", "alpha": alpha, "beta": beta, "knn_initcluster_graph": knn_initcluster_graph,
                 "KNN_list": KNN_list, "bh_correction": bh_correction, "sfs_list": scale_factor_list,
                 "scaling": scaling}

    filename = result_path + f"Run_{run}_parameters.txt"
    with open(filename, 'w') as file:
        for key, value in all_fixed.items():
            file.write(f"{key} = {value}\n")

print("--------- Routine executed successfully ---------")
