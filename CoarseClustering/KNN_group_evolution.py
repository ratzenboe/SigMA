import datetime
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from Loop_functions import setup_phase_space, remove_field_stars, consensus_function
from SigMA.SigMA import SigMA

from DistantSigMA.IsochroneArchive.myTools import my_utility

# ---------------------------------------------------------

# set sys and output paths
sys.path.append('/Users/alena/PycharmProjects/Sigma_Orion')
output_path = my_utility.set_output_path(main_path='/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/'
                                                   'Projects/Sigma_Orion/Coding/Code_output/')
output_path = output_path + "KNN_group_evolution/"
result_file = output_path + f"groups_coarse_robust.csv"
my_utility.setup_HP(result_file, name_string="id,knn,sf,n_groups")

# set fixed cluster features
error_features = ['ra', 'dec', 'parallax', 'pmra', 'pmdec',
                  'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']

correlations_features = ['ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
                         'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
                         'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']

add_features = ['radial_velocity', 'radial_velocity_error']

# Define all FIXED SigMA parameters
beta = 0.99
knn_initcluster_graph = 35
KNN_list = range(100, 300, 1)
fixed_params = {"beta": beta, "knn_initcluster_graph": knn_initcluster_graph, "KNN_list": KNN_list}

# coarse clustering
alpha = 0.01

# ---------------------------------------------------------

# load the dataframe
df_focus = pd.read_csv('../Data/Gaia/data_orion_focus.csv')

# ---------------------------------------------------------

# setup kwargs
setup_kwargs = setup_phase_space(df_fit=df_focus, coord_sys="Cartesian", **fixed_params)
sigma_kwargs = setup_kwargs["sigma_kwargs"]
scale_factor_list = setup_kwargs["scale_factor_list"]

# ---------------------------------------------------------

# initialize SigMA with sf_mean
clusterer_coarse = SigMA(data=df_focus, **sigma_kwargs)

# initialize the density sum
rho_sum = np.zeros(df_focus.shape[0], dtype=np.float32)

# initialize the label matrix for the different scaling factors
label_matrix_raw = np.empty(shape=(len(scale_factor_list), len(df_focus)))
label_matrix_coarse = np.empty(shape=(len(scale_factor_list), len(df_focus)))

n_group_list = []

for knn in KNN_list:
    # only sf loop
    for sf_id, sf in enumerate(scale_factor_list[:]):
        # Set current scale factor
        clusterer_coarse.set_scaling_factors(setup_kwargs["scale_factors"])
        print(f"Performing clustering for scale factor {sf}...")
        # Fit
        st = time.time()
        label_array = clusterer_coarse.fit(alpha=alpha, knn=knn, bh_correction=True)
        delta_t = str(datetime.timedelta(seconds=time.time() - st)).split('.')[0]
        print(f'Done! [took {delta_t}]. Found {np.unique(label_array).size} clusters')

        labels_real = LabelEncoder().fit_transform(label_array)
        n_real = len(np.unique(labels_real))
        label_matrix_raw[sf_id, :] = labels_real

        # density
        rho = clusterer_coarse.weights_
        rho_sum += rho

        # remove field stars
        nb_rfs = remove_field_stars(label_array, rho, label_matrix_coarse, sf_id)

    # Perform consensus clustering and plot the result
    labels_cc, n_cc = consensus_function(label_matrix_coarse, rho_sum, df_focus,
                                         f"pre-Step3_KNN_{knn}_CC_a_p01",
                                         output_path)

    n_group_list.append(n_cc)
    print(f"KNN {knn}: {n_cc} groups")

# Plot the evolution

# 5p median
avg_list = [np.median(n_group_list[i:i + 5]) for i in range(0, len(n_group_list), 5)]
avg_knn = [np.median(KNN_list[i:i + 5]) for i in range(0, len(n_group_list), 5)]

f = plt.figure()
plt.plot(KNN_list, n_group_list, ls="dotted", label="true")  # , marker  = "o")
plt.plot(avg_knn, avg_list, ls="solid", label="5kmedian")
plt.ylabel("n groups")
plt.xlabel("KNN")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
f.savefig(output_path + "group_trend_KNN_both.png", dpi=400)

# Save the results to a csv

max_length = max(len(KNN_list), len(avg_knn))
avg_knn += [pd.NA] * (max_length - len(avg_knn))
avg_list += [pd.NA] * (max_length - len(avg_list))

result_KNN = "/Users/alena/Library/CloudStorage/OneDrive-Personal/Work/PhD/Projects/Sigma_Orion/Coding/Code_output/" \
             "2023-09-21/KNN_group_evolution/" + "KNN_evo_log.csv"
output_set = {"KNN": list(KNN_list), "n_groups": n_group_list, "5p-median KNN": avg_knn, "5p-median n_groups": avg_list}
output_df = pd.DataFrame(data=output_set)
output_df.to_csv(result_KNN, mode="a", header=True)
