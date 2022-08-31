import numpy as np
import pandas as pd

# Add column to the data set indicating if point is in neighborhood of cluster
def calc_cluster_int(data, unique_neighbors_dict):
    """
    cluster array saving for each star the cluster belonging (since a star might be in the
    neighborhood of multiple clusters it's not 1-1)
    """
    cluster_arr = np.zeros(data.shape[0], dtype=int)
    for cluster_int, star_ids in unique_neighbors_dict.items():
        cluster_arr[data.id.isin(list(star_ids)).values] += 2**cluster_int
    return cluster_arr

# Returns true if a star is a neighbor of a cluster ()
def convert_to_boolarr(int_arr, cluster_id):
    """
    :param int_arr: array of integers which relate to no, one or multiple clusters
    cluster_id: 0=Pleiades, 1=Meingast 1, 2=Hyades, 3=Alpha Per, 4=Coma Ber
    """
    return np.array((np.floor(int_arr/2**cluster_id) % 2), dtype=bool)