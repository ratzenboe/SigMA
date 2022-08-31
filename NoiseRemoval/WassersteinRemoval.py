from scipy.sparse import csr_matrix
from scipy.stats import wasserstein_distance
from Graph.GabrielGraph import gabriel_graph_adjacency




#ajm = gabriel_graph_adjacency(data)
#data_ws = [wasserstein_distance(dist[row], dist[col]) for row, col in np.vstack(ajm.nonzero()).T]
#ajm_ws = csr_matrix((data_ws, ajm.nonzero()), shape=ajm.shape)
#ajm_ws_cp = copy.deepcopy(ajm_ws)
#ajm_ws_cp[ajm_ws_cp>th] = 0   # eliminate large distanes
#ajm_ws_cp.eliminate_zeros()
#_, cc_idx = connected_components(ajm_ws_cp)
# # Remove dense points not connected to the main cluster
#cluster_labels = np.copy(cut_filter)
#cluster_labels[np.arange(cluster_labels.size)[cluster_labels][cc_idx!=np.argmax(np.bincount(cc_idx))]] = False
#print(np.sum(cluster_labels))