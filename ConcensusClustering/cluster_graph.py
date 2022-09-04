import copy
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict
from ConcensusClustering.majority_voting import compute_jaccard_matrix
from scipy.stats import mode


class ClusterEnsemble:
    def __init__(self, *labels: np.ndarray):
        # --- labels are stored as a (k,n) np.ndarray.  ---
        #    k...number of clustering solutions; n...number of points
        if len(labels) == 1:
            self.labels = labels[0]
        else:
            self.labels = np.vstack([l for l in labels])
        # Labels transformed into unique indices: prevent overlaps between labels
        self.labels_transformed = None
        self.unique_labels_transformed = []
        self.labels_bool_dict2arr = defaultdict(np.ndarray)
        # Transformed labels that are -1 (i.e., noise) in original cluster labels
        self.noise_labels = []
        self.transform_labels()
        # --- Graph connecting clustering solutions via jaccard similarity ---
        self.G = nx.Graph()
        # --- Minimum spanning tree
        self.T = None

    def init_variables(self):
        # --- store noise as 0 --> needed for indexing adjacency matrix
        self.labels_transformed = copy.deepcopy(self.labels) + 1
        self.unique_labels_transformed = []
        # --- Transformed labels that are -1 (i.e., noise) in original cluster labels
        self.noise_labels = []
        return

    def transform_labels(self):
        # --- initialize member variables
        self.init_variables()
        # --- Add infos about first (unmodified) clustering solution
        self.unique_labels_transformed.append(np.unique(self.labels_transformed[0]))
        self.noise_labels.append(np.min(self.labels_transformed[0]))
        # --- Loop through other solutions
        for i in range(1, self.labels.shape[0]):
            # --- Shift labels[i] by max of labels[i-1]
            ul_im1_max = self.unique_labels_transformed[i - 1].max()
            ul_i_min = self.labels_transformed[i].min()
            self.labels_transformed[i] += (ul_im1_max - ul_i_min + 1)
            # --- Add infos about i'th clustering solution
            self.unique_labels_transformed.append(np.unique(self.labels_transformed[i]))
            self.noise_labels.append(np.min(self.labels_transformed[i]))
        # --- Create a mapping between each unique label in labels_transformed (independent of row) and the clustered points
        self.create_bool_labels()
        return

    def create_bool_labels(self):
        for labels_trafo_i, u_l_i in zip(self.labels_transformed, self.unique_labels_transformed):
            # --- Loop through individual unique labels
            for u_j in u_l_i:
                self.labels_bool_dict2arr[u_j] = labels_trafo_i == u_j
        return

    def add_edges_weights(self, G, i, j):
        # --- compute jaccard similarity between all pairwise clusters of solutions i and j
        jm = compute_jaccard_matrix(self.labels_transformed[i], self.labels_transformed[j])
        edges = np.vstack([np.nonzero(jm)]).T
        # --- Add edges to graph
        for e1, e2 in edges:
            # --- corresponding cluster labels to indices e1, e2
            c1 = self.unique_labels_transformed[i][e1]
            c2 = self.unique_labels_transformed[j][e2]
            # --- don't add noise clusters to graph
            if (c1 not in self.noise_labels) and (c2 not in self.noise_labels):
                # --- Store distance instead of similarity --> need small distances to hold up for MST
                js_minor = self.jaccard_similarity_minor(i=c1, j=c2)
                G.add_edge(c1, c2, distance=1 - jm[e1, e2], similarity=jm[e1, e2], similarity_minor=js_minor)
        return

    def build_graph(self):
        # --- Init graph
        self.G = nx.Graph()
        # --- loop through combinations of solutions
        nb_solutions_iterator = range(self.labels.shape[0])
        for i, j in product(nb_solutions_iterator, nb_solutions_iterator):
            if i != j:
                self.add_edges_weights(G=self.G, i=i, j=j)
        return

    def remove_edges(self, density):
        G_copy = nx.Graph(self.G)
        # --- test if we need to remove edges
        e2js = {frozenset({e1, e2}): G_copy[e1][e2]['similarity'] for e1, e2 in G_copy.edges}
        # --- remove edges with unmatching cluster solutions
        remove_edges = []
        for (e1, e2), js in e2js.items():
            if js < 0.5:
                # --- test if overlap of modes exist
                shares_mode = self.share_mode(e1, e2, density)
                js_minor = self.jaccard_similarity_minor(e1, e2)
                if not (shares_mode and js_minor >= 0.5):
                    # if not shares_mode:
                    remove_edges.append((e1, e2))
        G_copy.remove_edges_from(remove_edges)
        return G_copy

    def affinity_matrix(self):
        """In affinity matrix larger values indicate greater similarity between instances.
        Thus, here we save the similarity or (1 - self.G[i][j])
        """
        # Each cluster gets a row in the  == number of nodes in graph
        nb_clusters = max(list(self.G.nodes)) + 1
        A = np.eye(N=nb_clusters)
        for e1, e2 in self.G.edges:
            similarity = self.G[e1][e2]['similarity']
            A[e1, e2] = similarity
            A[e2, e1] = similarity
        return A

    def share_mode(self, i, j, density):
        """Test if clusters i & j share the modal point"""
        mode_i = np.max(density[self.labels_bool_dict2arr[i]])
        mode_j = np.max(density[self.labels_bool_dict2arr[j]])
        return np.isclose(mode_i, mode_j, 1e-5)

    def jaccard_similarity_minor(self, i, j):
        nb_i = self.labels_bool_dict2arr[i].sum()
        nb_j = self.labels_bool_dict2arr[j].sum()
        intersection = np.sum(
            (self.labels_bool_dict2arr[i].astype(int) + self.labels_bool_dict2arr[j].astype(int)) == 2)
        return intersection / min(nb_i, nb_j)

    def jaccard_similarity(self, bool_arr, j):
        if isinstance(bool_arr, int):
            zero_one_arr_i = self.labels_bool_dict2arr[bool_arr].astype(int)
        else:
            zero_one_arr_i = bool_arr.astype(int)

        zero_one_arr_j = self.labels_bool_dict2arr[j].astype(int)
        added_arr = zero_one_arr_i + zero_one_arr_j
        intersection = np.sum(added_arr == 2)
        union = np.sum(added_arr > 0)
        return intersection / union

    def find_cliques_subgraph(self, density):
        # --- Get graph with "bad" edges removed
        H = self.remove_edges(density)
        cliques_extracted = []
        # --- loop through larges cliques until we exhaust all nodes
        while len(H) > 0:
            # --- find "best clique", i.e. largest and with highest self similarity score
            cliques = list(nx.find_cliques(H))
            # --- Maximize the minor jaccard distance --> promotes merging of clusters with subcluster
            best_clique = np.argmax(
                [np.sum([[H[i][j]['similarity_minor'] for i, j in product(c, c) if i != j]]) for c in cliques])
            bc = cliques[best_clique]
            cliques_extracted.append(bc)
            # --- remove from community graph
            H.remove_nodes_from(bc)
        return cliques_extracted

    def noise_cluster_removal(self, labels_final, mode_count, min_cluster_size, min_overlaps, js_sig_min, min_occurance=2):
        # --- Input clusters without noise part
        notnoise = [l for l in self.labels_bool_dict2arr.keys() if l not in self.noise_labels]
        # --- Remove noise clusters based on 2 conditions
        for uid in np.unique(labels_final):
            # --- 1) if cluster has less than min_cluster_size "robust points" (appear at least twice)
            unique_labels, counts = np.unique(mode_count[labels_final == uid], return_counts=True)
            # --- 2) clusters need to appear like this "similarily" in the data set, i.e.,
            # ---    consensus clusters need a minimum jaccard similarity of 0.5 with clusters in labels
            js_sig = 0
            for nnl in notnoise:
                js = self.jaccard_similarity(labels_final == uid, nnl)
                if js > 0.5:
                    js_sig += 1
            # --- Remove cluster if any of the two conditions trigger:
            # --- 1) At least three stable cluster solutions
            # --- 2) At least three stable overlaps with existing cluster solutions
            if (counts[unique_labels >= min_overlaps].sum() < min_cluster_size) or (js_sig < js_sig_min):
                labels_final[labels_final == uid] = -1

        # --- Remove unstable points that only appear once in cluster extraction
        labels_final[mode_count < min_occurance] = -1
        return labels_final

    def fit(self, density, min_cluster_size=15, min_overlaps=2, js_sig_min=2, min_occurance=2):
        # --- Partition graph into cliques
        cluster_candidate_groups = self.find_cliques_subgraph(density=density)
        nb_unique_labels = np.unique([elem for ccg in cluster_candidate_groups for elem in ccg]).size
        voting_arr = np.full(shape=(nb_unique_labels, self.labels.shape[1]), fill_value=np.nan)
        # --- Each cluster in clique is labeled the same
        i = 0
        for group_id, cc in enumerate(cluster_candidate_groups):
            for cluster_i in cc:
                voting_arr[i][self.labels_bool_dict2arr[cluster_i]] = group_id
                i += 1
        # Vote on which points belong to which cluster
        labels_final, mode_count = mode(voting_arr, nan_policy='omit', axis=0)
        labels_final = np.array(labels_final.flatten().astype(int))
        mode_count = np.array(mode_count.flatten().astype(int))
        labels_final = self.noise_cluster_removal(
            labels_final, mode_count, min_cluster_size, min_overlaps, js_sig_min, min_occurance
        )
        return labels_final
