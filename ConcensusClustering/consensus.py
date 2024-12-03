import copy
import numpy as np
from scipy.stats import mode
import networkx as nx
from itertools import product
from collections import defaultdict
from ConcensusClustering.majority_voting import compute_jaccard_matrix


class ClusterConsensus:
    def __init__(self, *labels: np.ndarray):
        # --- labels are stored as a (k,n) np.ndarray.  ---
        #    k...number of clustering solutions; n...number of points
        if len(labels) == 1:
            self.labels = labels[0]
        else:
            self.labels = np.vstack([l for l in labels])
        self.indices = np.arange(self.labels.shape[1])
        # Labels transformed into unique indices: prevent overlaps between labels
        self.labels_transformed = None
        self.unique_labels_transformed = []
        self.labels_bool_dict2arr = defaultdict(np.ndarray)
        # Transformed labels that are -1 (i.e., noise) in original cluster labels
        self.noise_labels = []
        self.transform_labels()
        # --- Graph connecting clustering solutions via Jaccard similarity ---
        self.G = nx.Graph()
        self.build_graph()
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
        # --- Create a mapping between each unique label in labels_transformed
        # --- (independent of row) and the clustered points
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
                # --- Store distance, similarity, and minor Jaccard similarity
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

    def remove_edges(self, similarity='similarity', threshold=0.5):
        G_copy = nx.Graph(self.G)
        # --- test if we need to remove edges
        e2js = {frozenset({e1, e2}): G_copy[e1][e2][similarity] for e1, e2 in G_copy.edges}
        # --- remove edges with unmatching cluster solutions
        re = [(e1, e2) for (e1, e2), sim in e2js.items() if sim < threshold]
        G_copy.remove_edges_from(re)
        return G_copy

    def share_mode(self, i, j, density):
        """Test if clusters i & j share the modal point"""
        args_i = self.indices[self.labels_bool_dict2arr[i]]
        args_j = self.indices[self.labels_bool_dict2arr[j]]

        rho_i = density[args_i]
        rho_j = density[args_j]

        if rho_i.size > rho_j.size:
            ind = np.argpartition(rho_i, -rho_j.size)[-rho_j.size:]
            # compute Jaccard distance
            jacc = np.intersect1d(args_j, args_i[ind]).size / np.union1d(args_j, args_i[ind]).size
        else:
            ind = np.argpartition(rho_j, -rho_i.size)[-rho_i.size:]
            # compute Jaccard distance
            jacc = np.intersect1d(args_i, args_j[ind]).size / np.union1d(args_i, args_j[ind]).size

        return jacc > 0.5

    def remove_edges_density(self, density):
        G_copy = nx.Graph(self.G)
        # --- test if we need to remove edges
        e2js = {frozenset({e1, e2}): G_copy[e1][e2]['similarity'] for e1, e2 in G_copy.edges}
        # --- remove edges with unmatching cluster solutions
        remove_edges = []
        for (e1, e2), js in e2js.items():
            if js < 0.5:
                js_minor = self.jaccard_similarity_minor(e1, e2)
                if js_minor > 0.5:
                    # --- test if overlap of modes exist
                    shares_mode = self.share_mode(e1, e2, density)
                    if not shares_mode:
                        remove_edges.append((e1, e2))
                else:
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

    def jaccard_similarity_minor(self, i, j):
        nb_i = self.labels_bool_dict2arr[i].sum()
        nb_j = self.labels_bool_dict2arr[j].sum()
        intersection = np.sum(
            (self.labels_bool_dict2arr[i].astype(int) + self.labels_bool_dict2arr[j].astype(int)) == 2)
        return intersection / min(nb_i, nb_j)

    def fit(self, density, min_cluster_size, similarity_match='similarity', norm=False, aggregation_function=None):
        # Remove bad connections
        H = self.remove_edges_density(density)
        # Get cliques
        cliques = list(nx.find_cliques(H))
        # Represents final voting
        voting_arr = np.zeros(shape=(len(cliques), self.labels.shape[1]), dtype=np.float32)
        # Set-up voting array
        for clique_id, c in enumerate(cliques):
            clique_arr = np.zeros(shape=(self.labels.shape[1],), dtype=np.float32)
            if len(c) > 1:
                for cluster_id in c:
                    clique_arr[self.labels_bool_dict2arr[cluster_id]] += 1
                if norm:
                    clique_arr /= len(c)
            else:
                clique_arr[self.labels_bool_dict2arr[c[0]]] = 1
            # --- We multiply the clique by a factor proportional to its edge weights ---
            # Get unique edges
            clique_edges = {frozenset({c_id, n}): H[c_id][n][similarity_match] for c_id in c for n in H.neighbors(c_id)}
            unique_edge_weights = list(clique_edges.values())
            # Add clique to voting array
            voting_arr[clique_id] = clique_arr
            if aggregation_function is not None:
                if len(c) > 1:
                    voting_arr[clique_id] *= aggregation_function(unique_edge_weights)

        # We remove small cliques/clusters
        # --> This spurious cluster/clique removal produces -1 results where another clique might shine
        # --> we go back to voting arr, remove those cliques and vote again
        mode_decision, _ = mode(self.labels, keepdims=True)
        spurious = [1]
        while len(spurious) > 0:
            try:
                labels_cliques = np.argmax(voting_arr, axis=0)
            except ValueError:
                # We break array here if no -1 samples are found
                spurious = []
                labels_cliques = np.ones(shape=(self.labels.shape[1],), dtype=np.int32) * -1
                break
            # Set bg to -1 (by majority voting)
            if len(mode_decision[0] == -1) > 0:
                labels_cliques[mode_decision[0] == -1] = -1
                # Remove very small clusters
                unique, counts = np.unique(labels_cliques, return_counts=True)
                spurious = unique[counts < min_cluster_size]
                voting_arr = np.delete(voting_arr, spurious, axis=0)
            else:
                # We break array here if no -1 samples are found
                spurious = []
        # Remove spurious clusters
        labels_cliques[np.isin(labels_cliques, spurious)] = -1
        return labels_cliques
