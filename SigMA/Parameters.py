from SigMA.GraphSkeleton import GraphSkeleton
from collections import defaultdict
from scipy.stats import norm, chi2
from math import erf, sqrt
from numba import jit
import numpy as np
import copy


def global_pvalue_hmp(pvalue_list):
    """Implements the The harmonic mean p-value (HMP). It is a method for performing a combined test
    of the null hypothesis that no p-value is significant (Wilson, 2019).
    It is more robust to dependence between p-values than Fisher’s (1934) method, making it more broadly applicable
    :param pvalue_list: List of p-values from individual tests
    :param weights: weight list with length of pvalue_list and must sum to 1
    """
    # Some checks on the passed weights
    # if isinstance(weights, (list, np.ndarray)):
    #     if (len(weights) != len(pvalue_list)) or (np.abs(1-np.sum(weights) > 1e-5)):
    #         weights = np.ones(len(pvalue_list), dtype=np.float) / len(pvalue_list)
    # else:
    #     weights = np.ones(len(pvalue_list), dtype=np.float) / len(pvalue_list)

    weights = np.ones(len(pvalue_list)) / len(pvalue_list)
    # Convert to numpy (just to be sure)
    pvalue_list = np.asarray(pvalue_list)
    # Compute HMP
    pvalue_hmp = np.sum(weights) / np.sum(weights/pvalue_list)
    return pvalue_hmp


def global_pvalue_fisher(pvalue_list):
    """Fisher’s combination test. If the p_i are uniformly distributed in [0,1],
    then the negative logarithm follows an exponential distribution: -log pi ~ Exp(1).
     The test statistic  t  then follows a 2 distribution with 2n degrees of freedom:
        t = -2 in log pi  ~ chi2_n2
    The global hypothesis test is performed by evaluating if  t  is exceedingly large.
    """
    pvalue_list = np.array(pvalue_list)
    result = np.where(pvalue_list > 0.0000000001, pvalue_list, -23.0259)
    result = np.log(result, out=result, where=result > 0)

    t = -2*np.sum(result)
    degrees_of_freedom = 2 * pvalue_list.size
    return chi2.sf(t, degrees_of_freedom)


def global_pvalue_bonferroni(pvalue_list):
    """Implements the Bonferroni correction"""
    return min(pvalue_list)*len(pvalue_list)


@jit(nopython=True)
def phi(x):
    """Cumulative distribution function for the standard normal distribution
    Can be jit-ed compared to scipy.norm.cdf
    """
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


@jit(nopython=True)
def uf_find(i, parents):
    """
    Find function for Union-Find data structure.

    Parameters:
        i (int): ID 2of point for which parent is required.
        parents (numpy array of shape (num_points)): array storing parents of each point.
    """
    if parents[i] == i:
        return i
    else:
        return uf_find(parents[i], parents)


@jit(nopython=True)
def uf_union(i, j, parents, f):
    """
    Union function for Union-Find data structure. Peak of smaller function value is attached to peak of larger function value.

    Parameters:
        i (int): ID of first point to be merged.
        j (int): ID of second point to be merged.
        parents (numpy array of shape (num_points)): array storing parents of each point.
        f (numpy array of shape (num_points)): array storing function values of each point.
    """
    if f[i] > f[j]:
        parents[j] = i
    else:
        parents[i] = j


@jit(nopython=True)
def loop_data(threshold, num_pts, p, knn, distances, weights_, A_indices, A_idx_ptr):
    parents = -np.ones(num_pts, dtype=np.int32)
    # Prepare density: sort indices by density
    sorted_idxs = np.flip(np.argsort(weights_))
    inv_sorted_idxs = np.arange(num_pts)  # Mapping from index to density rank; lower index -> higher density
    for i in range(num_pts):
        inv_sorted_idxs[sorted_idxs[i]] = i
    # Store persistence and saddle point
    persistence = []
    saddle_points = []
    # Start Mode finding
    for i in range(num_pts):
        current_pt = sorted_idxs[i]  # loop through points in decreasing density fashion
        neighbors = A_indices[A_idx_ptr[current_pt]:A_idx_ptr[current_pt + 1]]
        # Neighbors with higher density:
        higher_neighbors = [n for n in neighbors if inv_sorted_idxs[n] <= i]
        if len(higher_neighbors) == 0:
            # point is new peak
            parents[current_pt] = current_pt  # in this case, a point is it's own parent
        else:  # if a point has neighbors with higher densities
            # attribute point to neighbor with highest density (weight_ is inversly proportional to k-distance)
            g = higher_neighbors[np.argmax(weights_[np.array(higher_neighbors)])]  # highest density neighbor
            pg = uf_find(g, parents)  # parent = respective modal point of cluster
            parents[current_pt] = pg  # add mode point to points info

            for neighbor in higher_neighbors:
                pn = uf_find(neighbor, parents)
                if pg != pn:
                    # ---------------- Store saddle point ----------------
                    # TODO: only save top N candidates already here (reduces memory load)
                    saddle_points.append((pg, pn, current_pt, neighbor))
                    # --------- Compute p-value & test merge condition ----------
                    # Calculate saddle point density
                    val = max(distances[pg], distances[pn])
                    # p * np.sqrt(k / 2) * (np.log(d_saddle) - np.log(d_max))
                    pers_curr = p * np.sqrt(knn / 2) * (np.log(distances[current_pt]) - np.log(val))
                    if pers_curr < threshold:
                        persistence.append(pers_curr)
                        uf_union(pg, pn, parents, weights_)

    labels = np.array([uf_find(n, parents) for n in range(num_pts)])
    return labels, persistence, saddle_points


class ParameterClass(GraphSkeleton):
    def __init__(self, **kwargs):
        """
        :param max_neighbors: Maximal number of k to consider for density estimation
        """
        super().__init__(**kwargs)
        self.knn = None             # Density estimation parameter
        self.dists_to_knn = None    # k-distance; used in p-value computation
        self.weights_ = None        # Estimated density value (not normalized)
        self.n_leaves_ = None       # Number of modes founds in gradient ascend
        self.leaf_labels_ = None    # Mode labels
        self.saddle_dknn = defaultdict(frozenset)   # Saddle point information
        self.mode_kdist = None                      # K-distance of modes (need for resampling)

    def initialize_clustering(self, knn: int, saddle_point_candidate_threshold: int = 20):
        # Fill information
        self.knn = knn
        # As alpha is 1, all initial clusters are kept separate, i.e., all saddle points are recovered
        # Most notably, all merge decisions also of the same two current point and parents are recorded
        # This makes the saddle point determination exact as all decisions can be compared later
        _, labels, n_leaves_, saddle_points = self.gradient_ascend_fast(knn, alpha=1)
        # Determine saddle point
        self.saddle_dknn = self.determine_saddle_points(saddle_points, knn, saddle_point_candidate_threshold)
        self.leaf_labels_ = labels
        self.n_leaves_ = n_leaves_
        self.mode_kdist = {m: [self.dists_to_knn[m]] for m in np.unique(self.leaf_labels_)}
        return

    @staticmethod
    def postprocessing_saddle_points(saddle_points, threshold: int = 20):
        """ Counts the saddle point candidates --> shouldn't be more than threshold
        see SigMA-UpperSco-update-fittingprocedure.ipynb for more details
        :param saddle_points: All saddle point candidates
        :param threshold: Number of saddle point candidates we save;
            the true saddle point is in 99.4% of all cases within the first 20 candidates
            this means we don't have to store over 90% of unlikely saddle point candidates
        """
        saddle_point_modes = defaultdict(frozenset)
        # Reduced saddle point storage
        saddle_points_reduced = []
        for pg, pn, current_pt, neighbor in saddle_points:
            # save memory space --> only save the top "threshold" combinations of pg & pn
            # the "true" saddle point is likely within these top densest points
            if frozenset({pg, pn}) not in saddle_point_modes:
                saddle_point_modes[frozenset({pg, pn})] = 0
            # If the saddle point has only been seen less or equal to threshold times, then we save it
            if saddle_point_modes[frozenset({pg, pn})] <= threshold:
                saddle_points_reduced.append((pg, pn, current_pt, neighbor))
            # Increment counter
            saddle_point_modes[frozenset({pg, pn})] += 1
        return saddle_points_reduced

    def determine_saddle_points(self, saddle_point_candidates, knn, saddle_point_candidate_threshold):
        # Remove unlikely saddle point candidates
        saddle_point_candidates = self.postprocessing_saddle_points(
            saddle_points=saddle_point_candidates,
            threshold=saddle_point_candidate_threshold
        )
        pg, pn, current_pt, neighbor = np.array(saddle_point_candidates).T
        # Get midpoints of connecting edge
        midpts = (self.X[current_pt] + self.X[neighbor]) * 0.5
        # Get distances to midpoints
        dists_midpoints, _ = self.kd_tree.query(midpts, k=knn + 1, n_jobs=-1)
        dknn_midpoint = np.max(dists_midpoints, axis=1)
        # Distances to knn of current_pt and neighor
        currpt_dists = self.dists_to_knn[current_pt]
        neigbr_dists = self.dists_to_knn[neighbor]
        # Decision for saddle point between midpoint, and both edges -->
        saddle_currpt_neighb_arr = np.vstack([dknn_midpoint, currpt_dists, neigbr_dists]).T
        dist_argmax = np.argmax(saddle_currpt_neighb_arr, axis=1)  # maximum distance == minimum density
        # Determine saddle point
        save_dict = defaultdict(frozenset)
        for i, (pg_i, pn_i, argmax_dist_i) in enumerate(zip(pg, pn, dist_argmax)):
            # Saddle point density
            curr_saddle_kdist = saddle_currpt_neighb_arr[i, argmax_dist_i]
            # If entry does not exist, create it
            if frozenset({pg_i, pn_i}) not in save_dict:
                if argmax_dist_i == 0:
                    # Then the mid point of the edge has the minimal density
                    saddle_point_position = midpts[i, :]
                elif argmax_dist_i == 1:
                    saddle_point_position = self.X[current_pt[i]]
                else:
                    saddle_point_position = self.X[current_pt[i]]
                # Saves information on the k-distance and the position of the saddle point
                save_dict[frozenset({pg_i, pn_i})] = (curr_saddle_kdist, saddle_point_position)
            else:
                # Entry already exists: we need to compare the densities of the given saddle points
                # Saddle point: Maximum density point of all these saddle candidates, i.e., minimum k-distance
                if curr_saddle_kdist < save_dict[frozenset({pg_i, pn_i})][0]:
                    # Get saddle point position
                    if argmax_dist_i == 0:
                        saddle_point_position = midpts[i, :]
                    elif argmax_dist_i == 1:
                        saddle_point_position = self.X[current_pt[i]]
                    else:
                        saddle_point_position = self.X[current_pt[i]]
                    # Save "better" saddle point candidate
                    save_dict[frozenset({pg_i, pn_i})] = (curr_saddle_kdist, saddle_point_position)
        # Sort dictionary by density of saddle point --> from densest (minimum distance) to the least dense one
        sd_aslist = [(pg, pn, saddle_kdist, saddle_pos) for (pg, pn), (saddle_kdist, saddle_pos) in save_dict.items()]
        sd_aslist = sorted(sd_aslist, key=lambda x: x[2])
        # Transform list back to dictionary
        save_dict = defaultdict(frozenset)
        for pg, pn, saddle_kdist, saddle_pos in sd_aslist:
            # The saddle point k-distances are saved in a list to allow for easy resampling extension
            save_dict[frozenset({pg, pn})] = ([saddle_kdist], saddle_pos)
        return save_dict

    def gradient_ascend_fast(self, knn: int, alpha: float):
        """
        :param knn: knn denisty estimation parameter
        :param alpha: significance level
        """
        print(f'Performing gradient ascend using a {knn}-NN density estimation.')
        # --- Start fitting proceduce ---
        num_pts, p = self.X.shape
        # Calculate density
        self.dists_to_knn = self.k_distance(k=knn)
        self.weights_ = self.knn_density(knn)
        # Loop through data set
        threshold = norm.ppf(1 - alpha, 0, 1)
        labels, persistence, saddle_points = loop_data(
            threshold,
            num_pts,
            p,
            knn,
            self.dists_to_knn,
            self.weights_,
            self.A.indices,
            self.A.indptr
        )
        # post-process label information
        # labels_ = LabelEncoder().fit_transform(labels)
        n_leaves_ = np.unique(labels).size
        return persistence, labels, n_leaves_, saddle_points

    def saddle_point_positions(self):
        if len(self.saddle_dknn) == 0:
            return np.empty(shape=(0, self.X.shape[-1]))
        # Extract saddle points
        all_saddle_points = []
        for _, saddle_position in self.saddle_dknn.values():
            all_saddle_points.append(saddle_position)
        return np.array(all_saddle_points)

    def merge_clusters(self, knn, alpha):
        # prepare parameters for fitting
        parents = copy.deepcopy(self.leaf_labels_)  # parents get modified --> need a copy
        densities = self.knn_density(knn)
        # threshold = norm.ppf(1 - alpha, 0, 1)
        num_pts, p = self.X.shape
        pvalues = []
        # loop through saddle points
        for (g, n), (saddle_kdist_list, _) in self.saddle_dknn.items():
            # Check if regions g & n have already been merged
            pg = uf_find(g, parents)
            pn = uf_find(n, parents)
            if pg != pn:
                # --------- Extract list of k-distances of modes and saddle points --------
                pg_kdist_list = self.mode_kdist[pg]
                pn_kdist_list = self.mode_kdist[pn]
                p_values = []   # Store p values of individual tests
                # TODO: vectorize following loop
                for pg_kdist, pn_kdist, saddle_kdist in zip(pg_kdist_list, pn_kdist_list, saddle_kdist_list):
                    val = max(pg_kdist, pn_kdist)
                    # p * np.sqrt(k / 2) * (np.log(d_saddle) - np.log(d_max))
                    SB_alpha = p * np.sqrt(knn / 2) * (np.log(saddle_kdist) - np.log(val))
                    # pval_curr = 1 - norm.cdf(SB_alpha)
                    pval_curr = 1 - phi(SB_alpha)
                    p_values.append(pval_curr)
                pval_final = global_pvalue_hmp(p_values)
                # pval_final = global_pvalue_bonferroni(p_values)
                # pval_final = global_pvalue_fisher(p_values)
                # Check if should merge or not
                if pval_final > alpha:
                    uf_union(pg, pn, parents, densities)

        labels = np.array([uf_find(n, parents) for n in range(num_pts)])
        return labels
