from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
import numpy as np
import copy


class FinalizeLables:
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels
        self.new_stars = None

    def relocate_labels(self, *cluster_ids, **kwargs):
        # Get data without these clusters
        labels2relocate = np.array([l for l in cluster_ids])
        lmask = np.isin(self.labels, labels2relocate)
        # Remove cluster id's to relocate from labels
        new_labels = copy.deepcopy(self.labels)
        new_labels[lmask] = -1
        X_wo = self.X[new_labels > -1]
        # Fit knn model
        n_jobs = kwargs.pop('n_jobs', -1)
        model = KNeighborsClassifier(n_jobs=n_jobs, **kwargs).fit(X=X_wo, y=new_labels[new_labels > -1])
        new_labels[lmask] = model.predict(self.X[lmask])
        return new_labels

    def update_labeling(self, *cluster_ids, **kwargs):
        # Remove cluster id's to relocate from labels
        new_labels = copy.deepcopy(self.labels)
        is_outlier = np.zeros_like(self.labels, dtype=bool)
        is_inlier = np.ones_like(self.labels, dtype=int)
        is_inlier[self.labels == -1] = -1
        # Get data without these clusters
        for l in cluster_ids:
            is_inlier[self.labels == l] = LocalOutlierFactor(**kwargs).fit_predict(self.X[self.labels == l])
            new_labels[(self.labels == l) & (is_inlier == -1)] = -1
            is_outlier[(self.labels == l) & (is_inlier == -1)] = True

        # Fit knn model
        X_wo = self.X[new_labels > -1]
        n_jobs = kwargs.pop('n_jobs', -1)
        model = KNeighborsClassifier(n_neighbors=5).fit(X=X_wo, y=new_labels[new_labels > -1])
        new_labels[is_outlier] = model.predict(self.X[is_outlier])
        return new_labels

    def preprocess_labels(self, *labels_other):
        if len(labels_other) == 1:
            labels_other = labels_other[0]
        else:
            labels_other = np.vstack([l for l in labels_other])
        # Find clustered stars
        in_cluster = np.sum(labels_other, axis=0) != -labels_other.shape[0]
        return in_cluster

    def find_novel_stars(self, *labels_other, **isolation_kwargs):
        self.new_stars = np.zeros(self.labels.size, dtype=bool)
        in_cluster = self.preprocess_labels(*labels_other)
        # Get defaults for relevant parameters
        n_jobs = isolation_kwargs.pop('n_jobs', -1)
        contamination = isolation_kwargs.pop('contamination', 0.1)
        # Fit model
        model = IsolationForest(n_jobs=n_jobs, contamination=contamination, **isolation_kwargs)
        model.fit(self.X[self.labels > -1])
        self.new_stars[in_cluster] |= model.predict(self.X[in_cluster]) == 1
        # only want stars not already assigned to clusters
        self.new_stars[self.labels > -1] = False
        return self.new_stars

    def fit(self, save_labels=False, **kwargs):
        new_labels = copy.deepcopy(self.labels)
        n_jobs = kwargs.pop('n_jobs', -1)
        model = KNeighborsClassifier(n_jobs=n_jobs, **kwargs).fit(X=self.X[self.labels > -1], y=self.labels[self.labels > -1])
        new_assignment = model.predict(self.X[self.new_stars])
        new_labels[self.new_stars] = new_assignment
        if save_labels:
            self.labels = new_labels
        return new_labels
