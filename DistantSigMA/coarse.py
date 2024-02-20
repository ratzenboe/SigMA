from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ConcensusClustering.majority_voting import match_labels_one2one

from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.stats import mode
import numpy as np
import pandas as pd

#20-2-24

def train_forests(X, Y):
    """
    Training function for the random forest classifier.

    :param X: 5D phase space columns
    :param Y: label array of found clusters
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, Y_train)
    # just for the test score
    y_pred_test = clf.predict(X_test)
    # to get labels for the entire region with each RF
    y_pred = clf.predict(X)
    print("Accuracy of the model:", metrics.accuracy_score(Y_test, y_pred_test))

    return y_pred


def get_segments(df_fit, five_d_cols, label_matrix):
    """
    Function that retrieves the segments that the random forest classifier created.

    :param df_fit: Input dataframe
    :param five_d_cols: phase-space (ICRS or Cartesian)
    :param label_matrix: matrix holding the labels of the different runs
    :return: input dataframe with an extra column called "region" that represents the coarse segments
    """

    X = df_fit[five_d_cols]
    # first bring all labels to the same areas
    Y_data = [rewrite_labels(label_matrix[0, :], label_matrix[sf, :]) for sf in range(label_matrix.shape[0])]
    # calculate the predictions for all Y-entries
    y_arr = [train_forests(X, y) for y in Y_data]
    # Calculate the majority-voted predictions
    predictions = np.vstack(y_arr)
    majority_voted_predictions, _ = mode(predictions, keepdims=True)
    combined_predictions = np.squeeze(majority_voted_predictions)

    return combined_predictions


def rewrite_labels(ref, to_change):
    """
    Function that brings cluster labels into a common reference frame based on their Jaccard-similarity.

    :param ref: Reference label array (smartest choice is to take the one with the most found clusters.
    :param to_change: Label array to calibrate to the reference
    :return: changed label array
    """
    unordered_dict = match_labels_one2one(ref, to_change)
    sorted_items = sorted(unordered_dict.items(), key=lambda x: x[0])
    sorted_dict = dict(sorted_items)

    # new_labels = to_change.replace(sorted_dict).to_numpy()
    # Create a mask for elements to be replaced
    mask = np.isin(to_change, list(sorted_dict.keys()))

    # Replace values in the array using the sorted dictionary
    to_change[mask] = [sorted_dict[val] for val in to_change[mask]]
    return to_change


def merge_subsets(df, chunk_labels, min_entries):

    df_copy = df.copy()
    df_copy['labels'] = chunk_labels
    # Group the DataFrame by the labels
    grouped = df_copy.groupby('labels')

    # Iterate over each group
    for label, group in grouped:
        # Check if the group has fewer entries than min_entries
        if len(group) < min_entries:
            # Fit KMeans to find the center of the group
            kmeans = KMeans(n_clusters=1).fit(group[['X', 'Y', 'Z']])
            center = kmeans.cluster_centers_[0]

            # Calculate distance to centers of other groups
            # Exclude the center of the current group from the centers list
            other_centers = [kmeans.cluster_centers_[0] for kmeans in
                             [KMeans(n_clusters=1).fit(g[['X', 'Y', 'Z']]) for _, g in grouped if not g.equals(group)]]

            # Calculate distances to other centers
            distances = [np.linalg.norm(center - c) for c in other_centers]

            # Find the nearest center
            nearest_idx = np.argmin(distances)
            nearest_label = list(grouped.groups.keys())[nearest_idx]

            # Update the label of the group with too few entries
            df_copy.loc[df_copy['labels'] == label, 'labels'] = nearest_label

    return df_copy
