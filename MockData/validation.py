import numpy as np
from ConcensusClustering.majority_voting import get_new_labels
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import accuracy_score, balanced_accuracy_score, adjusted_rand_score, matthews_corrcoef, \
                            adjusted_mutual_info_score, precision_score, recall_score


def validate_clutering(labels_true, labels_pred, verbose=False):
    # Relabel to match original labels
    # ---- Get mapping from predicted labels to true labels
    label_key = get_new_labels(labels_true, labels_pred)
    # ---- Relabel m'th cluster solutions
    labels_pred = np.vectorize(label_key.get)(labels_pred)
    # Focus on clustered samples
    is_bg_true = labels_true > -1
    is_bg_pred = labels_pred > -1
    sig = is_bg_true | is_bg_pred
    # Save validation results
    results = {
        'nmi': nmi(labels_true, labels_pred),
        'adjusted_mutual_info_score': adjusted_mutual_info_score(labels_true, labels_pred),
        'adjusted_rand_score': adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred),
        'acc': accuracy_score(labels_true, labels_pred),
        'precision': precision_score(labels_true, labels_pred, average='weighted'),
        'recall': recall_score(labels_true, labels_pred, average='weighted'),
        'balanced_acc': balanced_accuracy_score(y_true=labels_true, y_pred=labels_pred, adjusted=True),
        'matthews_coef': matthews_corrcoef(y_true=labels_true, y_pred=labels_pred),
    }
    results_sig = {
        'nmi': nmi(labels_true[sig], labels_pred[sig]),
        'adjusted_mutual_info_score': adjusted_mutual_info_score(labels_true[sig], labels_pred[sig]),
        'adjusted_rand_score': adjusted_rand_score(labels_true=labels_true[sig], labels_pred=labels_pred[sig]),
        'acc': accuracy_score(labels_true[sig], labels_pred[sig]),
        'precision': precision_score(labels_true[sig], labels_pred[sig], average='weighted'),
        'recall': recall_score(labels_true[sig], labels_pred[sig], average='weighted'),
        'balanced_acc': balanced_accuracy_score(y_true=labels_true[sig], y_pred=labels_pred[sig], adjusted=True),
        'matthews_coef': matthews_corrcoef(y_true=labels_true[sig], y_pred=labels_pred[sig]),
    }

    # Order by metric score
    results = dict(sorted(results.items(), key=lambda item: item[1]))
    results_sig = dict(sorted(results_sig.items(), key=lambda item: item[1]))
    # Printing:
    if verbose:
        N_true = np.unique(labels_true[labels_true > -1]).size
        N_pred = np.unique(labels_pred[labels_pred > -1]).size
        print(f'------ {N_pred}/{N_true} clusters found ------')
        print('------------- Results scores -------------')
        for metric, score in results.items():
            empty_space = 35 - len(metric)
            print(f'::  {metric}: {empty_space*" "}{score:.3f}')
        print('------------------------------------------')
    if verbose:
        print('------------- Results scores (sig) -------------')
        for metric, score in results_sig.items():
            empty_space = 35 - len(metric)
            print(f'::  {metric}: {empty_space * " "}{score:.3f}')
        print('------------------------------------------------')

    return results, results_sig
