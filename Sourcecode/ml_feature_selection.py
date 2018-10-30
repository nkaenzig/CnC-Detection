"""
Master Thesis
Network Monitoring and Attack Detection

ml_feature_selection.py
Contains functions used in the feature-selection process of the supervised-learning part of the thesis.

@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import RandomizedLogisticRegression, Lasso, Ridge

from scipy.stats import pearsonr
import pandas as pd
from minepy import MINE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import dcor
from collections import defaultdict
import time
import os
from datetime import datetime

import ml_data
import ml_helpers
import ml_training


def plot_feature_self_correlation_matrix(X, feature_names, size, fontsize):
    """
    This function plots a graphical correlation matrix for each pair of columns in X.

    :param X: see above
    :param feature_names: names of the features (columns of X)
    :param size: Figsize parameter for plotting
    :param fontsize: Fontsize parameter for plotting
    """
    df = pd.DataFrame(data=X, columns= feature_names)

    print('Calculating correlation matrix ...')
    corr_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr_matrix, cmap=plt.cm.gray)
    # plt.imshow(corr_matrix, cmap=plt.cm.gray)
    plt.xticks(range(len(feature_names)), feature_names, rotation='vertical', fontsize=fontsize)
    plt.yticks(range(len(feature_names)), feature_names, fontsize=fontsize)
    # plt.gcf().subplots_adjust(left=0.15, top=0.15)
    # plt.tight_layout()

    stamp = str(int(datetime.now().timestamp()))
    plt.savefig('./figures/X_self_corr-{}-features-{}.png'.format(len(feature_names), stamp), bbox_inches="tight")
    plt.savefig('./figures/X_self_corr-{}-features-{}.eps'.format(len(feature_names), stamp), format='eps', bbox_inches="tight")
    plt.show()


def plot_mutual_information_matrix(X, feature_names, size=20):
    """
    Same as plot_feature_self_correlation_matrix() but uses Mutual Information metric
    """
    def calc_MI(x, y, bins):
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    bins = 5  # ?
    n = X.shape[1]
    matMI = np.zeros((n, n))

    for ix in tqdm(np.arange(n), desc='Calculating pairwise mutual information'):
        for jx in np.arange(ix + 1, n):
            matMI[ix, jx] = calc_MI(X[:, ix], X[:, jx], bins)

    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(matMI + matMI.T, cmap=plt.cm.gray)
    plt.xticks(range(len(feature_names)), feature_names, rotation='vertical');
    plt.yticks(range(len(feature_names)), feature_names);

    plt.show()


def MI_vs_Corr_vs_DC(X, Y, feature_names):
    """
    This function calculates the Mutual Information and the Pearson Correlation metrics between the feature vectors
    and the labels of a dataset. Generates a bar plot with the results.

    :param X: Feature matrix (columns are the feature vectors)
    :param Y: Labels
    :param feature_names: Names of the features (corresponding to colums of X)
    """
    def rank_to_dict(ranks, names, scale=True):
        if scale:
            minmax = MinMaxScaler()
            ranks = minmax.fit_transform(np.array([ranks]).T).T[
                0]  # .T to change 1D to 2D array, as fit_transform() doesn't work for 1D
        # ranks = map(lambda x: round(x, 10), ranks)
        return dict(zip(names, ranks))
    def mutual_information_features_vs_labels(X, Y, feature_names):
        mic_scores = []
        m = MINE()
        # subsampling_idx = np.random.choice(X.shape[0], 50000)
        for i in range(X.shape[1]):
            print(' MI {}'.format(i))
            mutual_information = mutual_info_score(X[:, i], Y)
            # m.compute_score(X[subsampling_idx, i], Y[subsampling_idx])
            # mic_scores.append(m.mic())
            mic_scores.append(mutual_information)

        # rank_dict = rank_to_dict(mic_scores, feature_names, scale=False)
        rank_dict = rank_to_dict(mic_scores, feature_names)
        return rank_dict

    def abs_correlation_features_vs_labels(X, Y, feature_names):
        corr_scores = []
        for i in range(X.shape[1]):
            # corr = np.abs(np.correlate(X[:, i], Y))
            corr = np.abs(pearsonr(X[:, i], Y))
            corr_scores.append(corr[0])

        rank_dict = rank_to_dict(corr_scores, feature_names, scale=False)
        return rank_dict

    ranks_dict = {}

    ranks_dict['MI'] = mutual_information_features_vs_labels(X, Y, feature_names)
    ranks_dict['Corr'] = abs_correlation_features_vs_labels(X, Y, feature_names)

    MI_sorted_names = sorted(ranks_dict['MI'], key=ranks_dict['MI'].get, reverse=True)

    MI_sorted_values = [ranks_dict['MI'][name] for name in MI_sorted_names]
    Corr_sorted_values = [ranks_dict['Corr'][name] for name in MI_sorted_names]

    ml_helpers.multi_bar_plot([MI_sorted_values, Corr_sorted_values], ['MI', '|Corr|'], MI_sorted_names, save=True, fontsize=8)


def rec_feature_elimination(X, Y, nr_features_to_select, feature_names):
    """
    Recursive feature elimination algorithm, using Random Forest importance scores

    :param X: Feature matrix (columns are the feature vectors)
    :param Y: Labels
    :param nr_features_to_select: Size of resulting feature subset
    :param feature_names: Names of the features (corresponding to colums of X)
    :return: remaining_features, removed_feature_names, removed_feature_indices, removed_feature_scores
    """
    nr_features_to_remove = X.shape[1] - nr_features_to_select
    feature_names = list(feature_names)
    removed_feature_indices = []
    removed_feature_names = []
    removed_feature_scores = []

    for i in range(nr_features_to_remove):
        print('Round: {}'.format(i))
        # clf = clone(classifier)  # yields a new estimator with the same parameters that has not been fit on any data.

        rf = RandomForestRegressor(n_jobs=6)
        rf.fit(X, Y)

        scores = rf.feature_importances_
        min_score = np.min(scores)
        min_score_idx = np.argmin(scores)

        X = ml_helpers.eliminate_features_by_index(X, [min_score_idx])

        removed_feature_indices.append(min_score_idx)
        removed_feature_names.append(feature_names[min_score_idx])
        removed_feature_scores.append(min_score)

        print('Removed feature: {}\nScore: {}'.format(feature_names[min_score_idx], min_score))
        del feature_names[min_score_idx]
    remaining_features = feature_names
    print('Remaining Features: {}'.format(remaining_features))

    return remaining_features, removed_feature_names, removed_feature_indices, removed_feature_scores


def get_features_ordered_by_score(X, Y, feature_names, suffix):
    """
    Runs rec_feature_elimination() until only one feature is left. Writes the removed features to a file in the order
    in which they were removed. --> This list is ordered by the importance of the features.

    :param X: Feature matrix (columns are the feature vectors)
    :param Y: Labels
    :param feature_names: Names of the features (corresponding to colums of X)
    :param suffix: Suffix for the list filename
    """
    remaining_features, removed_feature_names, _, removed_feature_scores = rec_feature_elimination(X, Y, 1, feature_names)
    outpath = os.path.join('./data', 'rfe_removed_features_list-{}.txt'.format(suffix))

    features_ordered_by_score = removed_feature_names + remaining_features
    feature_scores = removed_feature_scores + [1.0] # set score of last feature to 1.0

    with open(outpath, 'w') as fp:
        fp.write(','.join([name for name in features_ordered_by_score]) + '\n')
        fp.write(','.join([str(index) for index in feature_scores]))


def train_and_test_with_different_feature_subsets(estimator, nr_features_list, featurename_csv_path, dataset_path, log_suffix = '',
                                                  nr_files_to_evaluate=None,  prob_scores=False, standardize=True, balance=False):
    """
    Using the random forest based feature-selection algorithm as described in Section we have already a list of all the
    features ordered by importance. So to select the best K features, we can just take the first K entries from this
    list. However, we still have to find the optimal number of features for each supervised algorithm. To do so, we evaluate
    each of the supervised models on the training and validation sets we have generated, and repeat this for different
    feature counts to find the number of features where each model performs best.

    :param estimator: Sklearn estimator object
    :param nr_features_list: List of the feature counts that should be evaluated
    :param featurename_csv_path: .csv file generated by get_features_ordered_by_score()
    :param dataset_path: Directory containing the train and validation sets
    :param log_suffix: The output of this function will be logged. Log_suffix will be appended to the logging filename.
    :param nr_files_to_evaluate: If set, only the first <nr_files_to_evaluate> train/val sets in <dataset_path> will
                                 be evaluated.
    :param prob_scores: Set True to generate Precision/Recall ROC type curves
    :param standardize: Set True to enable standardization
    :param balance: Set True to enable class balancing
    """
    precisions = []
    recalls = []
    f1s = []

    with open(featurename_csv_path, 'r') as fp:
        feature_names = fp.readline().strip().split(',')
        feature_indices = fp.readline().strip().split(',')

    feature_names.append('Label')

    for nr_features in nr_features_list:
        print('Using {} of {} features'.format(nr_features, len(feature_names)-1))
        start = time.time()
        suffix = '{}_features-'.format(nr_features) + log_suffix
        precision, recall, f1, log_dir = ml_training.train_and_test_multiple_sets(estimator, dataset_path, log_suffix=suffix,
                                                                                  selected_feature_names=feature_names[-(nr_features+1):], balance=balance,
                                                                                  standardize=standardize, nr_files_to_evaluate=nr_files_to_evaluate,
                                                                                  prob_scores=prob_scores)
        end = time.time()
        print('Took: {}'.format(end - start))

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print('##### SUMMARY #####')
    summary = ''
    for i, nr_features in enumerate(nr_features_list):
        summary += '{} of {} features\n'.format(nr_features, len(feature_names))
        summary += 'Precision (mean, std): {}\n'.format(precisions[i])
        summary += 'Recall (mean, std): {}\n'.format(recalls[i])
        summary += 'F1 (mean, std): {}\n\n'.format(f1s[i])

    with open(os.path.join(log_dir, 'summary_results.txt'), 'w') as fp:
        fp.write(summary)
        print(summary)


def test_multiple_classifiers_on_different_feature_subsets(estimators_dict, featurename_csv_path, datasets_path, nr_files_to_evaluate=None, standardize=True, balance=False):
    """
    Function used to conduct the feature-selection process using the function train_and_test_with_different_feature_subsets()
    See doc of train_and_test_with_different_feature_subsets()

    :param estimators_dict: Dictionary: String->SklearnModel
    :param featurename_csv_path: .csv file generated by get_features_ordered_by_score()
    :param datasets_path: Directory containing the train and validation sets
    :param nr_files_to_evaluate: If set, only the first <nr_files_to_evaluate> train/val sets in <dataset_path> will
                                 be evaluated.
    :param standardize: Set True to enable standardization
    :param balance: Set True to enable class balancing
    """
    for name, estimator in estimators_dict.items():
        print('Evaluating: {}'.format(name))
        if standardize:
            name += '-std'
        if balance:
            name += '-balanced'
        train_and_test_with_different_feature_subsets(estimator, [58, 40, 30, 20, 15, 10, 5], featurename_csv_path, datasets_path,
                                                      log_suffix=name, nr_files_to_evaluate=nr_files_to_evaluate, standardize=standardize, balance=balance)