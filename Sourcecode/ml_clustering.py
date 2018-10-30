"""
Master Thesis
Network Monitoring and Attack Detection

ml_clustering.py
Functions used to perform the unsupervised clustering experiments on the CICIDS2017 and the LS datasets


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import os
import sys
from collections import Counter

from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import time
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import math
import pickle

from tqdm import tqdm
from datetime import datetime, timezone, timedelta

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ml_feature_selection
import ml_helpers
from ml_clustering_helpers import *
import ml_feature_selection
import ml_data

from ml_data import label_mapping_dict
from ml_data import label_binary_mapping_dict


def rf_experiment(csv_train_path, label_mapping_dict):
    """
    Experiment where we train a supervised Random Forest classifier on the CICIDS2017 data

    :param csv_train_path: path of the CICIDS2017 .csv file
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    """
    X_train, Y_train, X_test, Y_test = ml_helpers.load_dataset(csv_train_path, 0.7, categorical_feature_mapping=label_mapping_dict, standardize=False)

    estimator = RandomForestClassifier(n_jobs=6, random_state=0)

    start = time.time()
    print('Training the model with {} samples...'.format(len(Y_train)))
    estimator.fit(X_train, Y_train)
    end = time.time()
    print('Training took: {}'.format(end - start))

    print('Performing predictions on testset ...')
    start = time.time()
    Y_predicted = estimator.predict(X_test)
    end = time.time()
    print('Inference took: {}'.format(end - start))
    print(accuracy_score(Y_test, Y_predicted))
    print(classification_report(Y_test, Y_predicted))


def cic17_feature_selection(csv_path):
    """
    Function used to eliminate the zero correlation features, and to run the recursive feature elimination algorithm

    :param csv_path: .csv path of the dataset
    """
    X, Y, _, _ = ml_helpers.load_dataset(csv_path, 1.0, selected_feature_names=ml_data.cic17_csv_header_names,
                                         categorical_feature_mapping=label_binary_mapping_dict, standardize=False)
    ml_feature_selection.MI_vs_Corr_vs_DC(X, Y, ml_data.cic17_csv_header_names[:-1])

    X, feature_names = ml_helpers.eliminate_features(X, ml_data.cic17_zero_corr_features, ml_data.cic17_csv_header_names[:-1])
    ml_feature_selection.get_features_ordered_by_score(X, Y, feature_names, 'binary')


def clustering(model_name, data_path, nr_clusters, selected_features, label_mapping_dict, categorical_feature_mapping=None,
               majority_classes_only=False, plot_manifold=False, subsampling=None, pca_components=None, standardize=True,
               save_name=None, evaluate=False, train_classifier=False):
    """
    Function used to train the K-means and DBSCAN clusterering models.


    :param model_name: set to 'K-means' or 'DBSCAN'
    :param data_path: The path of the dataset. (Can be either .csv or .pickle format)
    :param nr_clusters: Number of clusters (only used for K-means)
    :param selected_features: List of the selected feature-names
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param categorical_feature_mapping: Dictionary that maps the categorical feature names to numbers
    :param majority_classes_only: If True, print only the names of the majority classes for each cluster.
                                  If False, print all classes and the corresponding counts for each clusters
    :param plot_manifold: If True, generate T-SNE plots
    :param subsampling: Set to number between 0 and 1 for uniform subsampling, or to 'log', for the log-subsampling mode
    :param pca_components: If not None, run PCA on the features with pca_components
    :param standardize: Set to True to enable standardization of the feature vectors
    :param save_name: The trained model will be stored under this name. (If None, the model will not be stored)
    :param evaluate: If True, run evaluation of performance metrics on the data using the cluster-centers predictors
                     or a trained Random Forest classifier (DBSCAN predict is very slow for big data...)
    :param train_classifier: If True, train a Random Forest classifier, using the majority classes of each cluster as
                             ground truth
    :return: model, cluster_id_to_majority_name_dict, X, Y
    """
    if '.csv' in data_path:
        data = ml_helpers.load_dataset(data_path, 1.0, selected_feature_names=selected_features,
                                       categorical_feature_mapping=categorical_feature_mapping, standardize=standardize,
                                       pca_components=pca_components)
        X = data[0]
        Y = data[1]
        del data

    elif '.pickle' in data_path:
        X, Y, all_columns = load_data_from_pickle(data_path)

        if selected_features is not None:
            selected_feature_indices = ml_helpers.get_feature_indices_by_name(selected_features[:-1], all_columns)
            X = X[:, selected_feature_indices]

        if pca_components is not None:
            print('... PCA with {} components'.format(pca_components))
            pca = PCA(n_components=pca_components)
            pca.fit(X)
            X = pca.transform(X)

    if subsampling is not None:
        print('... subsampling')
        if subsampling == 'log':
            X, Y = log_subsampling(X, Y, 20, 1000)
        else:
            X, Y = ml_helpers.subsample(X, Y, subsampling)

    print('Fitting the model on {} samples...'.format(X.shape[0]))
    start_time = time.time()
    if model_name == 'K-means':
        model = KMeans(n_clusters=nr_clusters, n_jobs=5, max_iter=500, random_state=0)
        model.fit(X)
    elif model_name == 'DBSCAN':
        model = DBSCAN(eps=3, min_samples=3)
        model.fit(X)
        label_mapping_dict['Label']['Outlier'] = -1
        nr_clusters = len(set(model.labels_))
    else:
        print('Enter a valid model name')
        return
    end_time = time.time()
    print('Took {}s'.format(end_time - start_time))
    print('Nr clusters: {}'.format(nr_clusters))

    inv_label_mapping_dict = {v: k for k, v in label_mapping_dict['Label'].items()}
    # Y_names = [inv_label_mapping_dict[y_val] for y_val in Y]

    cluster_id_to_majority_name_dict = get_cluster_assignment_counts(model, Y, label_mapping_dict, majority_classes_only=majority_classes_only)
    if model_name == 'DBSCAN':
        cluster_id_to_majority_name_dict[-1] = 'Outlier'

    classifier = None
    if train_classifier:
        classifier = train_classifier_from_clustering_labels(X, model, cluster_id_to_majority_name_dict, label_mapping_dict)

    if save_name is not None:
        ml_helpers.save_model({'model': model, 'majority_class_dict': cluster_id_to_majority_name_dict}, './models/cic17', '{}.pickle'.format(save_name))
        if train_classifier:
            ml_helpers.save_model({'model': classifier, 'majority_class_dict': cluster_id_to_majority_name_dict}, './models/cic17', '{}-classifier.pickle'.format(save_name))

    if evaluate:
        evaluate_clustering_model_on_classes(Y, model.labels_, cluster_id_to_majority_name_dict, label_mapping_dict)
        if train_classifier:
            print('\n\nClassifier predictions: ')
            labels = classifier.predict(X)
            target_names = sorted(label_mapping_dict['Label'], key=label_mapping_dict['Label'].get)
            print(classification_report(Y, labels, target_names=target_names))

    if plot_manifold:
        unique_cluster_ids = summarize_clusters_with_same_majority_class(model.labels_, cluster_id_to_majority_name_dict)
        ml_helpers.plot_manifold(X, unique_cluster_ids, method_name='TSNE', y_to_name_mapping=cluster_id_to_majority_name_dict, colors=ml_data.color_map, suffix='unique_clusterIDs')
        ml_helpers.plot_manifold(X, model.labels_, method_name='TSNE', y_to_name_mapping=cluster_id_to_majority_name_dict, colors=ml_data.color_map, suffix='all_clusterIDs')
        ml_helpers.plot_manifold(X, Y, method_name='TSNE', y_to_name_mapping=inv_label_mapping_dict, colors=ml_data.color_map, suffix='GNDTruth')

    return model, cluster_id_to_majority_name_dict, X, Y


def clustering_with_different_featuresubsets(model_name, csv_path, feature_sets_dict, nr_features, nr_clusters=15,
                                             pca_components=None, subsampling=None):
    """
    Run the clustering() function using different feature-sets.
    We used this function in the feature-selection phase.

    For the argument descriptions, see clustering()
    """
    for name, features in feature_sets_dict.items():
        print('Using top {} {} features ...'.format(nr_features, name))

        clustering(model_name=model_name, csv_path=csv_path, nr_clusters=nr_clusters,
                   selected_features=features[-nr_features:] + ['Label'],
                   label_mapping_dict=label_mapping_dict, majority_classes_only=True, plot_manifold=False,
                   subsampling=subsampling)
        print()


def kmeans_predict(model, X):
    """
    Function to classify samples using the cluster-centers of a model.
    "K-predictor" type

    :param model: Loaded sklearn clusterer model
    :param X: Data with feature vectors
    :return: Labels of the closest centroid for each sample in X
    """
    nr_samples = X.shape[0]
    centroids = model.cluster_centers_
    closest_centroid_labels = np.zeros(nr_samples, dtype=int)

    # find its closest centroid:
    for i in tqdm(range(nr_samples)):
        diff = centroids - X[i, :]  # NumPy broadcasting

        # dist = np.sqrt(np.sum(diff ** 2, axis=1))  # Euclidean distance
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        closest_centroid_labels[i] = np.argmin(dist)

    return closest_centroid_labels


def dbscan_predict(model, X):
    """
    Classifier function for DBSCAN.
    check whether a given sample is within eps distance of one of the core\_samples.
    If it is, it takes the label of the core sample, if it is not, it's noise.

    :param model: Loaded sklearn clusterer model
    :param X: Data with feature vectors
    :return: Labels of the closest coresample within epsilon radius for each sample in X
    """
    nr_samples = X.shape[0]
    # Result is noise by default
    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in tqdm(range(nr_samples)):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


def predict_by_distance_threshold(X, Y_gndtruth, model_path, label_mapping_dict, distance_threshold):
    """
    The "Threshold-predictor" type we propose in the thesis.
    lassify flows that are close to the found malicious cluster as malicious, while classifying samples farther away
    as normal. We base the predictions of the classifier on a distance-threshold. All the flows whose distances to
    the malicious center lie below the distance threshold we classify as malicious.

    :param X: Data with feature vectors
    :param Y_gndtruth: Label ground truth (we use this only for evaluation of the predictor)
    :param model_path: Path of the clusterer model (.pickle)
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param distance_threshold: See above
    """
    model, majority_class_dict = load_model_and_majority_dict(model_path)

    mal_cluster_indices = []
    for cluster_id, maj_class in majority_class_dict.items():
        if maj_class == 'Malicious':
            mal_cluster_indices.append(cluster_id)

    Y_pred = np.zeros(X.shape[0], dtype=float)
    Y_pred[:] = 0.0

    for nr, centroid in enumerate(model.cluster_centers_[mal_cluster_indices]):
        print('Looking for closest points to centroid {}'.format(nr))
        distances = np.linalg.norm(X - centroid, axis=1)
        # diff = np.linalg.norm(centroid_1 - centroid_2, axis=0)

        X_mal = X[np.where(Y_gndtruth==1.0)]
        X_nor = X[np.where(Y_gndtruth == 0.0)]
        mal_distances = np.linalg.norm(X_mal - centroid, axis=1)
        nor_distances = np.linalg.norm(X_nor - centroid, axis=1)

        print('Distances to all samples: \n min: {}\nmax: {}\nmean (std): {} ({:.2f})\n'.format(np.min(distances), np.max(distances), np.mean(distances), np.std(distances)))
        print('Distances to malicious samples: \n min: {}\nmax: {}\nmean (std): {} ({:.2f})\n'.format(np.min(mal_distances), np.max(mal_distances), np.mean(mal_distances), np.std(mal_distances)))
        print('Distances to normal samples: \n min: {}\nmax: {}\nmean (std): {} ({:.2f})\n'.format(np.min(nor_distances), np.max(nor_distances), np.mean(nor_distances), np.std(nor_distances)))

        Y_pred[np.argwhere(distances < distance_threshold)] = 1.0


    print(classification_report(Y_gndtruth, Y_pred))
    print('Distance threshold used: {}'.format(distance_threshold))


def reduce_centroids(data_path, model_path, selected_columns, mode='greedy', distance_threshold=15, new_centroid_count=None):
    """
    Function used to reduce the the number of redundant clusters by merging cluster-centers that lie particularly
    close together. (Below a distance_threshold)

    :param data_path: Path to the test dataset to evaluate the original model on the data, in order to be able to compare the
                      prediction quality of the reduced model
    :param model_path: Path to the clusterer model file (.pickle)
    :param selected_columns: Selected feature names
    :param mode: 'K-means' for a second K-means stage, 'greedy' for the algorithm using distance_threshold proposed in the thesis
    :param distance_threshold: See above
    :param new_centroid_count: Only used for 'K-means' mode... K of the second K-means stage
    :return: model, new_cluster_id_to_majority_name_dict
    """

    # First evaluate the original model on the data, in order to be able to compare the prediction quality of the
    # reduced model
    if '.csv' in data_path:
        X, Y, _, _ = ml_helpers.load_dataset(data_path, 1.0, selected_feature_names=selected_columns,
                                             categorical_feature_mapping=label_mapping_dict, subsampling=None, standardize=True)
    elif '.pickle' in data_path:
        X, Y, all_columns = load_data_from_pickle(data_path)
        selected_feature_indices = ml_helpers.get_feature_indices_by_name(selected_columns[:-1], all_columns)
        X = X[:, selected_feature_indices]

    load_model_and_predict(X_test=X, model_path=model_path,
                           label_mapping_dict=label_mapping_dict, Y_test=Y,
                           plot_manifold=False)

    model, cluster_id_to_majority_name_dict = load_model_and_majority_dict(model_path)
    cluster_ids, cluster_sizes = np.unique(model.labels_, return_counts=True)

    if mode == 'K-means':
        centroid_model = KMeans(n_clusters=new_centroid_count, n_jobs=5, max_iter=500, random_state=0)
        centroid_model.fit(model.cluster_centers_)
        cluster_ids, cluster_sizes = np.unique(centroid_model.labels_, return_counts=True)

        old_cluster_id_to_new_id_dict = dict()
        for new_cluster_id, old_cluster_id in zip(centroid_model.labels_, np.arange(len(model.cluster_centers_))):
            old_cluster_id_to_new_id_dict[old_cluster_id] = new_cluster_id

        new_centroids = centroid_model.cluster_centers_
        model.cluster_centers_ = new_centroids
        model.labels_ = [old_cluster_id_to_new_id_dict[old] for old in model.labels_]

        save_name = os.path.splitext(os.path.basename(model_path))[0] + '-reduced{}.pickle'.format(new_centroid_count)

    elif mode == 'greedy':
        model.cluster_centers_, model.labels_ = reduce_clusters_by_distance(model, cluster_id_to_majority_name_dict=None, distance_threshold=distance_threshold)
        save_name = os.path.splitext(os.path.basename(model_path))[0] + '-reduced{}.pickle'.format(distance_threshold)

    new_cluster_id_to_majority_name_dict = get_cluster_assignment_counts(model, Y, label_mapping_dict, majority_classes_only=True)

    ml_helpers.save_model({'model': model, 'majority_class_dict': new_cluster_id_to_majority_name_dict}, './models/cic17', save_name)

    load_model_and_predict(X_test=X, model_path=os.path.join('./models/cic17', save_name), label_mapping_dict=label_mapping_dict, Y_test=Y,
                           plot_manifold=False)

    return model, new_cluster_id_to_majority_name_dict


def analyse_close_clusters(data_path, model_path, label_mapping_dict, distance_threshold):
    """
    Helper function for reduce_centroids()
    """
    model, cluster_id_to_majority_name_dict = load_model_and_majority_dict(model_path)
    # get_cluster_center_distances(model)
    _, Y, _ = load_data_from_pickle(data_path)
    get_cluster_assignment_counts(model, Y, label_mapping_dict, majority_classes_only=True)
    reduce_clusters_by_distance(model, cluster_id_to_majority_name_dict=cluster_id_to_majority_name_dict, distance_threshold=distance_threshold)


def reduce_clusters_by_distance(model, cluster_id_to_majority_name_dict=None, distance_threshold=2):
    """
    Called in reduce_centroids().
    Implements the distance_threshold cluster reduction algorithm.
    """
    centroids = model.cluster_centers_
    cluster_ids, cluster_sizes = np.unique(model.labels_, return_counts=True)
    cluster_id_to_clustersize_dict = dict((c_id, c_size) for c_id, c_size in zip(cluster_ids, cluster_sizes))

    distance_matrix = get_cluster_center_distances(model)
    clusters_below_threshold = set()
    cluster_id_to_below_threshold_ids_dict = dict()

    for m in range(len(centroids)):
        for n in range(len(centroids)):
            if m>=n:
                continue
            if distance_matrix[m][n] < distance_threshold:
                if cluster_id_to_majority_name_dict is not None:
                    clusters_below_threshold.add('{}({})-{}({})'.format(m, cluster_id_to_majority_name_dict[m], n, cluster_id_to_majority_name_dict[n]))
                else:
                    clusters_below_threshold.add('{}-{}'.format(m,n))

                if m in cluster_id_to_below_threshold_ids_dict:
                    cluster_id_to_below_threshold_ids_dict[m].add(n)
                else:
                    cluster_id_to_below_threshold_ids_dict[m] = set([n])
                if n in cluster_id_to_below_threshold_ids_dict:
                    cluster_id_to_below_threshold_ids_dict[n].add(m)
                else:
                    cluster_id_to_below_threshold_ids_dict[n] = set([m])

    cluster_id_to_new_centroid_dict = {}
    old_cluster_id_to_new_id_dict = dict()
    cluster_ids_used_for_reduction = set()

    for cluster_id, below_threshold_ids in cluster_id_to_below_threshold_ids_dict.items():
        # GREEDY ALGORITHM, if cluster center already used --> skip
        if cluster_id in cluster_ids_used_for_reduction:
            continue
        else:
            below_threshold_ids = below_threshold_ids.difference(cluster_ids_used_for_reduction)
            if below_threshold_ids == set():
                continue

        new_c = np.sum([centroids[cluster_id]] + [centroids[c_id] for c_id in list(below_threshold_ids)], axis=0)/(len(below_threshold_ids)+1)
        cluster_id_to_new_centroid_dict[cluster_id] = new_c
        cluster_ids_used_for_reduction.add(cluster_id)
        cluster_ids_used_for_reduction = cluster_ids_used_for_reduction.union(below_threshold_ids)

        for old_cluster_id in below_threshold_ids:
            old_cluster_id_to_new_id_dict[old_cluster_id] = cluster_id

    new_labels = []
    for old_label in model.labels_:
        # this cluster was one lying below the distance threshold, and was thus used for the reduction
        if old_label in cluster_ids_used_for_reduction:
            # the old cluster id got mapped to a new id
            if old_label in old_cluster_id_to_new_id_dict:
                new_labels.append(old_cluster_id_to_new_id_dict[old_label])
            # the cluster id remains the same
            else:
                new_labels.append(old_label)
        # cluster was not involved in reduction process --> leave it unchanged
        else:
            new_labels.append(old_label)

    # map the remaining cluster ids to integers starting at 0 (e.g. [2,4,15] --> [1,2,3])
    new_label_mapping = {old: new for new, old in enumerate(sorted(list(set(new_labels))))}
    new_labels = [new_label_mapping[label] for label in new_labels]

    new_nr_clusters = len(set(new_labels))

    new_centroids = np.zeros([new_nr_clusters, model.cluster_centers_.shape[1]])

    for old_id in range(len(centroids)):
        if old_id in cluster_ids_used_for_reduction:
            if old_id in old_cluster_id_to_new_id_dict:
                centroid_index = new_label_mapping[old_cluster_id_to_new_id_dict[old_id]]
                new_centroids[centroid_index] = cluster_id_to_new_centroid_dict[old_cluster_id_to_new_id_dict[old_id]]
            else:
                centroid_index = new_label_mapping[old_id]
                new_centroids[centroid_index] = cluster_id_to_new_centroid_dict[old_id]
        else:
            centroid_index = new_label_mapping[old_id]
            new_centroids[centroid_index] = model.cluster_centers_[old_id]

    # new_centroids = np.unique(np.array(new_centroids))
    print(clusters_below_threshold)
    print('Number of clusters reduced from {} to {}'.format(len(model.cluster_centers_), new_nr_clusters))
    print('Threshold used for centroid merging: {}'.format(distance_threshold))

    return np.array(new_centroids), new_labels


def save_subsampled_data_to_pickle(csv_path, out_name, label_mapping_dict, selected_columns, mode='1'):
    """
    Function to apply different subsampling techniques to the data, and to store the subsampled sets in .pickle format

    :param csv_path: Path to the .csv file containing the full dataset
    :param out_name: Name of the .pickle file holding the subsampled data
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param selected_columns: Selected feature names
    :param mode: 1 for log-subsampling, 2 for uniform subsampling
    :return: The filepath of the saved subsampled .pickle data
    """
    X, Y, _, _ = ml_helpers.load_dataset(merged_path, 1.0, selected_feature_names=selected_columns,
                                         subsampling=None, categorical_feature_mapping=label_mapping_dict, standardize=True)

    if mode == '1':
        X, Y = log_subsampling(X, Y, n=20, k=1000)
    elif mode == '2':
        X, Y = ml_helpers.subsample(X, Y, subsampling_fraction=0.1)

    out_path = os.path.join(os.path.dirname(csv_path), out_name)

    pickle.dump([X, Y, selected_columns], open(out_path, "wb"))
    print('Subsampled data saved to: {}'.format(out_path))

    return out_path


def log_subsampling(X, Y, n=20, k=1000):
    """
    Log-subsampling method.
    Reduces the sample count of each attack class in the data using a logarithmic function.  to keep the number of
    samples of the Benign class dominant, instead of using the above logarithmic downscaling we apply uniform subsampling
    selecting 0.1% of all Benign samples.

    :param X: Features
    :param Y: Labels
    :param n: n*math.log(k*size)
    :param k: n*math.log(k*size)
    :return: subsampled X, Y
    """
    classes = list(set(Y))
    x_subsampled_sets = []
    y_subsampled_sets = []

    for y in classes:
        x_subset = X[np.where(Y==y)]
        y_subset = Y[np.where(Y==y)]
        orig_size = len(x_subset)
        size = len(x_subset)

        if y == 0:
            # uniform subsambling of the benign class
            new_size = int(0.001*len(x_subset))
        else:
            new_size = math.ceil(n*math.log(k*size))
        if new_size < orig_size:
            subsampling_idx = np.random.choice(len(x_subset), new_size)
            x_subsampled_sets.append(x_subset[subsampling_idx])
            y_subsampled_sets.append(y_subset[subsampling_idx])
        else:
            x_subsampled_sets.append(x_subset)
            y_subsampled_sets.append(y_subset)

        new_size = len(x_subsampled_sets[-1])
        print('old: {}, new: {}'.format(orig_size, new_size))

    X = np.concatenate(x_subsampled_sets)
    Y = np.concatenate(y_subsampled_sets)

    # shuffle
    random_permutation = np.random.permutation(np.arange(len(Y)))
    X = X[random_permutation]
    Y = Y[random_permutation]

    return X, Y


def ls18_DoS_experiment(model_path, selected_features, label_mapping_dict, start_time=None, end_time=None, single_plot=False,
                        classifier_mode=False, suffix=''):
    """
    Experiment where we take the DoS cluster-centers of models trained on the CICIDS2017 and then try to detect
    DoS flows in the LS18 data using these clusters. (...Did not work)

    :param model_path: Path of the clusterer model trained on CICIDS2017
    :param selected_features: Selected feature names
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param start_time: In final plot, show predictions after start_time
    :param end_time: In final plot, show predictions until end_time
    :param single_plot: If true,  generate all plots in single figures. Else, use subfigures.
    :param classifier_mode: True if we use a classifier model trained on the clusters. Else use clusterer prediction functions
    :param suffix: Suffix for plot filename
    """
    ls18_csv_path = './data/FlowMeter/ls18_Dup15/ls18-all-traffic24.pcap_Flow-labelledT-intExt-sub0.3.csv'

    # Problem: featurenames in header of ls18 .csv differs the cic17 header... --> map the names first
    ls18_feature_names = []
    for cic_name in selected_features:
        ls18_feature_names.append(ml_data.cic17_to_ls18_headernames_dict[cic_name])

    meta_names = ['Timestamp', 'Src IP', 'Dst IP', 'Dst Port']
    readcols = ls18_feature_names + meta_names

    print('Loading the data ...')
    df = pd.read_csv(ls18_csv_path, sep=',', usecols=readcols)
    X = df[ls18_feature_names].values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    meta_data = df[['Timestamp', 'Src IP', 'Dst IP', 'Dst Port']].values

    # model, majority_class_dict = load_model_and_majority_dict(model_path)
    Y_pred = load_model_and_predict(X, model_path, label_mapping_dict, classifier_mode=classifier_mode)

    if start_time and end_time:
        start_time_dt = datetime.strptime(start_time, '%d/%m/%Y %H:%M:%S')
        start_timestamp = start_time_dt.replace(tzinfo=timezone.utc).timestamp()
        end_time_dt = datetime.strptime(end_time, '%d/%m/%Y %H:%M:%S')
        end_timestamp = end_time_dt.replace(tzinfo=timezone.utc).timestamp()

    DDoS_indices = np.where(Y_pred==label_mapping_dict['Label']['DDoS'])
    DoS_slowloris_indices = np.where(Y_pred==label_mapping_dict['Label']['DoS slowloris'])
    DoS_Hulk_indices = np.where(Y_pred==label_mapping_dict['Label']['DoS Hulk'])
    DoS_GoldenEye_indices = np.where(Y_pred==label_mapping_dict['Label']['DoS GoldenEye'])

    all_DoS_indices_dict = {'DDoS': DDoS_indices, 'Dos Slowloris': DoS_slowloris_indices, 'Dos Hulk': DoS_Hulk_indices, 'Dos GoldenEye': DoS_GoldenEye_indices}

    all_timelines_dict = {'DDoS': ([], []), 'Dos Slowloris': ([], []),
                          'Dos Hulk': ([], []), 'Dos GoldenEye': ([], [])}
    merged_malicious_timestamps = []
    merged_malicious_timestrings = []


    UTC_OFFSET = 2
    for attack_name, indices in all_DoS_indices_dict.items():
        for time, src_ip, dst_ip, dst_port in meta_data[indices]:
            timestamp_dt = datetime.strptime(time, '%d/%m/%Y %H:%M:%S') - timedelta(hours=UTC_OFFSET)
            utc_timestamp = timestamp_dt.replace(tzinfo=timezone.utc).timestamp()
            if start_time and end_time:
                if start_timestamp < start_timestamp or utc_timestamp > end_timestamp:
                    continue

            merged_malicious_timestamps.append(utc_timestamp)
            merged_malicious_timestrings.append(time)

            all_timelines_dict[attack_name][0].append(utc_timestamp)
            all_timelines_dict[attack_name][1].append(time)

    first_timestamp = min(merged_malicious_timestamps)
    last_timestamp = max(merged_malicious_timestamps)
    nr_time_ticks = 10

    if start_time and end_time:
        time_axis = np.arange(start_timestamp, end_timestamp + 1, (end_timestamp-start_timestamp)/nr_time_ticks)
    else:
        time_axis = np.arange(first_timestamp, last_timestamp+1, (last_timestamp-first_timestamp)/nr_time_ticks)

    time_axis_strings = [datetime.fromtimestamp(stamp).strftime('%d/%m/%Y %H:%M:%S') for stamp in time_axis]

    if single_plot:
        plt.figure()
        plt.hlines(y=1, xmin=first_timestamp, xmax=last_timestamp)  # Draw a horizontal line
        plt.eventplot(merged_malicious_timestamps, orientation='horizontal', colors='b')
        # plt.xticks(time_axis, rotation='vertical')
        plt.xticks(time_axis, time_axis_strings, rotation='vertical')
        plt.tight_layout()
        plt.show()

    else:
        nr_attack_types = len(all_DoS_indices_dict.keys())
        lineoffsets = np.arange(nr_attack_types)
        # linelengths = np.ones(nr_attack_types) * 0.5

        attack_names = []
        plt.figure()
        nr = 0
        for attack_name, timestamp_and_string_tuple in all_timelines_dict.items():
            # plt.hlines(i,1,10)  # Draw a horizontal line
            plt.eventplot(timestamp_and_string_tuple[0], lineoffsets=lineoffsets[nr], linelengths=0.7, orientation='horizontal', colors='b')
            attack_names.append(attack_name)
            nr += 1

        plt.xticks(time_axis, time_axis_strings, rotation='vertical')
        plt.yticks(lineoffsets, attack_names)
        plt.tight_layout()
        stamp = str(int(datetime.now().timestamp()))
        plt.savefig("figures/ls18eval-{}-{}.png".format(stamp, suffix))
        plt.show()
        plt.show()


def outlier_detection_approaches(data_path, selected_features, label_mapping_dict, method_name='HBOS', top_k_vector = [10]):
    """
    Function that implements three outlier detection algorithms: HBOS, IsolationForest, LocalOutlierFactor

    :param data_path:
    :param selected_features:
    :param label_mapping_dict:
    :param method_name: 'HBOS', 'IF', or 'LOF'
    :param top_k_vector: How many anomalies can we find in the top k samples? (samples with top-k outlier scores)
    """
    print('Method: {}'.format(method_name))
    print('Data path: {}'.format(data_path))
    if '.csv' in data_path:
        X, Y, _, _ = ml_helpers.load_dataset(data_path, 1.0, selected_feature_names=selected_features,
                                             categorical_feature_mapping=label_mapping_dict, subsampling=None, standardize=True)
    elif '.pickle' in data_path:
        X, Y, all_columns = load_data_from_pickle(data_path)
        selected_feature_indices = ml_helpers.get_feature_indices_by_name(selected_columns[:-1], all_columns)
        X = X[:, selected_feature_indices]

    data = pd.DataFrame(X)
    ascending = None

    start = time.time()
    if method_name == 'HBOS':
        import hbos as HBOS
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.401.5686&rep=rep1&type=pdf
        # the fixed bin width approach estimates the density poorly (a few bins may contain most of
        # the data). Since anomaly detection tasks usually involve such gaps in the value
        # ranges due to the fact that outliers are far away from normal data, we recommend
        # using the dynamic width mode, especially if distributions are unknown or long
        # tailed. Besides, also the number of bins k needs to be set. An often used rule of
        # thumb is setting k to the square root of the number of instances N.

        nr_bins = math.sqrt(len(data))
        model = HBOS.HBOS(mode_array=["dynamic binwidth"]*len(data.columns), bin_info_array = [nr_bins]*len(data.columns))
        ascending = False


    elif method_name == 'IF':
        model = IsolationForest()
        model.fit(data)
        ascending = True

    elif method_name == 'LOF':
        model = LocalOutlierFactor(n_neighbors=20)
        model.fit(data)
        ascending = True
    else:
        return

    if method_name == 'IF':
        scores = model.decision_function(data)
    elif method_name == 'LOF':
        scores = model.negative_outlier_factor_
    else:
        scores = model.fit_predict(data)
    end = time.time()
    print('Took: {}s'.format(end - start))

    # Merge results with data and labels
    data['scores'] = scores
    data['Label'] = Y.astype(int)
    precisions = []

    score_samples_sorted = data.sort_values(by=['scores'], ascending=ascending)
    for top_k in top_k_vector:
        top_k_score_samples = score_samples_sorted[:top_k]
        # How many anomalies can we find in the top k samples?
        nr_anomalies_in_top_k = len(top_k_score_samples[lambda x: x['Label'] != 0])
        print('Nr. of anomalies found in top {} score samples: {}'.format(top_k, nr_anomalies_in_top_k))
        print(top_k_score_samples[:top_k]['Label'].value_counts())

        current_precision = nr_anomalies_in_top_k*100.0/top_k
        precisions.append(current_precision)

    print('\nPRECISIONS:')
    for k, precision in zip(top_k_vector, precisions):
        print('k={}: {:.2f}%'.format(k, precision))



if __name__ == "__main__":

    ##################################
    ### CICIDS2017 EXPERIMENTS     ###
    ##################################

    if len(sys.argv) > 1:
        if sys.argv[1] == 'euler':
            data_path = './data/CIC17'
            subsampled_path_1 = './data/CIC17/sub1-std.pickle'
            subsampled_path_2 = './data/CIC17/sub2-std.pickle'
    else:
        data_path = '/mnt/data/datasets/CIS-2017'
        subsampled_path_1 = '/mnt/data/datasets/CIS-2017/sub1-std.pickle'
        subsampled_path_2 = '/mnt/data/datasets/CIS-2017/sub2-std.pickle'


    full_path = os.path.join(data_path, 'cic17_all_merged-noInf-shuffled.csv')
    test_path = os.path.join(data_path, 'Friday-WorkingHours-Morning.pcap_ISCX-noInf.csv')

    ### DATASET GENERATION ###
    # merge_all_csvs()
    # ml_helpers.remove_infinity_from_csv(os.path.join(data_path, 'cic17_all_merged.csv'))
    # ml_helpers.shuffle_csv(os.path.join(data_path, 'cic17_all_merged-noInf.csv'))


    ### SUBSAMPLING ####
    # pickle_path = save_subsampled_data_to_pickle(full_path, 'sub1-std.pickle', label_mapping_dict, ml_data.cic17_csv_header_names, mode='1')
    # pickle_path = save_subsampled_data_to_pickle(full_path, 'sub2-std.pickle', label_mapping_dict, ml_data.cic17_csv_header_names, mode='2')
    # pickle_to_csv(subsampled_path_1)

    ### CLASS COUNT STATISTICS ###
    # count_cic17_classes(full_path, label_mapping_dict, csv_mode=True)
    # count_cic17_classes(subsampled_path_1, label_mapping_dict, csv_mode=True)
    # count_cic17_classes(subsampled_path_2, label_mapping_dict, csv_mode=True)


    ### SUPERVISED LEARNING ###
    # rf_experiment(merged_path, label_binary_mapping_dict)


    ### FEATURE SELECTION ###
    # cic17_feature_selection(full_path)
    # topk_features = get_union_of_topk_features(ml_data.all_attack_featurelists, 5)
    # dos_features = get_union_of_topk_features(ml_data.dos_attacks_featurelists, 5)
    # clustering_with_different_featuresubsets('K-means', full_path, ml_data.all_attack_featurelists, nr_features=5,
    #                                          nr_clusters=15, subsampling=None)


    ### TRAINING ###
    model_name = 'K-means'
    # model_name = 'DBSCAN'
    training_path = subsampled_path_2
    selected_columns = ml_data.cic17_without_zero_features_noDstPort + ['Label']

    model, _, X, Y = clustering(model_name=model_name, data_path=training_path, nr_clusters=15, selected_features=selected_columns,
                                label_mapping_dict=label_mapping_dict, categorical_feature_mapping=label_mapping_dict,
                                majority_classes_only=True, plot_manifold=False, pca_components=None, subsampling=None,
                                save_name='test', evaluate=True, train_classifier=True)

    ### PLOTS ###
    # elbow_curve(full_path, 30, ml_data.csv_header_names_noDstPort, subsampling=0.01, standardize=True)

    # X, Y, _, _ = ml_helpers.load_dataset(merged_path, 1.0, selected_feature_names=ml_data.cic17_without_zero_features_noDstPort + ['Label'],
    #                                      subsampling=None, categorical_feature_mapping=label_mapping_dict, standardize=True)
    # X, Y = log_subsampling(X, Y, 20, 1000)
    # knn_distance_plot(X, k=4)



    ##################################
    ### LOCKED SHIELDS EXPERIMENTS ###
    ##################################

    ls17_labelled_path = './data/FlowMeter/ls17_Dup15/all-traffic24-ordered.pcap_Flow-labelledT.csv'
    ls18_labelled_path = './data/FlowMeter/ls18_Dup15/ls18-all-traffic24.pcap_Flow-labelledT.csv'

    ls17_selected_columns = ml_data.RF_top10 + ['Label']
    ls17_label_mapping_dict = {'Label': {'Normal': 0.0, 'Malicious': 1.0}}


    # model, _, X, Y = clustering(model_name='K-means', data_path=ls17_labelled_path, nr_clusters=15,
    #                             selected_features=ls17_selected_columns,
    #                             label_mapping_dict=ls17_label_mapping_dict, majority_classes_only=True, plot_manifold=False,
    #                             pca_components=None, standardize=True,
    #                             subsampling=None, save_name='ls17-k15', evaluate=True, train_classifier=False)
    #
    # X_test, Y_test, _, _ = ml_helpers.load_dataset(ls18_labelled_path, 1.0, selected_feature_names=ls17_selected_columns,
    #                                categorical_feature_mapping=None, standardize=True, subsampling=None,
    #                                pca_components=None)
    #
    # model, majority_class_dict = load_model_and_majority_dict('./models/cic17/ls17-k15.pickle')
    #
    # load_model_and_predict(X_test, './models/cic17/ls17-k15.pickle', ls17_label_mapping_dict, Y_test=Y_test, plot_manifold=False, classifier_mode=False)
    # predict_by_distance_threshold(X_test, Y_test, './models/cic17/ls17-k15.pickle', ls17_label_mapping_dict, distance_threshold=1.0)
    #
    #
    #
    # model, _, X, Y = clustering(model_name='K-means', data_path=ls18_labelled_path, nr_clusters=15,
    #                             selected_features=ls17_selected_columns,
    #                             label_mapping_dict=ls17_label_mapping_dict, majority_classes_only=True, plot_manifold=False ,
    #                             pca_components=None,
    #                             subsampling=None, save_name='ls18-k15', evaluate=True, train_classifier=False)
    #
    # X_test, Y_test, _, _ = ml_helpers.load_dataset(ls17_labelled_path, 1.0, selected_feature_names=ls17_selected_columns,
    #                                categorical_feature_mapping=None, standardize=True, subsampling=None,
    #                                pca_components=None)
    #
    # load_model_and_predict(X_test, './models/cic17/ls18-k15.pickle', ls17_label_mapping_dict, Y_test=Y_test, plot_manifold=False, classifier_mode=False)
    # predict_by_distance_threshold(X_test, Y_test, './models/cic17/ls18-k15.pickle', ls17_label_mapping_dict, distance_threshold=1.0)

