"""
Master Thesis
Network Monitoring and Attack Detection

ml_clustering_helpers.py
Helper functions used to perform the unsupervised clustering experiments on the CICIDS2017 and the LS datasets.


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import os

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors

import ml_helpers
import ml_data
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import math
import time


def merge_all_csvs(data_path):
    """
    Function to merge all .csvs of the CICIDS2017 dataset into one big .csv file

    :param data_path: Directory holding the .csvs
    """
    monday = 'Monday-WorkingHours.pcap_ISCX.csv'
    tuesday = 'Tuesday-WorkingHours.pcap_ISCX.csv'
    wednesday = 'Wednesday-workingHours.pcap_ISCX.csv'
    thursday_1 = 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    thursday_2 = 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    friday_1 = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    friday_2 = 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    friday_3 = 'Friday-WorkingHours-Morning.pcap_ISCX.csv'

    ml_helpers.merge_csv_2(os.path.join(data_path, monday), os.path.join(data_path, tuesday), os.path.join('tmp.csv'), False)
    ml_helpers.merge_csv_2(os.path.join(data_path,'tmp.csv'), os.path.join(data_path, wednesday), os.path.join('tmp2.csv'), False)
    ml_helpers.merge_csv_2(os.path.join(data_path,'tmp2.csv'), os.path.join(data_path, thursday_1), os.path.join('tmp.csv'), False)
    ml_helpers.merge_csv_2(os.path.join(data_path,'tmp.csv'), os.path.join(data_path, thursday_2), os.path.join('tmp2.csv'), False)
    ml_helpers.merge_csv_2(os.path.join(data_path,'tmp2.csv'), os.path.join(data_path, friday_1), os.path.join('tmp.csv'), False)
    ml_helpers.merge_csv_2(os.path.join(data_path,'tmp.csv'), os.path.join(data_path, friday_2), os.path.join('tmp2.csv'), False)
    ml_helpers.merge_csv_2(os.path.join(data_path,'tmp2.csv'), os.path.join(data_path, friday_3), os.path.join('cic17_all_merged.csv'), False)


def elbow_curve(csv_path, k_range, selected_features, label_mapping_dict, subsampling=None, standardize=True, suffix=''):
    """
    Function to generate a K-means elbow curve plot, for selecting the optimal K value.

    :param csv_path: Dataset path
    :param k_range: Integer, with maximal k value --> range(1, k_range) models will be trained
    :param selected_features: Names of the selected features
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param subsampling: Value between 0 and 1 for uniform subsampling. None to disable.
    :param standardize: Set to True to enable standardization
    :param suffix: Suffix for the figure filenames
    """
    X, Y, _, _ = ml_helpers.load_dataset(csv_path, 1.0, selected_feature_names=selected_features,
                                         categorical_feature_mapping=label_mapping_dict, standardize=standardize)
    if subsampling != None:
        X, Y = ml_helpers.subsample(X, Y, subsampling)

    Ks = range(1, k_range)
    km = [KMeans(n_clusters=k, n_jobs=5, max_iter=500) for k in Ks]
    fitted = [km[i].fit(X) for i in range(len(km))]
    sse = [fitted[i].inertia_ for i in range(len(km))] # Sum of squared distances of samples to their closest cluster center.
                                                       # / sum of squared errors (SSE)

    plt.plot(Ks, sse)
    plt.title('K-means elbow curve')
    plt.xlabel('k')
    plt.ylabel('SSE')

    figname = 'k-means-elbow-{}'.format(suffix)
    plt.savefig('./figures/{}.png'.format(figname))
    plt.savefig('./figures/{}.eps'.format(figname), format='eps')
    plt.show()


def knn_distance_plot(X, k):
    """
    Function used to generate a K nearest neighbors distance plot. (used for DBSCAN parameter tuning)

    :param X: Data with the feature vectors (np.array)
    :param k: nr. neighbors K
    """
    print('fitting ...')
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    print('Done')
    distances, indices = nbrs.kneighbors(X)
    distanceDec = sorted(distances[:, k - 1], reverse=True)
    plt.plot(np.arange(len(X)), distanceDec)

    print('fitting ...')
    nbrs2 = NearestNeighbors(n_neighbors=k+100).fit(X)
    print('Done')
    distances2, indices2 = nbrs2.kneighbors(X)
    distanceDec2 = sorted(distances2[:, k+100 - 1], reverse=True)
    plt.plot(np.arange(len(X)), distanceDec2)
    plt.show()


def multiple_knn_distances_plot(X, k_min, k_max, nr_meas):
    """
    Function to generate multiple K-nn distance plots for different K values


    :param X: Data with the feature vectors (np.array)
    :param k_min: Start value for K
    :param k_max: Last Value for K
    :param nr_meas: Number of measurements in the specified K range
    """
    nbrs = NearestNeighbors(n_neighbors=k_max).fit(X)
    print('Fitting Done')
    distances, indices = nbrs.kneighbors(X)
    step = int(math.ceil((k_max-k_min)/float(nr_meas)))

    for k in range (k_min-1, k_max, step):
        distanceDec = sorted(distances[:, k], reverse=True)
        plt.plot(np.arange(len(X)), distanceDec, label=str(k+1))

    plt.legend()
    plt.show()


def get_union_of_topk_features(all_attack_featurelists, k):
    """
    This function gets the union of the top-k feature names contained of all the featurelists in all_attack_featurelists

    """
    union = set()
    for name, features in all_attack_featurelists.items():
        union = union.union(features[-k:])

    # print(union)
    return sorted(union)


def load_data_from_pickle(pickle_path):
    """
    Helper function to load a dataset from a .pickle file.

    :param pickle_path: Path of the pickle file
    :return: X, Y, column_names (names of X columns)
    """
    print('Loading data from {} ...'.format(pickle_path))
    X, Y, column_names = pickle.load(open(pickle_path, "rb"))

    return X, Y, column_names


def pickle_to_csv(pickle_path):
    """
    Helperfunction to convert a dataset stored as .pickle to .csv format

    :param pickle_path: Path of the pickle file
    """
    X, Y, all_columns = load_data_from_pickle(pickle_path)
    data = pd.DataFrame(np.column_stack((X, Y)), columns=all_columns)
    data.to_csv(os.path.splitext(pickle_path)[0] + '.csv', index=False)


def count_cic17_classes(data_path, label_mapping_dict, csv_mode=False):
    """
    Function to count the class counts in the CICIDS2017 dataset

    :param data_path: The path of the dataset. (Can be either .csv or .pickle format)
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param csv_mode: Set to true to generate output in .csv format
    """
    if '.csv' in data_path:
        X, Y, _, _ = ml_helpers.load_dataset(data_path, 1.0, selected_feature_names=['Label'],
                                             subsampling=None, categorical_feature_mapping=label_mapping_dict, standardize=False)
    elif '.pickle' in data_path:
        X, Y, _ = load_data_from_pickle(data_path)

    nr_samples = len(Y)
    classes_ids = list(set(Y))
    inv_label_mapping_dict = {v: k for k, v in label_mapping_dict['Label'].items()}
    classes_names = [inv_label_mapping_dict[class_id] for class_id in classes_ids]

    print('Counting classes in {} ...'.format(data_path))
    print('Tot. nr. of samples: {}'.format(nr_samples))
    for class_id, class_name in zip(classes_ids, classes_names):
        y_subset = Y[np.where(Y == class_id)]

        if csv_mode:
            print('{};{};{:.5f}'.format(class_name, len(y_subset), len(y_subset)*100.0/nr_samples))
        else:
            print('{} - {} ({}%)'.format(class_name, len(y_subset), len(y_subset)*100.0/nr_samples))


def get_cluster_assignment_counts(model, Y, label_mapping_dict, majority_classes_only=False):
    """
    Function to count how many samples of which class are assigned to the clusters in <model>

    :param model: Sklearn clusterer model object
    :param Y: Labels (of the data on which the model was trained on)
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param majority_classes_only: If True, print only the names of the majority classes for each cluster.
                                  If False, print all classes and the corresponding counts for each clusters
    :return: cluster_id_to_majority_name_dict
    """
    inv_label_mapping_dict = {v: k for k, v in label_mapping_dict['Label'].items()}
    Y_names = [inv_label_mapping_dict[y_val] for y_val in Y]
    Y_names = np.array(Y_names)

    cluster_ids = list(set(model.labels_))
    cluster_id_to_majority_name_dict = {}

    label_names_per_cluster = map(
        lambda x: pd.Series([Y_names[i] for i in range(len(model.labels_)) if model.labels_[i] == x]), cluster_ids)

    for nr, names in enumerate(label_names_per_cluster):
        if majority_classes_only:
            value_counts = names.value_counts()
            majority_class_name = value_counts.index.values[0]
            majority_class_count = value_counts[0]

            cluster_id_to_majority_name_dict[cluster_ids[nr]] = majority_class_name

            tot_nr_samples_of_this_class = len(Y_names[np.where(Y_names==majority_class_name)])
            print('{}. {} - {}/{} ({}%/{}%)'.format(nr, majority_class_name, majority_class_count, tot_nr_samples_of_this_class,
                                                names.value_counts()[0]*100.0/sum(names.value_counts()), names.value_counts()[0]*100.0/tot_nr_samples_of_this_class))
            # print(value_counts)
        else:
            value_counts = names.value_counts()
            majority_class_name = value_counts.index.values[0]
            majority_class_count = value_counts[0]

            cluster_id_to_majority_name_dict[cluster_ids[nr]] = majority_class_name

            print("Cluster {} labels:".format(nr))
            print(value_counts)
            print()

    return cluster_id_to_majority_name_dict


def train_classifier_from_clustering_labels(X, model, cluster_id_to_majority_name_dict, label_mapping_dict):
    """
    This function trains a Random Forest classifier, using the majority classes of the clusters in <model>
    as a ground truth.

    :param X: Training-data (feature vectors)
    :param model: Sklearn clusterer model object
    :param cluster_id_to_majority_name_dict: Mapping from cluster ID to the majority class name in this cluster
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :return: estimator
    """
    unique_cluster_ids = summarize_clusters_with_same_majority_class(model.labels_, cluster_id_to_majority_name_dict)

    Y = [label_mapping_dict['Label'][cluster_id_to_majority_name_dict[c_id]] for c_id in unique_cluster_ids] # map cluster ID to class ID

    estimator = RandomForestClassifier(n_jobs=6, random_state=0)

    start = time.time()
    print('Training classifier with {} samples...'.format(len(Y)))
    estimator.fit(X, Y)
    end = time.time()
    print('Training took: {}'.format(end - start))

    return estimator


def evaluate_clustering_model_on_classes(Y, model_labels, cluster_id_to_majority_name_dict, label_mapping_dict):
    """
    Function to evaluate the found clusters using the ground truth.
    Reports precision and recall for all classes.

    :param Y: Labels (of the data on which the model was trained on)
    :param model_labels: model.labels_ attribute of sklearns clusterer
    :param cluster_id_to_majority_name_dict: Mapping from cluster ID to the majority class name in this cluster
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    """
    unique_cluster_ids = summarize_clusters_with_same_majority_class(model_labels, cluster_id_to_majority_name_dict)

    Y_pred = [label_mapping_dict['Label'][cluster_id_to_majority_name_dict[c_id]] for c_id in unique_cluster_ids]

    target_names = sorted(label_mapping_dict['Label'], key=label_mapping_dict['Label'].get)
    print(classification_report(Y, Y_pred, target_names=target_names))


def summarize_clusters_with_same_majority_class(model_labels, cluster_id_to_majority_name_dict):
    """
    Helper function for evaluate_clustering_model_on_classes()

    unique_cluster_ids: if e.g. portscan has 3 clusters with IDS [3, 7, 5] --> unique_cluster_ids will only contain one of these IDs, e.g. [3,3,3]
    Caution: don't use this for classification, after this you should map [3, 3, 3] to the corresponding class ID using label_mapping_dict

    :param model_labels: model.labels_ attribute of sklearns clusterer
    :param cluster_id_to_majority_name_dict: contains mapping from cluster ID to the class name of the most prominent class in this cluster
    :return: unique_cluster_ids
    """

    cluster_names = set()
    unique_cluster_ids = []
    name_to_id_dict = {}
    for cluster_id in model_labels:
        cluster_name = cluster_id_to_majority_name_dict[cluster_id]
        if cluster_name in cluster_names:
            unique_cluster_ids.append(name_to_id_dict[cluster_name])
        else:
            cluster_names.add(cluster_name)
            unique_cluster_ids.append(cluster_id)
            name_to_id_dict[cluster_name] = cluster_id

    return unique_cluster_ids


def getDistanceByPoint(X, model, cluster_centers):
    """
    Calculate euclidian distances between all points in <X> and all <cluster_centers>

    :param X: see above
    :param model: Sklearn clusterer model object
    :param cluster_centers: see above
    :return: pd.Series() object containing all distances
    """
    distances = pd.Series()
    for i in range(0,len(X)):
        a = X[i]
        b = cluster_centers[model.labels_[i]-1]
        distances.set_value(i, np.linalg.norm(a-b)) # euclidian distance
    return distances


def load_model_and_majority_dict(model_path):
    """
    Helper function to load a sklearn model and the corresponding cluter-ID->MajorityClass mapping stored in .pickle format

    :param model_path: Path to the model
    :return: model, ID->class mapping dict
    """
    model_and_dict = ml_helpers.load_model(model_path)

    return model_and_dict['model'], model_and_dict['majority_class_dict']


def load_model_and_predict(X_test, model_path, label_mapping_dict, Y_test=None, plot_manifold=False, classifier_mode=False):
    """
    Load a model from <model_path> and run predictions on a testset <X_test>

    :param X_test: see above
    :param model_path: see above
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param Y_test: Ground truth labels for X_test
    :param plot_manifold: Plot T-SNE if True
    :param classifier_mode: True if we use a classifier model trained on the clusters. Else use clusterer prediction functions
    :return: Predictions
    """
    from ml_clustering import dbscan_predict
    model, majority_class_dict = load_model_and_majority_dict(model_path)

    print('Performing predictions ...')
    start = time.time()
    if hasattr(model, 'predict'):
        Y_pred = model.predict(X_test)
        # Y_pred = kmeans_predict(model, X_test)
    else:
        Y_pred = dbscan_predict(model, X_test)
    end = time.time()
    print('Took {}s'.format(end - start))

    if Y_test is not None:
        Y_test = Y_test.astype(int)
        if classifier_mode:
            target_names = sorted(label_mapping_dict['Label'], key=label_mapping_dict['Label'].get)
            print(classification_report(Y_test, Y_pred, target_names=target_names))
        else:
            evaluate_clustering_model_on_classes(Y_test, Y_pred, majority_class_dict, label_mapping_dict)

    if plot_manifold:
        ml_helpers.plot_manifold(X_test, Y_pred, method_name='TSNE', y_to_name_mapping=majority_class_dict,
                                 colors=ml_data.color_map)

    return Y_pred


def get_cluster_center_distances(model):
    """
    Calculates a distance matrix, containing all pairwise distances between the centroids of <model>

    :param model: Sklearn clusterer model object
    :return: distance_matrix
    """
    centroids = model.cluster_centers_

    distance_matrix = np.zeros((len(centroids), len(centroids)))
    all_distances = []

    for m, centroid_1 in enumerate(centroids):
        for n, centroid_2 in enumerate(centroids):
            if m>=n:
                continue # matrix would be symetric, so calculate only half below the diagonal
            distance_matrix[m][n] = np.linalg.norm(centroid_1-centroid_2, axis=0)
            all_distances.append(distance_matrix[m][n])

    return distance_matrix


def plot_TSNE(X, Y, model_path, label_mapping_dict, plot_cluster_centers=False, suffix=''):
    """
    Function for T-SNE visualizations of a dataset X, Y.
    Color the clusters in the data using ground truth Y, and the cluster assignments of the model in <model_path>

    :param X: Dataset - Feature vectors
    :param Y: Dataset - Labels
    :param model_path: see above
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param plot_cluster_centers:
    :param suffix: Filename suffix for the stored plot files
    """
    inv_label_mapping_dict = {v: k for k, v in label_mapping_dict['Label'].items()}

    model, cluster_id_to_majority_name_dict = load_model_and_majority_dict(model_path)
    unique_cluster_ids = summarize_clusters_with_same_majority_class(model.labels_, cluster_id_to_majority_name_dict)

    cluster_centers = None
    if plot_cluster_centers:
        if hasattr(model, 'cluster_centers_'):
            cluster_centers = model.cluster_centers_
        elif hasattr(model, 'components_'):
            cluster_centers = model.components_


    ml_helpers.plot_manifold(X, unique_cluster_ids, method_name='TSNE', y_to_name_mapping=cluster_id_to_majority_name_dict,
                             cluster_centers=cluster_centers, colors=ml_data.color_map, suffix='unique_clusterIDs'+suffix)
    # ml_helpers.plot_manifold(X, model.labels_, method_name='TSNE', y_to_name_mapping=cluster_id_to_majority_name_dict, colors=ml_data.color_map, suffix='all_clusterIDs')
    ml_helpers.plot_manifold(X, Y, method_name='TSNE', y_to_name_mapping=inv_label_mapping_dict, colors=ml_data.color_map, suffix='GNDTruth'+suffix)


def manifold_plots(pickle_path, model_path, label_mapping_dict):
    """
    Helper function to generate T-SNE visualizations of a dataset stored in .pickle format

    :param pickle_path: Path of the dataset
    :param model_path: see plot_TSNE()
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    """
    X_sub, Y_sub, all_columns = load_data_from_pickle(pickle_path)
    selected_feature_indices = ml_helpers.get_feature_indices_by_name(ml_data.cic17_without_zero_features_noDstPort, all_columns)
    X_sub = X_sub[:, selected_feature_indices]

    plot_TSNE(X_sub, Y_sub, model_path, label_mapping_dict, plot_cluster_centers=True, suffix='-k30')
    plot_TSNE(X_sub, Y_sub, model_path, label_mapping_dict, plot_cluster_centers=False, suffix='-k30')


def load_model_and_plot_manifold(csv_path, model_path, selected_features, label_mapping_dict, subsampling):
    """
    Helper function to generate T-SNE visualizations of a dataset stored in .csv format

    :param csv_path: Path of the dataset
    :param model_path: see plot_TSNE()
    :param selected_features: Names of the selected features to be loaded from the .csv
    :param label_mapping_dict: Dictionary that maps the label strings to numbers
    :param subsampling: Subsampling factor
    """
    X, Y, _, _ = ml_helpers.load_dataset(csv_path, 1.0, selected_feature_names=selected_features,
                                         categorical_feature_mapping=label_mapping_dict, standardize=True)

    model, _ = load_model_and_majority_dict(model_path)

    nr_clusters = len(model.cluster_centers_ )
    ml_helpers.plot_manifold(X, model.labels, method_name='TSNE', subsampling=subsampling)

    inv_label_mapping_dict = {v: k for k, v in label_mapping_dict['Label'].items()}

    ml_helpers.plot_manifold(X, Y, method_name='TSNE', y_to_name_mapping=inv_label_mapping_dict, subsampling=subsampling)