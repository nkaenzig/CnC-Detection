"""
Master Thesis
Network Monitoring and Attack Detection

ml_training.py
This module contains helper functions used for training our supervised models.


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    average_precision_score, precision_recall_curve
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.base import clone

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import time
import pickle

import ml_helpers


def plot_performance_metrics(performance_metrics, out_path, suffix):
    """
    Generates box plots of precision, recall and f1 values as well as precision-recall curves

    :param performance_metrics: Dictionary containing different performance metrics
    :param out_path: Directory to store the plots
    :param suffix: Suffix for the plot filenames
    :return: precision, recall, f1 (Mean values)
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    precision_n_scores = []
    recall_n_scores = []
    f1_n_scores = []
    mal_accuracies = []
    normal_accuracies = []
    average_precision_scores = []
    precision_recall_curves = []
    nr_pos_samples = []
    nr_neg_samples = []

    for metrics in performance_metrics:
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
        precision_n_scores.append(metrics['precision_n'])
        recall_n_scores.append(metrics['recall_n'])
        f1_n_scores.append(metrics['f1_n'])
        mal_accuracies.append(metrics['mal_accuracy'])
        normal_accuracies.append(metrics['normal_accuracy'])
        if 'average_precision' in metrics:
            average_precision_scores.append(metrics['average_precision'])
            precision_recall_curves.append(metrics['precision_recall_curve'])
            nr_pos_samples.append(metrics['nr_pos_samples'])
            nr_neg_samples.append(metrics['nr_neg_samples'])


    precision = (np.mean(precision_scores), np.std(precision_scores))
    recall = (np.mean(recall_scores), np.std(recall_scores))
    f1 = (np.mean(f1_scores), np.std(f1_scores))

    precision_n = (np.mean(precision_n_scores), np.std(precision_n_scores))
    recall_n = (np.mean(recall_n_scores), np.std(recall_n_scores))
    f1_n = (np.mean(f1_n_scores), np.std(f1_n_scores))


    # all_metrics = [mal_accuracies, normal_accuracies, precision_scores, recall_scores, f1_scores]
    print('--- Metrics of normal class ---')
    print('Precision (mean, std): {}'.format(precision_n))
    print('Recall (mean, std): {}'.format(recall_n))
    print('F1 (mean, std): {}\n'.format(f1_n))

    print('--- Metrics of malicious class ---')
    print('Precision (mean, std): {}'.format(precision))
    print('Recall (mean, std): {}'.format(recall))
    print('F1 (mean, std): {}'.format(f1))

    if len(average_precision_scores) > 0:
        average_precision = (np.median(average_precision_scores), np.mean(average_precision_scores), np.std(average_precision_scores))
        print('Average precision score/AUC (median, mean, std): {}'.format(average_precision))
        print('Average precision score for each experiment: {}'.format(', '.join([str(nr) + '-' + str(round(score, 4)) for nr, score in enumerate(average_precision_scores)])))

        pr_plots_path = os.path.join(out_path, 'pr_curves')
        os.makedirs(pr_plots_path)

        for nr, pr_curves in enumerate(precision_recall_curves):
            plt.figure()
            for label, pr_curve in pr_curves.items():
                if 'malicious' in label:
                    color = '#ff7f0e' # orange
                else:
                    color = '#1f77b4' # blue
                plt.plot(pr_curve[0], pr_curve[1], linestyle='--', marker='o', label=label, color=color)
            pos_baseline_value = nr_pos_samples[nr]/(nr_neg_samples[nr]+nr_pos_samples[nr])
            neg_baseline_value = nr_neg_samples[nr]/(nr_neg_samples[nr]+nr_pos_samples[nr])
            plt.plot([0, 1], [pos_baseline_value, pos_baseline_value], '--', lw=1, color='black', label='baseline (malicious)')
            plt.plot([0, 1], [neg_baseline_value, neg_baseline_value], '-.', lw=1, color='grey', label='baseline (normal)')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision_scores[nr]))
            plt.legend()
            plt.savefig(os.path.join(pr_plots_path, 'pr_curve-{}-{}.png'.format(suffix, nr)))
            plt.savefig(os.path.join(pr_plots_path, 'pr_curve-{}-{}.eps'.format(suffix, nr)), format='eps')
            plt.close()
    # boxplot algorithm comparison
    # While an average has traditionally been a popular measure of a mid-point in a sample, it has the disadvantage of
    # being affected by any single value being too high or too low compared to the rest of the sample. This is why a median
    # is sometimes taken as a better measure of a mid point.
    all_metrics = [precision_scores, recall_scores, f1_scores]
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(all_metrics)
    # plt.violinplot(all_metrics, showmeans=False, showmedians=True)
    # ax.set_xticklabels(['malicious acc', 'normal acc', 'precision', 'recall', 'f1'])
    ax.set_xticklabels(['precision', 'recall', 'f1'])
    plt.savefig(os.path.join(out_path, 'boxplot-{}.png'.format(suffix)))
    plt.savefig(os.path.join(out_path, 'boxplot-{}.eps'.format(suffix)), format='eps')
    plt.close()

    return precision, recall, f1


def performance_metrics(Y_gndtruth, Y_predicted, Y_scores=None, report=True):
    """
    Function to calculate precision, recall and f1 performance metrics.

    :param Y_gndtruth: Ground-truth values
    :param Y_predicted: Predicted values by model
    :param Y_scores: Some classifiers generate a score. E.g. probability in interval [0,1] --> used for precision-recall curves
    :param report: If True, print the calculated scores
    :return: calculated performance metrics (dictionary)
    """
    metrics = {}

    mal_indices = np.where(Y_gndtruth == 1)
    nr_mal_samples = len(Y_gndtruth[mal_indices])
    normal_indices = np.where(Y_gndtruth == 0)
    nr_normal_samples = len(Y_gndtruth[normal_indices])

    metrics['nr_pos_samples'] = nr_mal_samples
    metrics['nr_neg_samples'] = nr_normal_samples

    metrics['accuracy'] = accuracy_score(Y_gndtruth, Y_predicted)
    metrics['mal_accuracy'] = accuracy_score(Y_gndtruth[mal_indices], Y_predicted[mal_indices])
    metrics['normal_accuracy'] = accuracy_score(Y_gndtruth[normal_indices], Y_predicted[normal_indices])

    metrics['precision'] = precision_score(Y_gndtruth, Y_predicted)
    metrics['recall'] = recall_score(Y_gndtruth, Y_predicted)
    metrics['f1'] = f1_score(Y_gndtruth, Y_predicted)

    metrics['precision_n'] = precision_score(Y_gndtruth, Y_predicted, pos_label=0)
    metrics['recall_n'] = recall_score(Y_gndtruth, Y_predicted, pos_label=0)
    metrics['f1_n'] = f1_score(Y_gndtruth, Y_predicted, pos_label=0)

    if np.array(Y_scores).any() != None:
        metrics['average_precision'] = average_precision_score(Y_gndtruth, Y_scores[:,1]) # scores of malicious class for AUC calculation
        metrics['precision_recall_curve'] = {'malicious_class': precision_recall_curve(Y_gndtruth, Y_scores[:,1], pos_label=1), 'normal_class': precision_recall_curve(Y_gndtruth, Y_scores[:,0], pos_label=0)}

    if report:
        print('Accuracy: {} ({} Samples)'.format(metrics['accuracy'], len(Y_gndtruth)))
        print('Accuracy on malicious class: {} ({} Samples)'.format(metrics['mal_accuracy'], nr_mal_samples))
        print('Accuracy on normal class: {} ({} Samples)'.format(metrics['normal_accuracy'], nr_normal_samples))
        print(classification_report(Y_gndtruth, Y_predicted, target_names=['normal', 'malicious']))

    return metrics


def train_and_test(classifier, csv_train_path, csv_test_path, selected_feature_names=None, swap_traintest=False, cv_fold_and_repeat=None,
                   balance=False, standardize=True, shuffle=False, categorical_feature_mapping=None, one_hot=False, prob_scores=False):
    """
    Function to train and test a new model on specified ntrain and test sets

    :param classifier: sklearn object of the model (not yet fitted) to be used
    :param csv_train_path: .csv path of train data
    :param csv_test_path: .csv path of test data
    :param selected_feature_names: List of names of the selected features
    :param swap_traintest: If True, swap the specified train and test sets (i.e. use csv_test_path as train set)
    :param cv_fold_and_repeat: Set this to a tuple (k, n) to perform cross validation. (k=k-fold CV, n=number of repetitions)
    :param balance: Set True to balance the data. (#samples same for all classes)
    :param standardize: Set to True to standardize data, or privide a path to a .pickle file containing a stored standardizer
    :param shuffle: Set to True to activate shuffling for cross validation
    :param categorical_feature_mapping: Dictionary to map categorical features to numerical values (see doc of function ml_helper.load_dataset())
    :param one_hot: If categorical features are present, set this parameter to True to enable one hot encoding
    :param prob_scores: Set true to include score predictions (e.g. probabilities) into reported performance metrics
                        --> necessary for precision-recall curves
    :return:
    """
    if cv_fold_and_repeat != None:
        if csv_test_path != None:
            X, Y = ml_helpers.load_dataset_seperate(csv_train_path, csv_test_path, selected_feature_names=selected_feature_names,
                                                balance=balance, standardize=standardize, merge=True, categorical_feature_mapping=categorical_feature_mapping, one_hot=one_hot)
        else:
            X, Y, _, _ = ml_helpers.load_dataset(csv_train_path, 1.0, selected_feature_names=selected_feature_names, balance=balance,
                         standardize=standardize, categorical_feature_mapping=categorical_feature_mapping, one_hot=one_hot)


        kf = RepeatedKFold(n_splits=cv_fold_and_repeat[0], n_repeats=cv_fold_and_repeat[1])

        # kf = KFold(n_splits=cv_fold_and_repeat[0], shuffle=shuffle)
        kf.get_n_splits(X)

        metrics = []

        print('Cross Validation ...\n')
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            clf = clone(classifier)  # yields a new estimator with the same parameters that has not been fit on any data.
            clf.fit(X_train, Y_train)

            Y_predicted = clf.predict(X_test)

            metrics.append(performance_metrics(Y_test, Y_predicted, report=True))

        return metrics
        # scores = cross_val_score(clf, X_train, Y_train, cv=6)
    else:
        X_train, Y_train, X_test, Y_test  = ml_helpers.load_dataset_seperate(csv_train_path, csv_test_path, selected_feature_names=selected_feature_names,
                                                                            balance=balance, standardize=standardize, swap_traintest=swap_traintest,
                                                                            categorical_feature_mapping=categorical_feature_mapping, one_hot=one_hot)

        print('Training the model ...')
        start = time.time()
        classifier.fit(X_train, Y_train)
        end = time.time()
        train_time = end - start
        print('... Training took {}s'.format(train_time))

        print('Performing predictions on testset ...')
        start = time.time()
        Y_predicted = classifier.predict(X_test)
        end = time.time()
        inference_time = end - start
        print('... Inference took {}s'.format(inference_time))

        if prob_scores:
            Y_scores = classifier.predict_proba(X_test)
            metrics = performance_metrics(Y_test, Y_predicted, Y_scores, report=True)
        else:
            metrics = performance_metrics(Y_test, Y_predicted, report=True)

        metrics['train_time'] = train_time
        metrics['inference_time'] = inference_time
        metrics['nr_train_samples'] = len(Y_train)
        metrics['nr_test_samples'] = len(Y_test)
        return metrics


def train_and_test_multiple_sets(classifier, datasets_path, selected_feature_names=None, cv_fold_and_repeat=None, shuffle=False,
                                 log_suffix='', balance=False, standardize=True, nr_files_to_evaluate=None,
                                 categorical_feature_mapping=None, one_hot=False, prob_scores=False):
    """
    See doc of train_and_test(). Runs train_and_test on multiple datasets.

    :param datasets_path: Path of directory containing the .csv train and test sets
    :return: precision, recall, f1, log_dir
    """
    timestamp = str(int(datetime.now().timestamp()))
    log_suffix = '-'.join([log_suffix, timestamp])
    log_filename = 'results-{}.txt'.format(log_suffix)
    log_dir = os.path.join(datasets_path, 'logs', '{}'.format(log_suffix))
    os.makedirs(log_dir)

    # sys.stdout = open(os.path.join(datasets_path, log_filename), 'w') # write all print calls to a log file
    orig_stdout = sys.stdout
    sys.stdout = ml_helpers.Logger(os.path.join(log_dir, log_filename))  # write all print calls to a log file

    train_files, test_files = ml_helpers.get_train_and_test_filenames(datasets_path)
    if nr_files_to_evaluate:
        print('Evaluating {} of the {} train/test sets'.format(nr_files_to_evaluate, len(train_files)))
        train_files, test_files = train_files[:nr_files_to_evaluate], test_files[:nr_files_to_evaluate]

    performance_metrics = []
    total_train_time = 0
    total_inference_time = 0
    tot_nr_train_samples = 0
    tot_nr_test_samples = 0

    if cv_fold_and_repeat != None:
        # take just one of the datasets (doesn't matter as they will be merged (and shuffled))
        performance_metrics = train_and_test(classifier, os.path.join(datasets_path, train_files[0]), os.path.join(datasets_path, test_files[0]),
                                             selected_feature_names, cv_fold_and_repeat=cv_fold_and_repeat, shuffle=shuffle,
                                             categorical_feature_mapping=categorical_feature_mapping, one_hot=one_hot, standardize=standardize)
    else:
        for train_file, test_file in zip(train_files, test_files):
            print('Processing: {}, {}'.format(train_file, test_file))
            clf = clone(classifier) # yields a new estimator with the same parameters that has not been fit on any data.
            metrics = train_and_test(clf, os.path.join(datasets_path, train_file), os.path.join(datasets_path, test_file),
                                                 selected_feature_names, balance=balance, standardize=standardize, swap_traintest=False,
                                                 categorical_feature_mapping=categorical_feature_mapping, one_hot=one_hot, prob_scores=prob_scores)
            performance_metrics.append(metrics)
            total_train_time += metrics['train_time']
            total_inference_time += metrics['inference_time']
            tot_nr_train_samples += metrics['nr_train_samples']
            tot_nr_test_samples += metrics['nr_test_samples']

            clf = clone(classifier)
            metrics = train_and_test(clf, os.path.join(datasets_path, train_file), os.path.join(datasets_path, test_file),
                                                 selected_feature_names, balance=balance, standardize=standardize, swap_traintest=True,
                                                 categorical_feature_mapping=categorical_feature_mapping, one_hot=one_hot, prob_scores=prob_scores)
            performance_metrics.append(metrics)
            total_train_time += metrics['train_time']
            total_inference_time += metrics['inference_time']
            tot_nr_train_samples += metrics['nr_train_samples']
            tot_nr_test_samples += metrics['nr_test_samples']

        print('Total train time: {}s | Traintime/sample: {}us'.format(total_train_time, total_train_time*1E6/tot_nr_train_samples))
        print('Total inference time: {}s | Inferencetime/sample: {}us'.format(total_inference_time, total_inference_time*1E6/tot_nr_test_samples))
    precision, recall, f1 = plot_performance_metrics(performance_metrics, log_dir, log_suffix)
    sys.stdout.close()
    sys.stdout = orig_stdout

    return precision, recall, f1, log_dir


def train_model_and_save(model, models_directory, model_name, selected_features, csv_train_path, csv_test_path=None,
                         balanced=False, standardize=True, categorical_feature_mapping=None, pca_components=None, subsampling=None):
    """
    Function to train a new model and save it to a .pickle file afterwards.

    :param model: sklearn object of the model (not yet fitted) to be used
    :param models_directory: Directory to store the trainedmodel
    :param model_name: Filename of the new model
    :param selected_features: List of names of the selected features
    :param csv_train_path: .csv path of train data
    :param csv_test_path: .csv path of test data
    :param balanced: see doc of function ml_helper.load_dataset()
    :param standardize: see doc of function ml_helper.load_dataset()
    :param categorical_feature_mapping: see doc of function ml_helper.load_dataset()
    :param pca_components: see doc of function ml_helper.load_dataset()
    """

    if csv_test_path!=None:
        X, Y = ml_helpers.load_dataset_seperate(csv_train_path, csv_test_path, selected_feature_names=selected_features, merge=True,
                                                balance=balanced, standardize=standardize, categorical_feature_mapping=categorical_feature_mapping)
    else:
        if pca_components:
            X, Y, _, _, pca = ml_helpers.load_dataset(csv_train_path, selected_feature_names=selected_features, train_fraction=1, balance=balanced,
                                                 standardize=standardize, categorical_feature_mapping=categorical_feature_mapping,
                                                 pca_components=pca_components)
        else:
            X, Y, _, _ = ml_helpers.load_dataset(csv_train_path, selected_feature_names=selected_features, train_fraction=1, balance=balanced,
                                                      standardize=standardize, categorical_feature_mapping=categorical_feature_mapping,
                                                      pca_components=pca_components, subsampling=subsampling)
            pca = None

    print('Training the model ...')
    start = time.time()
    model.fit(X, Y)
    end = time.time()
    print('Took: {}s'.format(end - start))

    ml_helpers.save_model(model, models_directory, model_name)
    if pca != None:
        with open(os.path.join(models_directory, '{}-pca.pickle'.format(model_name)), 'wb') as handle:
            pickle.dump(pca, handle)


def train_and_predict(estimator, csv_train_path, csv_test_path, selected_feature_names=None, balance=False, subsampling=None,
                      standardize=True, categorical_feature_mapping=None, pca_components=None):
    """
    Function to train a new model and to run predictions on a test set afterwards

    :param estimator: sklearn object of the model (not yet fitted) to be used
    :param csv_train_path: .csv path of train data
    :param csv_test_path: .csv path of test data
    :param selected_feature_names: List of names of the selected features
    :param balance: see doc of function ml_helper.load_dataset()
    :param subsampling: see doc of function ml_helper.load_dataset()
    :param standardize: see doc of function ml_helper.load_dataset()
    :param categorical_feature_mapping: see doc of function ml_helper.load_dataset()
    :param pca_components: see doc of function ml_helper.load_dataset()
    """

    if csv_test_path == None:
        X_train, Y_train, X_test, Y_test = ml_helpers.load_dataset(csv_train_path, train_fraction=0.5, selected_feature_names=selected_feature_names,
                                                                   balance=balance, subsampling=subsampling, standardize=standardize,
                                                                   categorical_feature_mapping=categorical_feature_mapping)
    else:
        X_train, Y_train, X_test, Y_test = ml_helpers.load_dataset_seperate(csv_train_path, csv_test_path, selected_feature_names=selected_feature_names,
                                                      balance=balance, subsampling=subsampling, standardize=standardize,
                                                      categorical_feature_mapping=categorical_feature_mapping, pca_components=pca_components)

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
    performance_metrics(Y_test, Y_predicted, report=True)

    return Y_predicted


def train_and_test_multiple_sets_seperate(classifier, train_datasets_path, test_datasets_path, log_suffix='', show_plot=True):
    """
    Function to train multiple models using .csv files in one directory and to test these models on .csv files
    in another directory

    :param classifier: sklearn object of the model (not yet fitted) to be used
    :param train_datasets_path: directory containing the .csv train sets
    :param test_datasets_path: directory containing the .csv test sets
    :param log_suffix: suffix for the .log file name, where all the output of this function is logged.
    :param show_plot: Call plt.show to display the performance score visualizations if set to True
    """
    # onlyfiles = [f for f in os.listdir(datasets_path) if isfile(join(mypath, f))]
    train_files = []
    test_files = []

    timestamp = str(int(datetime.now().timestamp()))
    log_suffix = '-'.join([log_suffix, timestamp])
    log_filename = 'results-{}.txt'.format(log_suffix)
    # sys.stdout = open(os.path.join(datasets_path, log_filename), 'w') # write all print calls to a log file
    print('Writing logs to {}'.format(os.path.join(train_datasets_path, log_filename)))
    sys.stdout = ml_helpers.Logger(os.path.join(train_datasets_path, log_filename))  # write all print calls to a log file

    for f in os.listdir(train_datasets_path):
        if 'meta' not in f and f.endswith(".csv"):
            if 'train' in f:
                train_files.append(f)

    for f in os.listdir(test_datasets_path):
        if 'meta' not in f and f.endswith(".csv"):
            if 'test' in f:
                test_files.append(f)

    train_files = sorted(train_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    test_files = sorted(test_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

    performance_metrics = []

    for train_file, test_file in zip(train_files, test_files):
        print('Processing: {}, {}'.format(train_file, test_file))
        clf = clone(classifier) # yields a new estimator with the same parameters that has not been fit on any data.
        performance_metrics.append(train_and_test(clf, os.path.join(train_datasets_path, train_file), os.path.join(test_datasets_path, test_file), swap_traintest=False))

    plot_performance_metrics(performance_metrics, train_datasets_path, log_suffix, show_plot)


def train_and_save_multiple_models(estimator_list, csv_train_path, selected_features, name_suffix=None, balanced=False, standardize=True):
    """
    Function used to train multiple models on the same data.

    :param estimator_list: List of the sklearn estimator objects to be trained
    :param csv_train_path: Path to the train data in .csv format
    :param selected_features: List of the features to be used for training
    :param name_suffix: Suffix for the model filenames
    :param balanced: See ml_helpers.load_dataset()
    :param standardize: ml_helpers.load_dataset()
    """
    nr_selected_features = len(selected_features)-1

    for name, clf in estimator_list.items():
        print('Training: {}'.format(name))
        if balanced:
            if standardize:
                model_name = '{}-{}feat-balanced-std'.format(name,nr_selected_features)
            else:
                model_name = '{}-{}feat-balanced'.format(name,nr_selected_features)
        else:
            model_name = '{}-{}feat'.format(name,nr_selected_features)

        if name_suffix:
            model_name += '-{}'.format(name_suffix)
        model_name += '.sav'

        train_model_and_save(clf, './models', model_name=model_name,
                                     selected_features=selected_features,
                                     csv_train_path=csv_train_path, balanced=balanced, standardize=standardize)


def random_forrest_parameter_optimization(csv_train_path, csv_test_path):
    """
    Helper function used, to test different parameters of random forest models

    :param csv_train_path:
    :param csv_test_path:
    """
    # nr_estimators = [10, 100, 200, 300, 400, 500]
    nr_estimators =[5, 10, 15, 20, 25, 30, 35, 40]
    min_samples_leaf = np.arange(1,11)
    # max_features = np.arange(1,21)
    parameters = nr_estimators
    # parameters = max_features

    for i in range(len(parameters)):
        clf = RandomForestClassifier(n_jobs=6, n_estimators=parameters[i], random_state=0)
        # clf = RandomForestClassifier(n_jobs=6, max_features=parameters[i], random_state=0)
        start = time.time()
        # train_and_test_multiple_sets(clf, dataset_path, log_suffix='', show_plot=False)
        train_and_test(clf, csv_train_path, csv_test_path)
        end = time.time()
        print('Round {} took: {}\nParameter: {}'. format(i, end - start, parameters[i]))