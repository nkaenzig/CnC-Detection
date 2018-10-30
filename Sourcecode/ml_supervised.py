"""
Master Thesis
Network Monitoring and Attack Detection

ml_supervised.py
This module contains the functions we used to perform the supervised experiments to detect C&C sessions


@author: Nicolas Kaenzig, D-ITET, ETH Zurich
"""

import csv
from collections import Counter

from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pickle
import time
from tqdm import tqdm
import random
import itertools
import json

import ml_helpers
import ml_training
import ml_feature_selection
import ml_data



def intrusion_detection_system(model_path, csv_test_path, start_time=None, end_time=None, selected_features=None):
    """
    Demo function for running the proposed two-stage IDS system

    This function generates a list ordered by the dst-IPs/port tuple that are most popular among the flows classified as malicious.
    Also the src IP addresses that are associated with these IP/port tuples are added to the list.
    If the model works as intended, the most popular dst IPs should belong to the C&C servers owned by the red team whereas
    the listed src IPs connecting to them originate from the infected machines in our network.

    :param model_path: Path to the model to be used
    :param csv_test_path: Path to the test data in .csv format
    :param start_time: (Optional) List only alerts with a timestamp > start_time (end_time must also be defined)
    :param end_time: (Optional) List only alerts with a timestamp < end_time
    :param selected_features: List of features to be used (has to match the features the selected model has been trained with)
    """
    model = ml_helpers.load_model(model_path)

    print('Load data from .csv ...')
    if selected_features:
        readcols = selected_features + ['Timestamp', 'Src IP', 'Dst IP', 'Dst Port']
        df = pd.read_csv(csv_test_path, sep=',', usecols=readcols)
        X = df[selected_features].values[:, :-1]
        Y = df[selected_features].values[:, -1]
    else:
        df = pd.read_csv(csv_test_path, sep=',')
        X = df.values[:, :-1]
        Y = df.values[:, -1]

    print('Performing predictions ...')
    Y_predicted = model.predict(X)
    print(classification_report(Y, Y_predicted, target_names=['normal', 'malicious'])) # TODO: remove
    mal_indices = np.where(Y_predicted == 1)

    malicious_times_and_ips_and_port = df[['Timestamp', 'Src IP', 'Dst IP', 'Dst Port']].values[mal_indices]
    print('Nr. of mal predictions: {}'.format(len(malicious_times_and_ips_and_port)))

    # convert formated times to utc unix timestamps
    if start_time and end_time:
        start_time_dt = datetime.strptime(start_time, '%d/%m/%Y %H:%M:%S')
        start_time = start_time_dt.replace(tzinfo=timezone.utc).timestamp()
        end_time_dt = datetime.strptime(end_time, '%d/%m/%Y %H:%M:%S')
        end_time = end_time_dt.replace(tzinfo=timezone.utc).timestamp()

    mal_dst_ip_cnt = Counter()
    count_dict = {'mal_dst_ip_cnt': mal_dst_ip_cnt, 'assigned_src_ips': {}}

    # caution: FlowMeter generates times according to local timezone, while malicios_sessions stores UTC timeformat
    UTC_OFFSET = 2
    for timestamp, src_ip, dst_ip, dst_port in malicious_times_and_ips_and_port:
        if start_time and end_time:
            timestamp_dt = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S') - timedelta(hours=UTC_OFFSET)
            if timestamp_dt < start_time_dt or timestamp_dt > end_time_dt:
                continue
        else:
            key = '{}:{}'.format(dst_ip,dst_port)
            count_dict['mal_dst_ip_cnt'][key] += 1

            if key in count_dict['assigned_src_ips']:
                count_dict['assigned_src_ips'][key][src_ip] += 1
            else:
                count_dict['assigned_src_ips'][key] = Counter({src_ip: 1})

    # mal_ip_cnt.items()  # convert to a list of (elem, cnt) pairs
    # mal_ip_cnt.most_common(10)  # n least common elements
    most_common_mal_dsts = count_dict['mal_dst_ip_cnt'].most_common(None)  # n least common elements
    nr_mal_dsts = len(most_common_mal_dsts)

    for i in range(nr_mal_dsts):
        dst_ip_and_port = most_common_mal_dsts[nr_mal_dsts-1-i][0]
        cnt = most_common_mal_dsts[nr_mal_dsts-1-i][1]
        print('{}:\t{}'.format(cnt, dst_ip_and_port))
        print(count_dict['assigned_src_ips'][dst_ip_and_port].most_common())
        print()


def intrusion_detection_system_multiple_csv(model_path, csv_directory, selected_features, get_times=True):
    """
    Same function as intrusion_detection_system() but runs on multiple .csv in a specified directory. The results
    from the different .csv test files are accumulated.

    :param model_path: Path to the model to be used
    :param csv_directory: Directory holding the .csv test files
    :param selected_features: List of features to be used (has to match the features the selected model has been trained with)
    :param get_times: If True, prints the times when the first and the last malicious flows were observed
    """
    model = ml_helpers.load_model(model_path)
    mal_dst_ip_cnt = Counter()
    count_dict = {'mal_dst_ip_cnt': mal_dst_ip_cnt, 'assigned_src_ips': {}}
    filenames = os.listdir(csv_directory)
    start_time = float('Inf')
    end_time = 0

    for nr, filename in enumerate(tqdm(filenames, total=len(filenames), desc='Performing predictions')):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_directory, filename)
            # print('Load data from {} ...'.format(filename))

            if selected_features:
                readcols = selected_features + ['Timestamp', 'Src IP', 'Dst IP', 'Dst Port']
                df = pd.read_csv(filepath, sep=',', usecols=readcols)
                X = df[selected_features].values[:, :-1]
                # Y = df[selected_features].values[:, -1]
            else:
                df = pd.read_csv(filepath, sep=',')
                X = df.values[:, :-1]
                # Y = df.values[:, -1]

            Y_predicted = model.predict(X)
            # print(classification_report(Y, Y_predicted, target_names=['normal', 'malicious']))
            mal_indices = np.where(Y_predicted == 1)

            malicious_times_and_ips_and_port = df[['Timestamp', 'Src IP', 'Dst IP', 'Dst Port']].values[mal_indices]

            for time, src_ip, dst_ip, dst_port in malicious_times_and_ips_and_port:
                if get_times:
                    UTC_OFFSET = 2
                    timestamp_dt = datetime.strptime(time, '%d/%m/%Y %H:%M:%S') - timedelta(hours=UTC_OFFSET)
                    timestamp = timestamp_dt.replace(tzinfo=timezone.utc).timestamp()
                    if timestamp < start_time:
                        start_time = timestamp
                    if timestamp > end_time:
                        end_time = timestamp

                key = '{}:{}'.format(dst_ip, dst_port)
                count_dict['mal_dst_ip_cnt'][key] += 1

                if key in count_dict['assigned_src_ips']:
                    count_dict['assigned_src_ips'][key][src_ip] += 1
                else:
                    count_dict['assigned_src_ips'][key] = Counter({src_ip: 1})

    most_common_mal_dsts = count_dict['mal_dst_ip_cnt'].most_common(None)  # n least common elements
    nr_mal_dsts = len(most_common_mal_dsts)

    for i in range(nr_mal_dsts):
        dst_ip_and_port = most_common_mal_dsts[nr_mal_dsts - 1 - i][0]
        cnt = most_common_mal_dsts[nr_mal_dsts - 1 - i][1]
        print('{}:\t{}'.format(cnt, dst_ip_and_port))
        print(count_dict['assigned_src_ips'][dst_ip_and_port].most_common())
        print()

    if get_times and nr_mal_dsts>0:
        print('First malicious flow seen at {}'.format(datetime.fromtimestamp(start_time).strftime('%d/%m/%Y %H:%M:%S')))
        print('Last malicious flow seen at {}'.format(datetime.fromtimestamp(end_time).strftime('%d/%m/%Y %H:%M:%S')))



def evaluate_multiple_classifiers_by_host(classifiers, datasets_path, selected_features, nr_files_to_evaluate):
    """
    Function used to evaluate different classifiers on different datasets with varying host-splits.

    :param classifiers: List of sklearn estimators
    :param datasets_path: Directory holding the .csv train and test files
    :param selected_features: List of features to be used
    :param nr_files_to_evaluate: Specify how many of the datasets containd in datasets_path should be evaluated
    """
    precisions = []
    recalls = []
    f1s = []
    names = []
    for name, clf in classifiers.items():
        print('Evaluating: {}'.format(name))
        start = time.time()
        precision, recall, f1, log_dir = ml_training.train_and_test_multiple_sets(clf, datasets_path, selected_feature_names=selected_features,
                                                                                  log_suffix=name, balance=False, standardize=True,
                                                                                  nr_files_to_evaluate=nr_files_to_evaluate)
        end = time.time()
        print('Took: {}'.format(end - start))

        names.append(name)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print('##### SUMMARY #####')
    summary = ''
    for i, name in enumerate(names):
        summary += '{}\n'.format(name)
        summary += 'Precision (mean, std): {}\n'.format(precisions[i])
        summary += 'Recall (mean, std): {}\n'.format(recalls[i])
        summary += 'F1 (mean, std): {}\n\n'.format(f1s[i])

    with open(os.path.join(log_dir, 'summary_results.txt'), 'w') as fp:
        fp.write(summary)
        print(summary)


def generate_csv_with_attack(csv_path, feature_to_attack, new_values, suffix):
    """
    Function used to simulate different attacks on the model. The values of the specified features to be attacked
    will be changed to the new values passed to this function.

    :param csv_path: path to the dataset in .csv format
    :param feature_to_attack: the feature/column of the .csv that should be modified/attacked
    :param new_values: list of the new values for the attacked feature. If multiple values are given, for each sample,
                       one of the values is selected at random
    :param suffix: Suffix for the new modified .csv file
    :return: path to the new .csv file, containing the altered values
    """
    out_path = os.path.splitext(csv_path)[0] + '-{}{}.csv'.format(feature_to_attack.replace(" ", "").replace("/",""), suffix)

    with open(csv_path, 'r') as csv_r, open(out_path, 'w') as csv_w:
        csv_reader = csv.reader(csv_r, delimiter=',', quotechar='#')
        csv_writer = csv.writer(csv_w, delimiter=',', quotechar='#')

        header = csv_reader.__next__()

        feature_idx = header.index(feature_to_attack)
        label_idx = header.index('Label')

        csv_writer.writerow(header)

        for row in csv_reader:
            if row[label_idx] == '1':
                row[feature_idx] = random.choice(new_values)
            csv_writer.writerow(row)

    return out_path


def generate_csv_with_multiple_attacks(csv_path, features_to_new_values_dict, suffix=''):
    """
    Same purpose as generate_csv_with_attack() except that various features can be attacked at the same time.

    :param csv_path: path to the dataset in .csv format
    :param features_to_new_values_dict: Dictionary holding the names of the features to be attacked and the
                                        corresponding new values. {feature_name1: new_values1, feature_name2: new_values2, ...}
    :param suffix: Suffix for the new modified .csv file
    :return: path to the new .csv file, containing the altered values
    """
    feature_names = [name.replace(" ", "").replace("/", "") for name in features_to_new_values_dict.keys()]
    out_path = os.path.splitext(csv_path)[0] + '-'.join(feature_names) + suffix + '.csv'

    with open(csv_path, 'r') as csv_r, open(out_path, 'w') as csv_w:
        csv_reader = csv.reader(csv_r, delimiter=',', quotechar='#')
        csv_writer = csv.writer(csv_w, delimiter=',', quotechar='#')

        header = csv_reader.__next__()
        label_idx = header.index('Label')
        feature_idxs = []
        new_values_list = []

        for feature_name, new_values in features_to_new_values_dict.items():
            feature_idxs.append(header.index(feature_name))
            new_values_list.append(new_values)

        csv_writer.writerow(header)
        for row in csv_reader:
            if row[label_idx] == '1':
                for feature_idx, new_values in zip(feature_idxs, new_values_list):
                    row[feature_idx] = random.choice(new_values)
            csv_writer.writerow(row)

    return out_path


def get_all_attack_combinations(attack_dict):
    """
    Prints all the possible combinations of the keys in attack_dict.

    :param attack_dict: Dictionary holding the names of the features to be attacked and the corresponding new values,
                        such as used in generate_csv_with_multiple_attacks().
                        {feature_name1: new_values1, feature_name2: new_values2, ...}
    """
    feature_names = attack_dict.keys()
    for L in range(0, len(feature_names) + 1):
        for subset in itertools.combinations(feature_names, L):
            print(subset)


def print_feature_value_counts(data, feature_name, data_format='csv', top_k=None):
    """
    Prints the counts of the specified features in descending order for both the normal class and the malicious class
    samples seperately.

    :param data: Data where to count the feature values. Can be a path to a .csv file or a pandas dataframe
    :param feature_name: The name of the feature whose values should be counted
    :param data_format: Set to 'csv' if data is a path to a .csv file, or to 'df' if data is a pandas dataframe
    :param top_k: When printing the feature value counts, only print the top-k values.
    """
    if data_format == 'csv':
        usecols = [feature_name]+['Label']
        df =  pd.read_csv(data, sep=',', usecols=usecols)[usecols]
    elif data_format == 'df':
        df = data
    else:
        print('Invalid data_format specified. Options are: csv, df')
        return

    normal_samples = df[df['Label']==0]
    malicious_samples = df[df['Label']==1]

    nor_feature_values = normal_samples[feature_name].value_counts().keys().tolist()
    nor_counts = normal_samples[feature_name].value_counts().tolist()

    mal_feature_values = malicious_samples[feature_name].value_counts().keys().tolist()
    mal_counts = malicious_samples[feature_name].value_counts().tolist()

    print('NORMAL count - {}'.format(feature_name))

    nr = 0
    for nor_val, nor_cnt in zip(nor_feature_values, nor_counts):
        if nr == top_k:
            break
        print('{} - {}'.format(nor_cnt, nor_val))
        nr += 1

    print('\nMALICIOUS count - {}'.format(feature_name))
    nr = 0
    for mal_val, mal_cnt in zip(mal_feature_values, mal_counts):
        if nr == top_k:
            break
        print('{} - {}'.format(mal_cnt, mal_val))
        nr += 1



if __name__ == "__main__":
    ls17_labelled_path_Snap96 = './data/FlowMeter/ls17_Dup15/snap96/all-traffic24-ordered-snap96.pcap_Flow-labelled-intExt.csv'
    ls17_labelled_path = './data/FlowMeter/ls17_Dup15/all-traffic24-ordered.pcap_Flow-labelledT-intExt.csv'
    ls18_labelled_path = './data/FlowMeter/ls18_Dup15/ls18-all-traffic24.pcap_Flow-labelledT-intExt.csv'
    ls18_labelled_path_Snap96 = './data/FlowMeter/ls18_Dup15/subsampled/intExt/ls18-all-traffic24-snap96-sub1.0.pcap_Flow-labelled-intExt.csv'
    session_labelled_path = '/./data/FlowMeter/out_Dup300/all-traffic24-ordered.pcap_Flow-labelledSessT.csv'

    kdd_ls17_labelled_path = './data/KDD/ls17/kdd_features_ls17-labelled_established_states_only.csv'
    kdd_ls18_labelled_path = './data/KDD/ls18/kdd_features_ls18-labelled_established_states_only.csv'

    extracted_path17 = './extracted/ls17'
    extracted_path18 = './extracted/ls18'

    # Load malicious IP lists
    malicious_ips_path17 = os.path.join(extracted_path17, 'malicious_ips.json')
    malicious_ips_path18 = os.path.join(extracted_path18, 'malicious_ips.json')
    with open(malicious_ips_path17, 'r') as fp:
        malicious_ips_ls17 = json.load(fp)['malicious_ips']
    with open(malicious_ips_path18, 'r') as fp:
        malicious_ips_ls18 = json.load(fp)['malicious_ips']

    featurename_csv_path = './data/ls17_rfe_features_sorted_by_importance.txt'
    host_split_path = './data/host_train_test_IP_splits_2905.pickle'

    with open(host_split_path, 'rb') as fp:
        host_splits = pickle.load(fp)


    ##################
    ### ESTIMATORS ###
    ##################

    rf = RandomForestClassifier(n_jobs=6, random_state=0)
    rf_tuned = RandomForestClassifier(n_jobs=6, random_state=0, n_estimators=128, max_depth=10)
    svm = svm.LinearSVC(loss='hinge', random_state=0)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=4),

    all_classifiers = {
         'RandomForestClassifier(n_jobs=6, random_state=0)':  RandomForestClassifier(n_jobs=6, random_state=0),
         'GaussianNB()': GaussianNB(),
         'LogisticRegression(max_iter=1000, penalty=\'l2\', random_state=0, solver=\'sag\')': LogisticRegression(max_iter=1000, penalty='l2', random_state=0, solver='sag'),
         'neighbors.KNeighborsClassifier(n_neighbors=5, weights=\'uniform\', n_jobs=4)': neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=4),
         'LinearSVC(random_state=0, loss=\'hinge\')': svm.LinearSVC(loss='hinge', random_state=0)
    }


    ##########################
    ### DATASET GENERATION ###
    ##########################

    ## DATASET GENERATION FOR MODEL-SELECTION STEP-1
    ml_helpers.generate_multiple_sets_by_host(10, ls17_labelled_path, ml_data.all_features, subsampling_fraction=1,
                                              train_hosts_fraction=0.5, normal_traffic_fraction=0.5,
                                              malicious_hosts=ml_data.malicious_ips,
                                              suffix='H-splits2905-all-features', host_splits=host_splits)

    ml_helpers.generate_multiple_sets_by_host(10, session_labelled_path, ml_data.ls_all_features_ordered_by_importance + ['Label'],
                                              subsampling_fraction=1, train_hosts_fraction=0.5,
                                              normal_traffic_fraction=0.5, malicious_hosts=ml_data.malicious_ips,
                                              suffix='S-splits2905-all-features', host_splits=host_splits)

    ml_helpers.generate_multiple_sets_by_host(10, kdd_ls17_labelled_path, ml_data.kdd_all_features,
                                              subsampling_fraction=1, train_hosts_fraction=0.5,
                                              normal_traffic_fraction=0.5, malicious_hosts=malicious_ips_ls17,
                                              suffix='KDD-splits2905-all-features', host_splits=host_splits)


    #########################
    ### FEATURE SELECTION ###
    #########################

    ## LOCKED SHIELDS
    # | STEP-1
    X, Y, _, _ = ml_helpers.load_dataset(ls17_labelled_path, subsampling=0.3, train_fraction=1.0,
                                         selected_feature_names=ml_data.flowmeter_csv_header_names, standardize=False)
    X, feature_names = ml_helpers.eliminate_features(X, ml_data.constant_features, ml_data.flowmeter_csv_header_names)
    ml_feature_selection.MI_vs_Corr_vs_DC(X, Y, feature_names)
    # | STEP-2
    ml_feature_selection.get_features_ordered_by_score(X, Y, ml_data.kdd_nozero_features, 'FlowMeter')

    ## KDD
    # | STEP-1
    X, Y, _, _ = ml_helpers.load_dataset(kdd_ls17_labelled_path, train_fraction=1.0,
                                         selected_feature_names=ml_data.kdd_no_cat_features + ['Label'],
                                         standardize=True)
    ml_feature_selection.MI_vs_Corr_vs_DC(X, Y, ml_data.kdd_no_cat_features)
    X, Y, _, _ = ml_helpers.load_dataset(kdd_ls17_labelled_path, train_fraction=1.0,
                                         selected_feature_names=ml_data.kdd_no_cat_features + ['Label'],
                                         standardize=True,
                                         categorical_feature_mapping=ml_data.kdd_categorical_feature_mapping)

    # | STEP-2
    ml_feature_selection.get_features_ordered_by_score(X, Y, ml_data.kdd_nozero_features, 'KDD')


    ## CREATE FEATURE SELF CORRELATION MATRICES
    X, Y, _, _ = ml_helpers.load_dataset(ls17_labelled_path, train_fraction=1.0,
                                         selected_feature_names=ml_data.flowmeter_csv_header_names, standardize=False)
    ml_feature_selection.plot_feature_self_correlation_matrix(X, ml_data.ls_all_features_ordered_by_importance[-10:], size=20, fontsize=35)
    ml_feature_selection.plot_feature_self_correlation_matrix(X, ml_data.ls_all_features_ordered_by_importance, size=20, fontsize=35)

    ml_feature_selection.train_and_test_with_different_feature_subsets(all_classifiers, [58, 40, 30, 20, 15, 10, 5],
                                                                       featurename_csv_path, ls17_labelled_path,
                                                                       log_suffix='all-classifiers-different-feature-counts',
                                                                       nr_files_to_evaluate=None,
                                                                       prob_scores=False, standardize=True, balance=False)

    ml_feature_selection.train_and_test_with_different_feature_subsets(all_classifiers, [58, 40, 30, 20, 15, 10, 5],
                                                                       featurename_csv_path, ls17_labelled_path,
                                                                       log_suffix='all-classifiers-different-feature-counts',
                                                                       nr_files_to_evaluate=None,
                                                                       prob_scores=False, standardize=True, balance=True)


    #######################
    ### MODEL-SELECTION ###
    #######################

    ## MODEL-SELECTION STEP-1 EVALUATION
    host_datasets_path = './data/FlowMeter/out_Dup300/10sets-0.3-0.5-0.5-H-splits2905-all-features-1527862554'
    session_datasets_path = './data/FlowMeter/out_Dup300/10sets-1-0.5-0.5-S-splits2905-all-features-1534768068'

    ml_feature_selection.test_multiple_classifiers_on_different_feature_subsets(all_classifiers, featurename_csv_path,
                                                                                host_datasets_path,
                                                                                nr_files_to_evaluate=None,
                                                                                standardize=True, balance=True)

    ml_training.train_and_test_multiple_sets(rf, session_datasets_path, log_suffix='sessions-20feat',
                                             selected_feature_names=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                             balance=False, standardize=False)
    ml_training.train_and_test_multiple_sets(rf, session_datasets_path, standardize=False, balance=False,
                                             selected_feature_names=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                             cv_fold_and_repeat=(2, 10), shuffle=False,
                                             log_suffix='sessions-20feat-CVshuffle(2,10)')
    ml_training.train_and_test_multiple_sets(rf, host_datasets_path, log_suffix='hosts-20feat',
                                             selected_feature_names=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                             balance=False, standardize=False)
    ml_training.train_and_test_multiple_sets(rf, host_datasets_path, standardize=False, balance=False,
                                             selected_feature_names=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                             cv_fold_and_repeat=(2, 10), shuffle=False,
                                             log_suffix='host-20feat-CVshuffle(2,10)')
    ml_training.train_and_test_multiple_sets(rf, kdd_ls17_labelled_path, log_suffix='KDD-cat-RF', balance=False,
                                             standardize=False, selected_feature_names=ml_data.kdd_without_zerocor_features,
                                             categorical_feature_mapping=ml_data.kdd_categorical_feature_mapping,
                                             one_hot=False)


    ## ORIGINAL RANDOM FOREST MODELS
    ml_training.train_model_and_save(rf, './models', model_name='RF_ls17_RF_top10.sav',
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-10:] + ['Label'],
                                     csv_train_path=ls17_labelled_path, standardize=False, balanced=False)
    ml_training.train_model_and_save(rf, './models', model_name='RF_ls18_RF_top10.sav',
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-10:] + ['Label'],
                                     csv_train_path=ls18_labelled_path, standardize=False, balanced=False)

    ml_training.train_model_and_save(rf, './models', model_name='RF_ls17_RF_top20.sav',
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                     csv_train_path=ls17_labelled_path, standardize=False, balanced=False)
    ml_training.train_model_and_save(rf, './models', model_name='RF_ls18_RF_top20.sav',
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                     csv_train_path=ls18_labelled_path, standardize=False, balanced=False)

    ## TUNED RANDOM FOREST MODELS

    ml_training.train_model_and_save(rf_tuned, './models',
                                     model_name='RF_ls17-RF_top10_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
                                     selected_features=ml_data.RF_top10_noSYN_noPSH_withProtocol_intExt + ['Label'],
                                     csv_train_path=ls17_labelled_path, standardize=False, balanced=False)
    ml_training.train_model_and_save(rf_tuned, './models',
                                     model_name='RF_ls18-RF_top10_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
                                     selected_features=ml_data.RF_top10_noSYN_noPSH_withProtocol_intExt + ['Label'],
                                     csv_train_path=ls18_labelled_path, standardize=False, balanced=False)

    ml_training.train_model_and_save(rf_tuned, './models',
                                     model_name='RF_ls17-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
                                     selected_features=ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'],
                                     csv_train_path=ls17_labelled_path, standardize=False, balanced=False)
    ml_training.train_model_and_save(rf_tuned, './models',
                                     model_name='RF_ls18-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
                                     selected_features=ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'],
                                     csv_train_path=ls18_labelled_path, standardize=False, balanced=False)

    # | snaplen = 96 byte
    ml_training.train_model_and_save(rf_tuned, './models',
                                     model_name='RF_ls17-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10-snap96.sav',
                                     selected_features=ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'],
                                     csv_train_path=ls17_labelled_path_Snap96, standardize=False, balanced=False)
    ml_training.train_model_and_save(rf_tuned, './models',
                                     model_name='RF_ls18-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10-snap96.sav',
                                     selected_features=ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'],
                                     csv_train_path=ls18_labelled_path_Snap96, standardize=False, balanced=False)


    ## OTHER MODELS
    ml_training.train_model_and_save(knn, './models', model_name='KNN_ls18-15feat-std-bal.sav', subsampling=0.3,
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-15:] + ['Label'],
                                     csv_train_path=ls18_labelled_path, standardize=True, balanced=False)
    ml_training.train_model_and_save(knn, './models', model_name='KNN_ls17-15feat-std-bal.sav', subsampling=0.3,
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-15:] + ['Label'],
                                     csv_train_path=ls17_labelled_path, standardize=True, balanced=False)

    ml_training.train_model_and_save(svm, './models', model_name='SVM_ls18-20feat-std-bal.sav',
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                     csv_train_path=ls18_labelled_path, standardize=True, balanced=True)
    ml_training.train_model_and_save(svm, './models', model_name='SVM_ls17-20feat-std-bal.sav',
                                     selected_features=ml_data.ls_all_features_ordered_by_importance[-20:] + ['Label'],
                                     csv_train_path=ls17_labelled_path, standardize=True, balanced=True)

    ##########################
    ### SIMULATING ATTACKS ###
    ##########################

    ## LOOK AT THE MOST COMMON VALUE OF THE 10 BEST FEATURE TO IDENTIFY BEST ATTACK VALUES
    print_feature_value_counts(ls17_labelled_path, 'Init Fwd Win Byts', top_k=20)
    print_feature_value_counts(ls17_labelled_path, 'Active Mean', top_k=100)
    print_feature_value_counts(ls17_labelled_path, 'Bwd Pkt Len Min', top_k=20)
    print_feature_value_counts(ls17_labelled_path, 'Flow Pkts/s', top_k=100)
    print_feature_value_counts(ls17_labelled_path, 'Fwd IAT Max', top_k=20)
    print_feature_value_counts(ls17_labelled_path, 'Fwd IAT Mean', top_k=20)
    print_feature_value_counts(ls17_labelled_path, 'Subflow Fwd Pkts', top_k=20)
    print_feature_value_counts(ls17_labelled_path, 'FIN Flag Cnt', top_k=20)
    print_feature_value_counts(ls17_labelled_path, 'PSH Flag Cnt', top_k=20)

    ## GENERATING THE CSVS WITH THE SIMULATED ATTACKS
    generate_csv_with_multiple_attacks(ls17_labelled_path, ml_data.all_attacks_dict)

    generate_csv_with_attack(ls17_labelled_path, 'Flow IAT Mean', ml_data.Flow_IAT_Mean_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'Flow IAT Mean', ml_data.Flow_IAT_Mean_attack_values, '')

    generate_csv_with_attack(ls18_labelled_path, 'PSH Flag Cnt', ml_data.PSH_Flag_Cnt_attack_values, '')
    generate_csv_with_attack(ls17_labelled_path, 'PSH Flag Cnt', ml_data.PSH_Flag_Cnt_attack_values, '')

    generate_csv_with_attack(ls17_labelled_path, 'Subflow Fwd Pkts', ml_data.Subflow_Fwd_Pkts_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'Subflow Fwd Pkts', ml_data.Subflow_Fwd_Pkts_attack_values, '')

    generate_csv_with_attack(ls17_labelled_path, 'SYN Flag Cnt', ml_data.SYN_Flag_Cnt_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'SYN Flag Cnt', ml_data.SYN_Flag_Cnt_attack_values, '')

    generate_csv_with_attack(ls17_labelled_path, 'Flow Pkts/s', ml_data.Flow_Pkts_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'Flow Pkts/s', ml_data.Flow_Pkts_attack_values, '')

    generate_csv_with_attack(ls17_labelled_path, 'Fwd IAT Max', ml_data.Fwd_IAT_Max_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'Fwd IAT Max', ml_data.Fwd_IAT_Max_attack_values, '')

    generate_csv_with_attack(ls17_labelled_path, 'Bwd Pkt Len Min', ml_data.Bwd_Pkt_Len_Min_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'Bwd Pkt Len Min', ml_data.Bwd_Pkt_Len_Min_attack_values, '')

    generate_csv_with_attack(ls17_labelled_path, 'Init Fwd Win Byts', ml_data.Init_Fwd_Win_Byts_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'Init Fwd Win Byts', ml_data.Init_Fwd_Win_Byts_attack_values, '')

    generate_csv_with_attack(ls17_labelled_path, 'Active Mean', ml_data.Active_Mean_attack_values, '')
    generate_csv_with_attack(ls18_labelled_path, 'Active Mean', ml_data.Active_Mean_attack_values, '')


    ## EVALUATION OF ALL ATTACKS
    ml_helpers.load_model_and_evaluate_multiple(
        './models/RF_ls17-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
        './data/FlowMeter/ls18_Dup15/attacks', ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'],
        standardize=False)
    ml_helpers.load_model_and_evaluate_multiple(
        './models/RF_ls18-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
        './data/FlowMeter/ls17_Dup15/attacks', ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'],
        standardize=False)



    ##############################
    ### TESTING THE IDS SYSTEM ###
    ##############################

    intrusion_detection_system_multiple_csv('./models/final/RF_ls17.sav', '/mnt/data/LockedShields18/LS18/demo',
                                            selected_features=ml_data.ls_all_features_ordered_by_importance[-10:] + ['Label'])
    intrusion_detection_system_multiple_csv('./models/RF_ls17-RF_top20_noSYN_noPSH_withProtocol_intExt_128est.sav',
                                            '/mnt/data/LockedShields18/LS18/demo',
                                            selected_features=ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + [
                                                'Label'])

    ##################
    ### EVALUATION ###
    ##################

    ml_helpers.load_model_and_test('./models/RF_ls17-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
                                   ls18_labelled_path,
                                   ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'], standardize=False)
    ml_helpers.load_model_and_test('./models/RF_ls18-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav',
                                   ls17_labelled_path,
                                   ml_data.RF_top20_noSYN_noPSH_withProtocol_intExt + ['Label'], standardize=False)



    ml_helpers.get_rf_tree_nodeCnt_and_depth('./models/RF_ls17-RF_top20_noSYN_noPSH_withProtocol_intExt_128est_maxdepth10.sav')