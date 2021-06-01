#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import os
import json
from src.models.sklearn.parameters import *
from src.utils.utils import split_labels_indices, split_train_test, getScalerFromString
from src.utils.dataset import ms_data
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from src.models.sklearn.ordination import pca

np.random.seed(42)

warnings.filterwarnings('ignore')

DIR = 'src/models/sklearn/'


def get_estimators_list():
    rf = RandomForestClassifier(max_depth=300, max_features=100, min_samples_split=300, n_estimators=100)
    gnb = GaussianNB()
    lr = LogisticRegression(max_iter=4000)
    lsvc = SVC(kernel='linear', probability=True)
    estimators_list = [('gnb', gnb),
                       ('lr', lr),
                       ('lsvc', lsvc)
                       ]
    return estimators_list


def train(models, args):
    data = ms_data(args.intensities_csv)
    labels = pd.read_csv(args.labels_csv, header=0).loc[0]
    for i, label in enumerate(labels):
        if label != 'Normal':
            labels[i] = 'Not Normal'

    categories = labels.astype("category").cat.codes
    labels = pd.concat((categories, labels), 1)
    labels.columns = ['category', 'label']
    nb_classes = len(set(labels))
    data[np.isnan(data)] = 0

    train_indices, test_indices = split_train_test(labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    x_test = data[test_indices]
    y_test = labels['category'][test_indices].tolist()
    all_x_train = data[train_indices]
    all_y_train = labels['category'][train_indices].tolist()

    samples, targets, categories, train_nums = get_samples_names(labels.index[train_indices])

    estimators_list = get_estimators_list()
    best_params = {}
    scaler = getScalerFromString('robust')()
    for name, (model, param_grid) in zip(models.keys(), models.values()):
        best_scores_train = []
        best_scores_valid = []
        for g in ParameterGrid(param_grid):
            best_score = np.inf
            for i, (inds, _) in enumerate(skf.split(train_nums, categories)):
                # Just plot the first iteration, it will already be crowded if doing > 100 optimization iterations
                print(f"CV: {i + 1}")
                new_nums = [train_nums[i] for i in inds]

                train_indices, valid_indices = split_labels_indices(labels, new_nums)

                x_train = data[train_indices]
                y_train = labels['category'][train_indices]
                x_valid = data[valid_indices]
                y_valid = labels['category'][valid_indices]
                timetrack = datetime.now().timetuple().tm_min
                print(datetime.now(), "Training", name, 'Looking for best hyperparameters')

                scaler.fit(x_train)
                x_train = scaler.transform(x_train)
                x_valid = scaler.transform(x_valid)
                dir_name = f"saved_models/sklearn/"
                os.makedirs(dir_name, exist_ok=True)
                pickle.dump(scaler, open(f"{dir_name}/scaler.sav", 'wb'))

                m = model()
                m.set_params(**g)
                m = m.fit(x_train, y_train)
                valid_score = m.score(x_valid, y_valid)
                score_train = m.score(x_train, y_train)
                score_valid = m.score(x_valid, y_valid)
                print('h_params:', g)
                best_scores_train += [score_train]
                best_scores_valid += [score_valid]

            # save if best
            if np.mean(best_scores_valid) < best_score:
                best_score = valid_score
                best_grid = g
        best_params[name] = best_grid
        best_model = model()
        best_model.set_params(**best_grid)
        best_model.fit(all_x_train, all_y_train)
        score_train = best_model.score(all_x_train, all_y_train)
        score_test = best_model.score(all_x_train, all_y_train)
        total_time = datetime.now().timetuple().tm_min - timetrack
        print(f"Best model\n"
              f"Train score: {score_train}, "
              f"Best Valid score: {score_test}, "
              f"Trained in {total_time} minutes"
              )
        os.makedirs(dir_name, exist_ok=True)
        filename = f"{dir_name}/{name}.sav"
        pickle.dump(model, open(filename, 'wb'))

        # TODO find best grid according to all cv iterations
        best_model = model()
        best_model.set_params(**best_grid)
        best_model.fit(X=all_x_train, y=all_y_train)
        score_test = best_model.score(x_test, y_test)
        best_params[name]['train_acc_mean'] = np.mean(best_scores_train)
        best_params[name]['train_acc_std'] = np.std(best_scores_train)
        best_params[name]['valid_acc_mean'] = np.mean(best_scores_valid)
        best_params[name]['valid_acc_std'] = np.std(best_scores_valid)
        best_params[name]['test_acc'] = score_test

        print(
            f"Best model\n"
            f"Train score: {np.mean(best_scores_train)} +- {np.std(best_scores_train)}\n"
            f"Valid score: {np.mean(best_scores_valid)} +- {np.std(best_scores_valid)}\n"
            f"Test score: {score_test}\n"
        )
    for name in best_params.keys():
        for param in best_params[name].keys():
            best_params[name][param] = str(best_params[name][param])

    json.dump(best_params, open('saved_models/sklearn/best_params.json', 'w'))


def final_train(models, args):
    print('Getting the data...')
    data = ms_data(args.intensities_file)
    labels = pd.read_csv(args.labels_file, header=0).loc[0]

    for name, model in zip(models.keys(), models.values()):
        print("Training", name)
        model.fit(X=data, y=labels)
        filename = f"results/sklearn/saved/{name}_finalized.sav"
        pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--scaler_str', type=str, default='robust')
    parser.add_argument('--bs', type=int, default=2048)
    parser.add_argument("--intensities_csv", type=str, default="data\\canis_intensities.csv",
                        help="Path to intensities csv file")
    parser.add_argument("--labels_csv", type=str, default="data\\canis_labels.csv",
                        help="Path to labels csv file")
    args = parser.parse_args()
    if args.verbose == 0:
        args.verbose = False
    else:
        args.verbose = True

    models = {
        "LogisticRegression": [LogisticRegression, param_grid_logreg],
        "BaggingClassifier": [BaggingClassifier, param_grid_bag],
        "LinearSVC": [LinearSVC, param_grid_linsvc],
        "SVCLinear": [SVC, param_grid_svc],
        "Gaussian_Naive_Bayes": [GaussianNB, {}],
        "SGDClassifier": [SGDClassifier, params_sgd],
        "KNeighbors": [KNeighborsClassifier, {}],
        # "AdaBoost_Classifier": [AdaBoostClassifier, param_grid_ada],
        "LDA": [LinearDiscriminantAnalysis, param_grid_lda],
        "QDA": [QuadraticDiscriminantAnalysis, param_grid_qda],
        "RandomForestClassifier": [RandomForestClassifier, param_grid_rf],
        # "Voting_Classifier": [VotingClassifier, param_grid_voting],
    }
    final_models = {
        "SVCLinear": SVC(max_iter=10000, kernel='linear', probability=True),
        "RandomForestClassifier": RandomForestClassifier(max_depth=300, max_features=100, min_samples_split=300,
                                                         n_estimators=100, class_weight='balanced',
                                                         criterion='entropy'),
        "Bagging_Classifier":
            BaggingClassifier(
                base_estimator=LinearSVC(max_iter=4000), n_estimators=100),
        "LogisticRegression": LogisticRegression(max_iter=10000, penalty='l2', class_weight='balanced'),
        "LinearSVC": GridSearchCV(estimator=LinearSVC(max_iter=4000), param_grid={}, n_jobs=-1, cv=5),
        # "Voting_Classifier": VotingClassifier(estimators=estimators_list, voting='hard')
    }
    train(models, args)
    # final_train(final_models, args)
