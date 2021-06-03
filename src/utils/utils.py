#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import pandas as pd
import numpy as np
import torch


def get_labels(fname):
    meta = pd.read_excel(fname, header=0)
    toremove = pd.isnull(meta.values[:, 0])
    tokeep = [i for i, x in enumerate(toremove) if x == 0]

    meta = meta.iloc[tokeep, :]
    samples_classes = meta['Pathological type']
    classes = np.unique(samples_classes)

    return classes, samples_classes


def to_categorical(y, num_classes, dtype=torch.int):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes, dtype=dtype)[y]

"""
def get_samples_names(labels):
    samples = {s: [] for s in np.unique(labels['label'])}

    new_keys = []
    categories = []
    nums = []
    for i, label in enumerate(samples.keys()):
        tmp = label.split('-')
        lab = tmp[0].split('c..')[1]
        num = tmp[1]
        cat = 0
        if lab != 'Normal':
            cat = 1
            lab = 'Not Normal'
        new_keys += [f'{lab}-{num}']
        categories += [cat]
        if num not in nums:
            nums += [int(num)]
    # samples = dict(zip(new_keys, list(samples.values())))

    return categories, nums
"""

def split_labels_indices(labels, train_inds):
    train_indices = []
    test_indices = []
    for j, sample in enumerate(list(labels)):
        if sample in train_inds:
            train_indices += [j]
        else:
            test_indices += [j]

    assert len(test_indices) != 0
    assert len(train_indices) != 0

    return train_indices, test_indices


def split_train_test(labels):
    from sklearn.model_selection import StratifiedKFold
    # First, get all unique samples and their category
    unique_samples = []
    unique_cats = []
    for sample, cat in zip(labels['sample'], labels['category']):
        if sample not in unique_samples:
            unique_samples += [sample]
            unique_cats += [cat]

    # StratifiedKFold with n_splits of 5 to ranmdomly split 80/20.
    # Used only once for train/test split.
    # The train split needs to be split again into train/valid sets later
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    train_inds, test_inds = next(skf.split(unique_samples, unique_cats))

    # After the samples are split, we get the duplicates of all samples.
    train_samples, test_samples = [unique_samples[s] for s in train_inds], [unique_samples[s] for s in test_inds]
    train_cats = [unique_cats[ind] for ind in train_inds]

    assert len(unique_samples) == len(train_inds) + len(test_inds)
    assert len([x for x in test_inds if x in train_inds]) == 0
    assert len([x for x in test_samples if x in train_samples]) == 0
    assert len(np.unique([unique_cats[ind] for ind in test_samples])) > 1

    return train_samples, test_samples, train_cats


def getScalerFromString(scaler_str):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, StandardScaler
    if str(scaler_str) == 'normalizer':
        scaler = Normalizer
    elif str(scaler_str) == 'standard':
        scaler = StandardScaler
    elif str(scaler_str) == 'minmax':
        scaler = MinMaxScaler
    elif str(scaler_str) == "robust":
        scaler = RobustScaler
    else:
        exit(f"Invalid scaler {scaler_str}")
    return scaler

