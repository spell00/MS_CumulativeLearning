#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import pandas as pd
import numpy as np

from src.utils.utils import split_train_test
from src.utils.dataset import ms_data
from src.models.sklearn.ordination import pca


# TODO validate, display valid data differently to see if they look distant
def PCA(get_data_function, data_file):
    # labels = pd.read_csv(labels_file, header=0).loc[0]
    data, labels, samples = get_data_function(data_file)
    data[np.isnan(data)] = 0
    pca(data, labels, 'all')
    if 'Normal' in labels:
        for i, label in enumerate(labels):
            if label != 'Normal':
                labels[i] = 'Not Normal'
                # categories[i] = 1
            # else:
            # categories[i] = 0
    categories = pd.Categorical(labels).codes

    pca(data, categories, 'binary')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--intensities_csv", type=str, default="data\\canis_intensities.csv",
                        help="Path to intensities csv file")
    args = parser.parse_args()

    PCA(ms_data, args.intensities_csv)
