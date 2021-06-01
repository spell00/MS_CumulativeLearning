#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
from src.models.pytorch.cnn_pytorch import CNN

random.seed(42)


# scaling data
def ms_data(fname):
    from sklearn.preprocessing import minmax_scale
    mat_data = pd.read_csv(fname)
    labels = mat_data.index.values
    categories = [int(lab.split('_')[1]) for lab in labels]
    labels = [lab.split('_')[0] for lab in labels]
    mat_data = np.asarray(mat_data)
    mat_data = minmax_scale(mat_data, axis=0, feature_range=(0, 1))
    mat_data = mat_data.astype("float32")
    return mat_data, labels, categories


def resize_data_1d(data, new_size=(160,)):
    initial_size_x = data.shape[0]

    new_size_x = new_size[0]

    delta_x = initial_size_x / new_size_x

    new_data = np.zeros((new_size_x))

    for x, y, z in itertools.product(range(new_size_x)):
        new_data[x][y][z] = data[int(x * delta_x)]

    return new_data


class MSDataset(Dataset):
    def __init__(self, data, labels=None, transform=None, crop_size=-1, device='cuda'):
        self.device = device
        self.crop_size = crop_size
        self.samples = data
        self.transform = transform
        self.crop_size = crop_size
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        if self.crop_size != -1:
            max_start_crop = x.shape[1] - self.crop_size
            ran = np.random.randint(0, max_start_crop)
            x = torch.Tensor(x)[:, ran:ran + self.crop_size]  # .to(self.device)
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        if self.transform:
            x = self.transform(x)
        return x, label


def load_checkpoint(checkpoint_path,
                    model,
                    optimizer,
                    save,
                    predict,
                    input_shape,
                    name,
                    variant,
                    nb_classes
                    ):
    losses = {
        "train": [],
        "valid": [],
    }
    if name not in os.listdir(checkpoint_path) and not predict:
        print("Creating checkpoint...")
        if save:
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            learning_rate=None,
                            epoch=0,
                            checkpoint_path=checkpoint_path,
                            losses=losses,
                            input_shape=input_shape,
                            name=name,
                            model_name=CNN,
                            best_loss=None,
                            best_accuracy=None,
                            variant=variant,
                            nb_classes=nb_classes,
                            activation="relu",

                            )
    checkpoint_dict = torch.load(checkpoint_path + '/' + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    best_loss = checkpoint_dict['best_loss']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    try:
        losses_recon = checkpoint_dict['losses_recon']
        kl_divs = checkpoint_dict['kl_divs']
    except:
        losses_recon = None
        kl_divs = None

    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, epoch, losses, kl_divs, losses_recon, best_loss


def save_checkpoint(model,
                    optimizer,
                    learning_rate,
                    epoch,
                    checkpoint_path,
                    losses,
                    input_shape,
                    best_loss,
                    best_accuracy,
                    name="cnn",
                    variant="lecun",
                    nb_classes=2,
                    activation="relu",
                    model_name=CNN,
                    ):
    model_for_saving = model_name(input_shape=input_shape, nb_classes=nb_classes, variant=variant, activation=activation)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'losses': losses,
                'best_loss': best_loss,
                'best_accuracy': best_accuracy,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + '/' + name)

class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


class validation_spliter:
    def __init__(self, dataset, cv):
        self.cv = cv
        self.dataset = dataset
        self.current_cv = 0
        self.val_offset = int(np.floor(len(self.dataset) / self.cv))
        self.current_pos = 0

    def __next__(self):
        self.current_cv += 1
        # if self.current_cv == self.cv:
        #     val_offset = len(self.dataset) - self.current_pos
        # else:
        #     val_offset = self.val_offset
        partial_dataset = PartialDataset(self.dataset, 0, self.val_offset), \
                          PartialDataset(self.dataset, self.val_offset, len(self.dataset) - self.val_offset)

        # Move the samples currently used for the validation set at the end for the next split
        tmp = self.dataset.samples[:self.val_offset]
        self.dataset.samples = np.concatenate([self.dataset.samples[self.val_offset:], tmp], 0)

        return partial_dataset

