#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import torch
from torch import nn
from keras.models import Sequential
from keras.layers import Dense, Layer, Flatten, Conv1D, Dropout, BatchNormalization, MaxPooling1D, LeakyReLU


class CNN(torch.nn.Module):
    def __init__(self, nb_classes, input_shape, variant='lecun', activation='relu'):
        super(CNN, self).__init__()
        self.variant = 'lecun'
        self.nb_classes = nb_classes
        self.shape = input_shape
        self.variant = variant
        self.activation = activation
        self.model = None
        self.build(variant)
        self.random_init()

    def build(self, variant):
        if variant == 'lecun':
            self.lecun()
        elif variant == 'lenet':
            self.lenet()
        elif variant == 'vgg9':
            self.vgg9()
        else:
            exit(f'Model {variant} unrecognized.\n Accepted values: lecun, lenet and vgg9')

    def lecun(self):
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(1, 6, kernel_size=21, stride=21)
        self.bn1 = nn.BatchNorm1d(6)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=11, stride=11)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 256, kernel_size=11, stride=11)
        self.bn3 = nn.BatchNorm1d(256)
        self.dense1 = nn.Linear(256, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, self.nb_classes)


    def lenet(self):
        self.model = Sequential([
            Conv1D(filters=16, kernel_size=21, strides=1, padding='same', input_shape=self.shape,
                   kernel_initializer=keras.initializers.he_normal()),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=32, kernel_size=11, strides=1, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=64, kernel_size=5, strides=1, padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(2050, activation='relu'),
            Dropout(0.5),
            Dense(self.nb_classes, activation='softmax')  # or Activation('softmax')
        ])

    def vgg9(self):
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=21, strides=1, padding='same', activation='relu',
                   input_shape=self.shape, kernel_initializer=keras.initializers.he_normal()),
            BatchNormalization(),
            Conv1D(filters=64, kernel_size=21, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=128, kernel_size=11, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            Conv1D(filters=128, kernel_size=11, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2, strides=2, padding='same'),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(self.nb_classes, activation='softmax')  # or Activation('softmax')
        ])

    def random_init(self, init_func=nn.init.kaiming_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.squeeze()
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return torch.softmax(x, 1)
