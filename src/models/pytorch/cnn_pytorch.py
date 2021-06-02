#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import torch
from torch import nn


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
