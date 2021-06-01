#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import numpy as np
import pandas as pd
import torch
import json

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from ax.service.managed_loop import optimize

from src.models.pytorch.cnn_pytorch import CNN
from src.utils.utils import ms_data
from src.utils.dataset import MSDataset, validation_spliter, load_checkpoint, save_checkpoint
from src.utils.CycleAnnealScheduler import CycleScheduler  # TODO compare/replace to keras one_cycle
from src.utils.utils import to_categorical

np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Train:
    def __init__(self, intensities_file, labels_file, get_data_function=ms_data, n_channels=1, verbose=1, cv=5,
                 n_epochs=50, batch_size=64, epochs_per_checkpoint=1, save=False, early_stop_val=10, epochs_per_print=1):

        self.data = get_data_function(intensities_file)
        self.labels = pd.read_csv(labels_file, header=0).loc[0]

        for i, label in enumerate(self.labels):
            if label != 'Normal':
                self.labels[i] = 'Not Normal'

        self.labels = self.labels.astype("category").cat.codes

        self.nb_classes = len(np.unique(self.labels))
        self.input_shape = [self.data.shape[1], n_channels]
        self.cv = cv
        self.verbose = 1
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.optimizer_type = None
        self.scheduler = None
        self.learning_rate = None
        self.weight_decay = None
        self.beta1 = None
        self.beta2 = None
        self.min_lr = None
        self.l1 = None

        self.save = save
        self.epochs_per_print = epochs_per_print
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.early_stop = early_stop_val
        self.criterion = torch.nn.BCELoss()

    def train(self, params):
        optimizer = None
        # self.optimizer_type = h_params['optimizer']
        # self.scheduler = h_params['scheduler']
        # self.learning_rate = h_params['learning_rate'].__format__('e')
        # self.weight_decay = h_params['weight_decay'].__format__('e')
        self.optimizer_type = params['optimizer']
        self.scheduler = params['scheduler']
        self.learning_rate = params['lr']
        self.weight_decay = params['wd']

        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.min_lr = params['min_lr']
        self.l1 = params['l1']

        indices = list(range(len(self.data)))
        np.random.shuffle(indices)

        all_set = MSDataset(self.data[indices], self.labels[indices], transform=None, crop_size=-1, device='cuda')
        spliter = validation_spliter(all_set, cv=self.cv)

        n_epochs = 10
        epoch = 0
        epoch_offset = max(1, epoch)

        self.model_name = f"vae_3dcnn_ \
                          {self.optimizer_type} \
                          _nepochs + {self.n_epochs} \
                          _lr' + {self.learning_rate}  \
                          _wd' + {self.weight_decay} \
                          _l1' + {self.l1} \
                          _l2' + {self.l2} "


        test_accuracies = []
        for cv in range(self.cv):
            cnn = CNN(self.nb_classes, self.input_shape).to(device)
            # cnn.random_init()
            best_loss = -1
            valid_set, train_set = spliter.__next__()

            train_loader = DataLoader(train_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      pin_memory=False,
                                      drop_last=True)
            valid_loader = DataLoader(valid_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      pin_memory=False,
                                      drop_last=True)
            weight_decay = float(str(self.weight_decay)[:1] + str(self.weight_decay)[-4:])
            learning_rate = float(str(self.learning_rate)[:1] + str(self.learning_rate)[-4:])
            beta1 = float(str(self.beta1)[:1] + str(self.beta1)[-4:])
            beta2 = float(str(self.beta2)[:1] + str(self.beta2)[-4:])
            if self.optimizer_type == 'adam':
                optimizer = torch.optim.Adam(params=cnn.parameters(),
                                             lr=learning_rate,
                                             weight_decay=weight_decay,
                                             betas=(beta1, beta2)
                                             )
            elif self.optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(params=cnn.parameters(),
                                            lr=learning_rate,
                                            weight_decay=weight_decay,
                                            momentum=0.9)
            elif self.optimizer_type == 'rmsprop':
                optimizer = torch.optim.RMSprop(params=cnn.parameters(),
                                                lr=learning_rate,
                                                weight_decay=weight_decay,
                                                momentum=0.9)
            else:
                exit('error: no such optimizer type available')
            # model.summary()

            # l1 = float(str(l1)[:1] + str(l1)[-4:])
            # l2 = float(str(l2)[:1] + str(l2)[-4:])
            l1 = 0
            l2 = 0
            # Get shared output_directory ready
            logger = SummaryWriter('logs_old')

            if self.scheduler == 'ReduceLROnPlateau':
                lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                         factor=0.1,
                                                                         cooldown=50,
                                                                         patience=200,
                                                                         verbose=True,
                                                                         min_lr=1e-15)
            elif self.scheduler == 'CycleScheduler':
                lr_schedule = CycleScheduler(optimizer,
                                             learning_rate,
                                             n_iter=n_epochs * len(train_loader)
                                             )
            else:
                lr_schedule = None
            losses = {
                "train": [],
                "valid": [],
            }
            shapes = {
                "train": len(train_set),
                "valid": len(valid_set),
            }
            early_stop_counter = 0
            print("\n\n\nCV:", cv, "/", self.cv, "\nTrain samples:", len(train_set),
                  "\nValid samples:", len(valid_set), "\n\n\n")

            values = {
                'train': {
                    'losses': [],
                    'acc': []
                },
                'valid': {
                    'losses': [],
                    'acc': []
                }
            }

            traces = {
                'train': {
                    'losses': [],
                    'acc': []
                },
                'valid': {
                    'losses': [],
                    'acc': []
                }
            }

            for epoch in range(epoch_offset, self.n_epochs):
                if early_stop_counter == self.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break
                best_epoch = False
                cnn.train()
                for i, batch in enumerate(train_loader):
                    # pbar.update(1)
                    # optimizer.zero_grad()
                    cnn.zero_grad()
                    data, labels = batch
                    data[torch.isnan(data)] = 0
                    data = data.to(device).unsqueeze(1)
                    preds = cnn(data)
                    loss = self.criterion(preds, torch.Tensor(to_categorical(labels, self.nb_classes)).float().to(device))
                    loss.backward()

                    values['train']['losses'] += [loss.item()]
                    values['train']['acc'] += [1 if pred == label else 0 for pred, label in
                                               zip(preds.argmax(1), labels)]
                    optimizer.step()
                    if self.scheduler == "CycleScheduler":
                        lr_schedule.step()
                    logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
                    logger.add_scalar('train_acc', values['train']['acc'][-1], i + len(train_loader) * epoch)
                    del loss
                traces['train']['losses'] += [np.mean(values['train']['losses'])]
                traces['train']['acc'] += [np.mean(values['train']['acc'])]

                cnn.eval()
                for i, batch in enumerate(valid_loader):
                    data, labels = batch
                    data = data.to(device).unsqueeze(1)
                    data[torch.isnan(data)] = 0
                    preds = cnn(data)
                    loss = self.criterion(preds, torch.Tensor(to_categorical(labels, self.nb_classes)).float().to(device))
                    values['valid']['losses'] += [loss.item()]
                    values['valid']['acc'] += [1 if pred == label else 0 for pred, label in zip(preds.argmax(1), labels)]
                    logger.add_scalar('valid_loss', loss.item(), i + len(train_loader) * epoch)
                    logger.add_scalar('valid_acc', values['valid']['acc'][-1], i + len(train_loader) * epoch)
                    del loss
                traces['valid']['losses'] += [np.mean(values['valid']['losses'])]
                traces['valid']['acc'] += [np.mean(values['valid']['acc'])]

                if (traces['valid']['losses'][-1] < best_loss or best_loss == -1) and not np.isnan(
                        traces['valid']['losses'][-1]):
                    if self.verbose > 1:
                        print('BEST EPOCH!', traces['valid']['losses'][-1])
                    early_stop_counter = 0
                    best_loss = traces['valid']['losses'][-1]
                    best_epoch = True
                else:
                    early_stop_counter += 1
                if epoch % self.epochs_per_checkpoint == 0:
                    if best_epoch and self.save:
                        if self.verbose > 1:
                            print('Saving model...')
                        save_checkpoint(model=cnn,
                                        optimizer=optimizer,
                                        learning_rate=learning_rate,
                                        epoch=epoch,
                                        checkpoint_path='results',
                                        losses=losses,
                                        best_loss=best_loss,
                                        name=self.model_name,
                                        )
                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 0:
                        print("Epoch: {}:\t"
                              "Valid Loss: {:.5f} , "
                              "Valid Accuracy: {:.2f} , "
                              .format(epoch,
                                      traces['valid']['losses'][-1],
                                      traces['valid']['acc'][-1],
                                      )
                              )

                    if self.verbose > 1:
                        print("Current LR:", optimizer.param_groups[0]['lr'])
                    if 'momentum' in optimizer.param_groups[0].keys():
                        print("Current Momentum:", optimizer.param_groups[0]['momentum'])

        # TODO Return the best loss to enable Bayesian Optimisation
        return np.mean(test_accuracies)

    def print_parameters(self):
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'n_epochs: ' + str(self.n_epochs) + "\n\t",
                  'learning_rate: ' + self.learning_rate.__format__('e') + "\n\t",
                  'weight_decay: ' + self.weight_decay.__format__('e') + "\n\t",
                  'l1: ' + self.l1.__format__('e') + "\n\t",
                  'l2: ' + self.l2.__format__('e') + "\n\t",
                  'optimizer_type: ' + self.optimizer_type + "\n\t",
                  )

    def load_checkpoint(self):
        model, _, \
        epoch, losses, \
        kl_divs, losses_recon, \
        best_loss = load_checkpoint('results',
                                    None,
                                    self.maxpool,
                                    save=self.save,
                                    name=self.model_name,
                                    )

    def save_checkpoint(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--intensities_csv", type=str, default="data\\canis_intensities.csv",
                        help="Path to intensities csv file")
    parser.add_argument("--labels_csv", type=str, default="data\\canis_labels.csv",
                        help="Path to labels csv file")
    args = parser.parse_args()
    train = Train(args.intensities_csv, args.labels_csv)

    train.train()

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "scheduler", "type": "choice", "values":
                ['RecueLROnPleateau', 'RecueLROnPleateau']},
            {"name": "optimizer", "type": "choice", "values": ['adam', 'adam']},
            {"name": "beta1", "type": "range", "bounds": [0.9, 0.99], "log_scale": True},
            {"name": "beta2", "type": "range", "bounds": [0.99, 0.9999], "log_scale": True},
            {"name": "min_lr", "type": "range", "bounds": [1e-8, 1e-6], "log_scale": True},
            {"name": "l1", "type": "range", "bounds": [1e-8, 1e-1], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-5, 1e-1], "log_scale": True},
            {"name": "momentum", "type": "choice", "values": [0, 0]},
            {"name": "learning_rate", "type": "range", "bounds": [1e-5, 1e-4], "log_scale": True},
        ],
        evaluation_function=train.train,
        objective_name='loss',
        minimize=True,
        total_trials=3
    )
    from matplotlib import pyplot as plt

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))

    # cv_results = cross_validate(model)
    # render(interact_cross_validation(cv_results))
