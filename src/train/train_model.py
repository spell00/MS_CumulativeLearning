#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 2021
@author: Simon Pelletier
"""

import os
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import random
from tensorflow.keras.regularizers import l1_l2

from src.models.keras.CNN import CNN
from src.models.keras.Linear import Linear
from src.utils.utils import split_train_test, getScalerFromString
from src.utils.metrics import matthews_correlation as mcc
from src.utils.dataset import ms_data

from numpy.random import seed
import tensorflow as tf

from skopt.space import Real, Integer
from skopt import gp_minimize
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from datetime import datetime

seed(42)
tf.random.set_seed(42)
random.seed(42)

os.makedirs(f'logs/hparam_tuning', exist_ok=True)


def compute_confusion_matrix(y_test, y_classes):
    tn, fp, fn, tp = confusion_matrix(y_test, y_classes).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return sensitivity, specificity


class TensorboardLogging:

    def __init__(self, hparams_filepath, params):
        self.params = params
        self.hparams_filepath = hparams_filepath
        HP_EPOCHS = hp.HParam('epochs', hp.IntInterval(1, 50))
        HP_LR = hp.HParam('lr', hp.RealInterval(1e-6, 1e-3))
        HP_WD = hp.HParam('wd', hp.RealInterval(1e-8, 1e-3))
        HP_BS = hp.HParam('bs', hp.IntInterval(1, 512))
        HP_L1 = hp.HParam('l1', hp.IntInterval(1, 512))
        HPARAMS = [HP_EPOCHS, HP_LR, HP_WD, HP_BS, HP_L1]
        with tf.summary.create_file_writer(hparams_filepath).as_default():
            hp.hparams_config(
                hparams=HPARAMS,
                metrics=[
                    hp.Metric('train_accuracy', display_name='Train Accuracy'),
                    hp.Metric('valid_accuracy', display_name='Valid Accuracy'),
                    hp.Metric('test_accuracy', display_name='Test Accuracy'),
                    hp.Metric('train_loss', display_name='Train Loss'),
                    hp.Metric('valid_loss', display_name='Valid Loss'),
                    hp.Metric('test_loss', display_name='Test Loss'),
                    hp.Metric('train_mcc', display_name='Train MCC'),
                    hp.Metric('valid_mcc', display_name='Valid MCC'),
                    hp.Metric('test_mcc', display_name='Test MCC')
                ],
            )

    def logging(self, traces):
        epochs = self.params['n_epochs']
        lr = self.params['lr']
        wd = self.params['wd']
        l1 = self.params['l1']
        bs = self.params['bs']
        with tf.summary.create_file_writer(self.hparams_filepath).as_default():
            hp.hparams({
                'epochs': epochs,
                'lr': lr,
                'wd': wd,
                'bs': bs,
                'l1': l1,
            })  # record the values used in this trial
            tf.summary.scalar('train_accuracy', np.mean([np.mean(x) for x in traces['train']['accuracies']]), step=1)
            tf.summary.scalar('valid_accuracy', np.mean(traces['valid']['accuracies']), step=1)
            tf.summary.scalar('test_accuracy', np.mean(traces['test']['accuracies']), step=1)
            tf.summary.scalar('train_loss', np.mean([np.mean(x) for x in traces['train']['losses']]), step=1)
            tf.summary.scalar('valid_loss', np.mean(traces['valid']['losses']), step=1)
            tf.summary.scalar('test_loss', np.mean(traces['test']['losses']), step=1)
            tf.summary.scalar('train_mcc', np.mean([np.mean(x) for x in traces['train']['mccs']]), step=1)
            tf.summary.scalar('valid_mcc', np.mean(traces['valid']['mccs']), step=1)
            tf.summary.scalar('test_mcc', np.mean(traces['test']['mccs']), step=1)


class Train:
    def __init__(
            self,
            intensities_file,
            cumulative_step,
            scaler,
            criterion='categorical_crossentropy',
            variant='logistic',
            get_data_function=ms_data,
            n_channels=1,
            save_train_models=True,
            model_name=CNN,
            verbose=0,
            model_path=False,
            freeze=True,
            retrain=True,
    ):
        self.dataset_name = intensities_file.split('/')[-1].split('_')[0]
        self.freeze = freeze
        self.verbose = verbose
        self.variant = variant
        self.scaler = scaler
        self.retrain = retrain
        self.cumulative_step = cumulative_step
        self.model_name = model_name
        self.save_train_models = save_train_models
        self.data, labels, samples = get_data_function(intensities_file)
        self.data[np.isnan(self.data)] = 0

        # TODO have this step done in R import_data.R

        if 'Normal' in labels:
            for i, label in enumerate(labels):
                if label != 'Normal':
                    labels[i] = 'Not Normal'
                    # categories[i] = 1
                # else:
                # categories[i] = 0
        categories = pd.Categorical(labels).codes
        self.labels_df = pd.concat([
            pd.DataFrame(np.array(samples).reshape([-1, 1])),
            pd.DataFrame(np.array(categories).reshape([-1, 1])),
            pd.DataFrame(np.array(labels).reshape([-1, 1])),
        ], 1)
        self.labels_df.columns = ['sample', 'category', 'label']
        self.nb_classes = len(np.unique(self.labels_df['category']))
        self.input_shape = [self.data.shape[1], n_channels]
        self.criterion = criterion
        self.step = 0
        self.call_num = 0
        self.previous_datasets = ""
        if model_path != 'None':
            self.model_path, self.previous_datasets = self.select_best_model(model_path)
        else:
            self.model_path = False
        if self.previous_datasets == "":
            self.datasets = self.dataset_name
        else:
            self.datasets = f"{self.previous_datasets}_{self.dataset_name}"
        self.params = {}

    def select_best_model(self, params_fname):
        # import from json file
        params = pd.read_json(params_fname)
        best_call = 0
        best_valid_loss = np.inf
        for call in list(params.keys()):
            values = params[call]
            if float(values['scores']['valid_loss']) < best_valid_loss:
                best_valid_loss = float(values['scores']['valid_loss'])
                best_call = call
        hparams = params[best_call]['h_params']
        base_path = params_fname.split('/')
        previous_datasets = params[best_call]['datasets']
        assert previous_datasets == base_path[4]
        base_path = f"{base_path[0]}/{base_path[1]}/{base_path[2]}/{base_path[3]}/{base_path[4]}/{base_path[5]}/{base_path[6]}"

        path = \
            f'{self.criterion}/' + \
            f'{hparams["n_epochs"]}/' \
            f'{"{:.8f}".format(float(hparams["lr"]))}/' \
            f'{"{:.8f}".format(float(hparams["wd"]))}/' \
            f'{"{:.8f}".format(float(hparams["l1"]))}/' \
            f'{hparams["bs"]}/'

        dir_path = f"{base_path}/{path}"

        return dir_path, previous_datasets

    def update_model(self, model_source, path, wd):
        model_source.model.build([1, 15020])
        model_source.model.load_weights(f"{path}/{self.variant}_{self.cumulative_step - 1}.h5")

        for layer in model_source.model.layers:
            layer.trainable = self.freeze
        if self.variant == 'lecun':
            model_source.model.pop()
            model_source.model.pop()

            new_model = Sequential([
                Dense(2050, activation='relu', activity_regularizer=l1_l2(wd)),
                Dropout(0.5),
                Dense(self.nb_classes, activation='softmax', activity_regularizer=l1_l2(wd))
            ])

            for layer in new_model.layers:
                layer.trainable = self.freeze

            model_source.model = Sequential([
                model_source.model,
                new_model
            ])
        elif self.variant == 'lenet':
            model_source.model.pop()
            model_source.model.pop()
            model_source.model.pop()
            model_source.model = Sequential([
                model_source,
                Flatten(),
                Dense(120, activation='relu', activity_regularizer=l1_l2(wd)),
                Dense(84, activity_regularizer=l1_l2(wd)),
                Dense(self.nb_classes, activation='softmax', activity_regularizer=l1_l2(wd))
            ])
        elif self.variant == 'vgg9':
            model_source.model.pop()
            model_source.model.pop()
            model_source.model.pop()
            model_source.model = Sequential([
                model_source.model,
                Flatten(),
                Dense(4096, activation='relu', activity_regularizer=l1_l2(wd)),
                Dropout(0.5),
                Dense(4096, activation='relu', activity_regularizer=l1_l2(wd)),
                Dropout(0.5),
                Dense(self.nb_classes, activation='softmax', activity_regularizer=l1_l2(wd))
            ])

        elif self.variant == 'logistic':
            pass
        else:
            exit("Wrong variant. Chose lenet, lecun, vgg9 or logistic ")
        return model_source

    def train(self, h_params):
        self.call_num += 1
        n_epochs = h_params[0]
        lr = h_params[1]
        wd = h_params[2]
        l1 = h_params[3]
        bs = h_params[4]

        h_params = {
            "n_epochs": n_epochs,
            "lr": lr,
            "wd": wd,
            "l1": l1,
            "bs": bs,
        }

        path = \
            f'{self.criterion}/' + \
            f'{n_epochs}/' \
            f'{"{:.8f}".format(float(lr))}/' \
            f'{"{:.8f}".format(float(wd))}/' \
            f'{"{:.8f}".format(float(l1))}/' \
            f'{bs}/'
        dnn = self.model_name(h_params, self.nb_classes, variant=self.variant, activation='relu', batch_size=bs)
        hparams_filepath = f"logs/{dnn.get_model_name()}/{self.variant}/{self.datasets}/{self.freeze}/{self.retrain}" \
                           f"/hparam_tuning/{path}"
        log_filepath = f"logs/{dnn.get_model_name()}/{self.variant}/{self.datasets}/{self.freeze}/{self.retrain}/{path}"
        del dnn
        os.makedirs(log_filepath, exist_ok=True)
        tb_logging = TensorboardLogging(hparams_filepath, h_params)
        traces = {
            "train": {
                "losses": [],
                "accuracies": [],
                "mccs": [],
                "sensitivities": [],
                "specificities": [],
            },
            "valid": {
                "losses": [],
                "accuracies": [],
                "mccs": [],
                "sensitivities": [],
                "specificities": [],
            },
            "test": {
                "losses": [],
                "accuracies": [],
                "mccs": [],
                "sensitivities": [],
                "specificities": [],
            },
        }

        all_train_samples, test_samples, train_cats = split_train_test(self.labels_df)
        all_train_indices = [s for s, lab in enumerate(self.labels_df['sample']) if lab in all_train_samples]
        test_indices = [s for s, lab in enumerate(self.labels_df['sample']) if lab in test_samples]

        x_test = self.data[test_indices]
        y_test = self.labels_df['category'][test_indices].tolist()

        assert len(set(y_test)) == self.nb_classes

        # 3-fold CV; there is only 8 Normal samples for canis sarcoma, so will avoid having only 1 Normal per set
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        for i, (train_samples, valid_samples) in enumerate(skf.split(all_train_samples, train_cats)):
            # Just plot the first iteration, it will already be crowded if doing > 100 optimization iterations
            if self.verbose:
                print(f"CV: {i}")
            # new_nums = [train_nums[i] for i in inds]

            valid_samples = [all_train_samples[s] for s in valid_samples]
            train_samples = [all_train_samples[s] for s in train_samples]
            train_indices = [s for s, lab in enumerate(self.labels_df['sample'].tolist()) if lab in train_samples]
            valid_indices = [s for s, lab in enumerate(self.labels_df['sample'].tolist()) if lab in valid_samples]

            x_train = self.data[train_indices]
            y_train = self.labels_df['category'][train_indices]
            x_valid = self.data[valid_indices]
            y_valid = self.labels_df['category'][valid_indices]

            assert len(set(y_train)) == self.nb_classes and len(set(y_valid)) == self.nb_classes
            assert len(all_train_indices) == len(train_indices) + len(valid_indices)
            assert len([x for x in valid_indices if x in train_indices]) == 0
            assert len([x for x in valid_samples if x in train_samples]) == 0

            scaler = getScalerFromString(self.scaler)()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_valid = scaler.transform(x_valid)

            x_train_conv = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_valid_conv = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
            y_train_conv = to_categorical(y_train, self.nb_classes)
            y_valid_conv = to_categorical(y_valid, self.nb_classes)

            dnn = self.model_name(h_params, self.nb_classes, batch_size=bs, variant=self.variant, activation='relu')
            dnn.build(input_shape=self.input_shape)

            if self.model_path:
                dnn = self.update_model(model_source=dnn, path=self.model_path, wd=wd)
            dnn.model.compile(loss=self.criterion,
                              optimizer='adam',
                              metrics=['accuracy', mcc])
            callbacks = []
            if i == 0:
                callbacks += [keras.callbacks.TensorBoard(
                    log_dir=log_filepath,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=False,
                    update_freq="epoch",
                    profile_batch=2,
                    embeddings_freq=0,
                    embeddings_metadata=None,
                )]
            callbacks += [keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0,
                patience=10,
                verbose=self.verbose,
                mode='min'
            )]

            y_integers = np.argmax(y_train_conv, axis=1)
            class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
            d_class_weights = dict(enumerate(class_weights))

            # CAN't USE VALIDATION SPLIT, so also can't use early_stopping or reduceLROnPLateau
            if self.verbose == 2:
                fit_verbose = 1
            else:
                fit_verbose = 0
            history = dnn.model.fit(
                x=x_train_conv,
                y=y_train_conv,
                batch_size=bs,
                verbose=fit_verbose,
                epochs=n_epochs,
                validation_split=0.,
                class_weight=d_class_weights,
                callbacks=callbacks
            )
            base_path = f'saved_models/keras/{dnn.get_model_name()}/{self.variant}/{self.datasets}/' \
                        f'{self.freeze}/{self.retrain}/'
            file_path = f'{base_path}/{path}'
            os.makedirs(file_path, exist_ok=True)

            train_acc = history.history['accuracy']
            train_loss = history.history['loss']
            train_mcc = history.history['matthews_correlation']

            # train_loss, train_acc, train_mcc = dnn.model.evaluate(x_train_conv, y_train_conv, verbose=self.verbose)
            y_classes_train = np.argmax(dnn.model.predict(x_train_conv), axis=-1)
            train_sensitivity, train_specificity = compute_confusion_matrix(y_train, y_classes_train)

            valid_loss, valid_acc, valid_mcc = dnn.model.evaluate(x_valid_conv, y_valid_conv, verbose=self.verbose)
            y_classes_valid = np.argmax(dnn.model.predict(x_valid_conv), axis=-1)
            valid_sensitivity, valid_specificity = compute_confusion_matrix(y_valid, y_classes_valid)

            best_epoch = np.argmin(valid_loss)

            traces['train']['losses'].append(train_loss)
            traces['train']['accuracies'].append(train_acc)
            traces['train']['mccs'].append(train_mcc)
            traces['train']['sensitivities'].append(train_sensitivity)
            traces['train']['specificities'].append(train_specificity)
            traces['valid']['losses'].append(valid_loss)
            traces['valid']['accuracies'].append(valid_acc)
            traces['valid']['mccs'].append(valid_mcc)
            traces['valid']['sensitivities'].append(valid_sensitivity)
            traces['valid']['specificities'].append(valid_specificity)

        if self.retrain:
            x_all_train = self.data[all_train_indices]
            y_all_train = self.labels_df['category'][all_train_indices]

            scaler = getScalerFromString('robust')()
            scaler.fit(x_all_train)
            all_x_train = scaler.transform(x_all_train)

            x_train_conv = np.reshape(all_x_train, (all_x_train.shape[0], all_x_train.shape[1], 1))
            y_train_conv = to_categorical(y_all_train, self.nb_classes)

            history = dnn.model.fit(
                x=x_train_conv,
                y=y_train_conv,
                batch_size=bs,
                verbose=fit_verbose,
                epochs=n_epochs,
                validation_split=0.,
                class_weight=d_class_weights,
                callbacks=callbacks
            )

        x_test_trans = scaler.transform(x_test)
        x_test_conv = np.reshape(x_test_trans, (x_test_trans.shape[0], x_test_trans.shape[1], 1))
        y_test_conv = to_categorical(y_test, self.nb_classes)

        test_loss, test_acc, test_mcc = dnn.model.evaluate(x_test_conv, y_test_conv, verbose=self.verbose)
        y_classes_test = np.argmax(dnn.model.predict(x_test_conv), axis=-1)
        test_sensitivity, test_specificity = compute_confusion_matrix(y_test, y_classes_test)
        traces['test']['losses'].append(test_loss)
        traces['test']['accuracies'].append(test_acc)
        traces['test']['mccs'].append(test_mcc)
        traces['test']['specificities'].append(test_specificity)
        traces['test']['sensitivities'].append(test_sensitivity)
        # dnn.model.summary()

        dnn.model.save_weights(f'{file_path}/{self.variant}_{self.cumulative_step}.h5')
        self.step += 1
        try:
            tb_logging.logging(traces)
        except:
            print("\nProblem with logging\n")

        self.params[f"call_{self.call_num}"] = {
            'datasets': self.datasets,
            'h_params': {
                'criterion': f'{self.criterion}',
                'n_epochs': f'{n_epochs}',
                'lr': f'{lr}',
                'wd': f'{wd}',
                'l1': f'{l1}',
                'bs': f'{bs}',
            },
            'scores': {
                'best_epoch': f'{best_epoch}',
                'train_loss': f'{train_loss[int(best_epoch)]}',
                'valid_loss': f'{valid_loss}',
                'test_loss': f'{test_loss}',
                'train_acc': f'{train_acc[int(best_epoch)]}',
                'valid_acc': f'{valid_acc}',
                'test_acc': f'{test_acc}',
                'train_mcc': f'{train_mcc[int(best_epoch)]}',
                'valid_mcc': f'{valid_mcc}',
                'test_mcc': f'{test_mcc}',
            }
        }
        json.dump(self.params, open(f'{base_path}/params.json', 'w'))

        return 1 - np.mean(traces['valid']['accuracies'])

    def test(self, params):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--intensities_csv", type=str, default="data/canis_intensities.csv",
                        help="Path to intensities csv file")
    parser.add_argument("--verbose", type=int, default=0,
                        help="Path to labels csv file")
    parser.add_argument("--freeze", type=int, default=1,
                        help="Path to labels csv file")
    parser.add_argument("--model", type=str, default='logistic',
                        help="Name of the model to use [lenet, lecun, vgg9 or logistic].")
    parser.add_argument("--criterion", type=str, default='categorical_crossentropy',
                        help=".")
    parser.add_argument("--pretrained_path", type=str, default=None,  #
                        help="")
    parser.add_argument("--cumulative_step", type=int, default=0,  #
                        help="")
    parser.add_argument("--scaler", type=str, default='standard',  #
                        help="")

    args = parser.parse_args()

    if args.model in ['lenet', 'lecun', 'vgg9']:
        model = CNN
    elif args.model in ['logistic']:
        model = Linear
    else:
        exit("This model is not implemented, chose lenet, lecun, vgg9 or logistic ")
    if args.freeze == 0:
        args.freeze = False
    elif args.freeze == 1:
        args.freeze = True
    else:
        exit('freeze should be 0 or 1')

    train = Train(args.intensities_csv,
                  args.cumulative_step,
                  criterion=args.criterion,
                  variant=args.model,
                  verbose=args.verbose,
                  model_name=model,
                  model_path=args.pretrained_path,
                  freeze=args.freeze,
                  scaler=args.scaler
                  )
    space = [
        Integer(1, 100, "uniform", name='epochs'),
        Real(1e-6, 1e-3, "log-uniform", name='lr'),
        Real(1e-8, 1e-3, "log-uniform", name='wd'),
        Real(1e-8, 1e-3, "log-uniform", name='l1'),
        Integer(1, 512, "uniform", name='bs'),
    ]

    test_mean = gp_minimize(train.train, space, n_calls=100, random_state=42)
    print(test_mean)
