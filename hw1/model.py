#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.regularizers import l2


class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

    def on_train_end(self, log):
        print('')


def plot_history(hist, plots_dir):
    def plot_metric(metric, df):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.plot(df['epoch'], df[metric], label='train')
        plt.plot(df['epoch'], df['val_' + metric], label='val')
        plt.legend()

    # save plots
    plot_metric('mean_absolute_error', hist)
    plt.savefig(os.path.join(plots_dir, '{}.png'.format('mean_absolute_error')))

    plot_metric('mean_squared_error', hist)
    plt.savefig(os.path.join(plots_dir, '{}.png'.format('mean_squared_error')))


def plot_predictions(test_labels, test_predictions, plots_dir):
    # create figure for a scatter plot between labels and predictions
    plt.figure()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.xlabel('Predictions Values')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-10, 10], [-10, 10])
    plt.savefig(os.path.join(plots_dir, '{}.png'.format('preds_scatter')))

    # create figure for error distribution plot
    plt.figure()
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, '{}.png'.format('preds_err')))


def norm(dataset, norm_params):
    return (dataset - norm_params['mean']) / norm_params['std']


def build_model(input_shape, output_shape):
    model = keras.Sequential([
        Dense(128, activation=tf.nn.relu, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dense(64, activation=tf.nn.relu, kernel_regularizer=l2(0.01)),
        Dense(output_shape)])

    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


def predict(data, model, norm_params):
    [data.drop(c, inplace=True, axis=1) for c in norm_params['removed'] if c in data]

    data = norm(data, norm_params['train_stats'])
    return model.predict(data)


def load(checkpoint_path, norm_params_path):
    """
      Loads policy for behavior cloning agent from checkpoint file.
    """
    # load trained model from disk
    model = keras.models.load_model(checkpoint_path)
    print(model.summary())

    # load normalization parameters from disk
    with open(norm_params_path, 'rb') as fp:
        norm_params = pickle.load(fp)

    return model, norm_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to expert policy pickle')
    parser.add_argument('--epochs', type=int, required=False, help='Number of epochs', default=1000)
    parser.add_argument('--output', type=str, required=True, help='output dir name')
    parser.add_argument('--checkpoint-name', type=str, required=False, help='checkpoint file name')
    parser.add_argument('--test-size', type=float, required=False, default=0.05, help='test dataset')
    args = parser.parse_args()

    # read dataset from disk
    with open(args.data_dir, 'rb') as fp:
        expert_data = pickle.load(fp)

    assert 'observations' in expert_data, 'expert data missing observations'
    assert 'actions' in expert_data, 'expert data missing actions'
    assert len(expert_data['observations']) == len(expert_data['actions']), \
        'number of observations is not equal to number of action'
    assert len(expert_data['observations']), 'empty expert policy'

    # create name for each input feature
    feature_names = ['f{}'.format(i) for i in range(len(expert_data['observations'][0]))]
    print('observation data shape: {}'.format(expert_data['observations'].shape))

    # create name for each input feature
    predictor_names = ['p{}'.format(i) for i in range(len(expert_data['actions'][0]))]
    print('actions data shape: {}'.format(expert_data['actions'].shape))

    # load features and predictors in a data frame
    dataset = pd.DataFrame(np.column_stack([expert_data['observations'], expert_data['actions']]),
                           columns=feature_names + predictor_names)
    print('dataset info before cleanup: ')
    print(dataset.head())
    print(dataset.describe().transpose())

    # cleanup dataset from columns with identical values
    to_remove = [c for c in feature_names if len(dataset[c].unique()) == 1]
    dataset.drop(to_remove, inplace=True, axis=1)
    feature_names = [x for x in feature_names if x not in to_remove]

    # split data into train and test
    train_dataset = dataset.sample(frac=1-args.test_size)
    test_dataset = dataset.drop(train_dataset.index)

    # splitting labels from features
    train_labels = pd.concat([train_dataset.pop(l) for l in predictor_names], 1)
    test_labels = pd.concat([test_dataset.pop(l) for l in predictor_names], 1)

    # inspect the data
    train_stats = train_dataset.describe().transpose()
    print(train_stats)

    # normalize the data
    normed_train_dataset = norm(train_dataset, train_stats)
    normed_test_dataset = norm(test_dataset, train_stats)

    # build model
    model = build_model([len(feature_names)], len(predictor_names))
    model_params = {'train_stats': train_stats, 'removed': to_remove}
    print(model.summary())

    # # try out the model and confirm it works
    # example_batch = normed_train_dataset[:10]
    # example_result = model.predict(example_batch)
    # print(example_result)

    # train the model
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(normed_train_dataset, train_labels, epochs=args.epochs, validation_split=0.2,
                        callbacks=[early_stop, PrintDot()], verbose=0)

    # create plots directory
    plots_dir = os.path.join(args.output, 'plots')
    os.makedirs(plots_dir)

    # visualize training stats using history object
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plot_history(hist, plots_dir)

    # evaluate model on test dataset
    loss, mse, mae = model.evaluate(normed_test_dataset, test_labels, verbose=0)
    print('Testing set:\n\tloss: {:5.2f}\n\tmse: {:5.2f}\n\tmse: {:5.2f}'.format(loss, mse, mae))

    # make pedictions on test dataset and visualize results
    test_predictions = predict(test_dataset, model, model_params)
    plot_predictions(test_labels, test_predictions, plots_dir)
    print(pd.DataFrame(test_predictions, columns=predictor_names).describe().transpose())

    # save the trained model to disk
    checkpoint_name = args.checkpoint_name
    if not checkpoint_name:
        checkpoint_name = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    checkpoint_dir = os.path.join(args.output, 'checkpoints')
    os.makedirs(checkpoint_dir)
    checkpoint = '{}.h5'.format(os.path.join(checkpoint_dir, checkpoint_name))
    norm_params = '{}.pkl'.format(os.path.join(checkpoint_dir, checkpoint_name))
    model.save(checkpoint)
    print('checkpoint saved at {}'.format(checkpoint))
    with open(norm_params, 'wb') as f:
        pickle.dump(model_params, f, pickle.HIGHEST_PROTOCOL)
    print('normalization parameters saved at {}'.format(norm_params))


if __name__ == '__main__':
    main()
