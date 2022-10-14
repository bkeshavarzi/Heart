import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model, metrics, model_selection, preprocessing, ensemble
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import sparse
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.ticker as mtick


def plot_randomforest(model, X_train, X_test, Y_train, Y_test):
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    error_train = Y_pred_train - Y_train
    error_test = Y_pred_test - Y_test

    fig, axis = plt.subplots(1, 2, figsize=(18, 6))

    axis[0].scatter(Y_train, Y_pred_train, color='red', edgecolors='black', s=50)
    axis[0].set_xlabel('Measured Sale Price (Train) ', fontsize=16)
    axis[0].set_ylabel('Predicted Sale Price (Train)', fontsize=16)
    axis[0].set_xlim([min(Y_train), max(Y_train)])
    axis[0].set_ylim([min(Y_train), max(Y_train)])
    axis[0].plot([0, max(Y_train)], [0, max(Y_train)], color='black')
    axis[0].grid()

    axis[1].scatter(Y_test, Y_pred_test, color='red', edgecolors='black', s=50)
    axis[1].set_xlabel('Measured Sale Price (Test) ', fontsize=16)
    axis[1].set_ylabel('Predicted Sale Price (Test)', fontsize=16)
    axis[1].set_xlim([min(Y_test), max(Y_test)])
    axis[1].set_ylim([min(Y_test), max(Y_test)])
    axis[1].plot([0, max(Y_test)], [0, max(Y_test)], color='black')
    axis[1].grid();

    fig, axis = plt.subplots(1, 2, figsize=(18, 6))

    av = sns.histplot(error_train, bins=30, kde=True, color='cyan', ax=axis[0])
    av.set_xlabel('Sale Price Error (Train) ($)', fontsize=16)
    av.set_ylabel('Frequency', fontsize=16)

    av = sns.histplot(error_test, bins=30, kde=True, color='cyan', ax=axis[1])
    av.set_xlabel('Sale Price Error (Test) ($)', fontsize=16)
    av.set_ylabel('Frequency', fontsize=16);

    fig, axis = plt.subplots(1, 2, figsize=(18, 6))

    axis[0].scatter(Y_train, error_train, color='red', edgecolors='black', s=18)
    axis[0].set_xlabel('Measured Sale Price (Train) ($)', fontsize=16)
    axis[0].set_ylabel('Error in Sale Price Prediction (Train) ($)', fontsize=16)
    axis[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    axis[0].grid()

    av = qqplot(error_train, line='s', ax=axis[1])
    av = av.gca()
    av.set_xlabel('Theoretical Quantile', fontsize=16)
    av.set_ylabel('Sample Quantile for Sale Price (Train) ($)', fontsize=16)
    av.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

    fig, axis = plt.subplots(1, 2, figsize=(18, 6))

    axis[0].scatter(Y_test, error_test, color='red', edgecolors='black', s=18)
    axis[0].set_xlabel('Measured Sale Price (Test) ($)', fontsize=18)
    axis[0].set_ylabel('Error in Sale Price Prediction (Test) ($)', fontsize=18)
    axis[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    axis[0].grid()

    av = qqplot(error_test, line='s', ax=axis[1])
    av = av.gca()
    av.set_xlabel('Theoretical Quantile', fontsize=16)
    av.set_ylabel('Sample Quantile for Sale Price (Test) ($)', fontsize=16)
    av.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'));