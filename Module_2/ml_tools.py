import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn import datasets
from sklearn import metrics



def getTruePositiveRate(Y, Y_hat):

    TP = Y[Y_hat == True].sum()
    FP = (~Y[Y_hat == True]).sum()
    FN = Y[Y_hat == False].sum()
    TN = (~Y[Y_hat == False]).sum()

    tpr = TP/(TP+FN)

    return tpr


def getFalsePositiveRate(Y, Y_hat):

    TP = Y[Y_hat == True].sum()
    FP = (~Y[Y_hat == True]).sum()
    FN = Y[Y_hat == False].sum()
    TN = (~Y[Y_hat == False]).sum()

    fpr = FP/(FP+TN)

    return fpr

def thresholdingAUC(Y, Y_score):

    thresholds = np.unique(Y_score)
    tprList = []
    fprList = []

    for tt in thresholds:

        Y_hat = Y_score > tt
        tpr = getTruePositiveRate(Y,Y_hat)
        fpr = getFalsePositiveRate(Y,Y_hat)
        tprList.append(tpr)
        fprList.append(fpr)

    return tprList, fprList


def getMSE(Y, Y_hat):

    N = Y.shape[0]
    E = Y-Y_hat
    MSE = np.sum(E**2)/N

    return MSE

def getTSE(Y):

    N = Y.shape[0]
    E = Y-Y.mean()
    MSE = np.sum(E**2)/N

    return MSE

def getR2(Y, Y_hat):

    MSE = getMSE(Y, Y_hat)
    TSE = getTSE(Y)

    R2 = 1-MSE/TSE

    return R2
