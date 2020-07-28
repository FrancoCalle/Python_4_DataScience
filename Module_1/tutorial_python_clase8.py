"""

Segunda Clase: Métodos de Machine Learning para Clasificación
- Train and test sample
- Logistic Regression
- K-Nearest Neighbors Classification
- Decision Trees
- Random Forests

"""

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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


#Load Dataset and show scatterplot
iris = datasets.load_iris()

X = iris['data']
X = X[:,:2] # Usamos ancho y largo de la planta
Y = (iris['target'] == 2) #[setosa, versicolor, **virginica**]

sns.scatterplot(X[:,0],X[:,1], hue=Y)

# Logistic Regression:
clf = LogisticRegression(random_state=0).fit(X[:,:2], Y) # Definir el problema
clf.coef_
logitPredictionScore = clf.predict_proba(X)[:,1]
sns.scatterplot(X[:,0],X[:,1], hue=logitPredictionScore)


# KNeighbors Classifier:
# we create an instance of Neighbours Classifier and fit the data.
n_neighbors=15
h = .02  # step size in the mesh
weights = 'uniform'

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

clf = KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X, Y)
kNNPredictionScore = clf.predict_proba(X)[:,1]

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Predecimos para cada punto en la grilla

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("3-Class classification (k = %i, weights = '%s')"
          % (n_neighbors, weights))
plt.show()


#Decision Tree Classifier:
clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
clf.fit(X,Y)
dtcPredictionScore = clf.predict_proba(X)[:,1]

tree.plot_tree(clf)
plt.show()

#Random Forest Classifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X,Y)
rfcPredictionScore = clf.predict_proba(X)[:,1]


# Compare Area Under the Curve for All Models and select the best one:

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

    thresholds = np.linspace(0, 1, num=100)
    tprList = []
    fprList = []

    for tt in thresholds:

        Y_hat = Y_score > tt
        tpr = getTruePositiveRate(Y,Y_hat)
        fpr = getFalsePositiveRate(Y,Y_hat)
        tprList.append(tpr)
        fprList.append(fpr)

    return tprList, fprList

# Get ROC curve for Logistic Regression
tprListLogistic, fprListLogistic = thresholdingAUC(Y, logitPredictionScore)
AUC1 = metrics.auc(fprListLogistic, tprListLogistic)
# Get ROC curve for KNN
tprListKNN, fprListKNN = thresholdingAUC(Y, kNNPredictionScore)
AUC2 = metrics.auc(fprListKNN, tprListKNN)
# Get ROC curve for DecisionTreeClassifier
tprListDTC, fprListDTC = thresholdingAUC(Y, dtcPredictionScore)
AUC3 = metrics.auc(fprListDTC, tprListDTC)
# Get ROC curve for Logistic Regression
tprListRFC, fprListRFC = thresholdingAUC(Y, rfcPredictionScore)
AUC4 = metrics.auc(fprListRFC, tprListRFC)


plt.plot(fprListLogistic, tprListLogistic,  label='ROC curve Logit (area = %0.2f)' % AUC1)
plt.plot(fprListKNN, tprListKNN,  label='ROC curve KNN (area = %0.2f)' % AUC2)
plt.plot(fprListDTC, tprListDTC,  label='ROC curve DTC (area = %0.2f)' % AUC3)
plt.plot(fprListRFC, tprListRFC,  label='ROC curve RFC (area = %0.2f)' % AUC4)
plt.plot(tprListLogistic, tprListLogistic, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
