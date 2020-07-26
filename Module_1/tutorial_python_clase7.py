"""

Python para Economistas: Septima Clase
Autor: Franco Calle

- The Prediction Problem
- Confusion Matrix
- Accuracy Precision Recall
- The receiving operations curve and Area Under the Curve

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn import metrics

# Problema de prediccion binaria:

# Definimos o cargamos base de datos:

candidates = {'Puntaje Examen': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'GPA': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'Experiencia Laboral': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'Admitido': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              } # Fuente "https://datatofish.com/logistic-regression-python/"

df = pd.DataFrame(candidates)

X = df[['Puntaje Examen', 'GPA', 'Experiencia Laboral']]
Y = df[['Admitido']]

clf = LogisticRegression(random_state=0).fit(X, Y) # Definir el problema
clf.coef_
clf.predict(X) #Predecir el modelo
Y_score = clf.predict_proba(X)



# Confusion Matrix:
"""
[True Positives  False Positives
 False Negatives  True Negatives]
"""

# Threshold our probabilities
Y=np.array(Y).reshape(Y.shape[0])
Y_hat=np.array(Y_score[:,1])

# Convert our Thresholded prob to boolean
Y = Y.astype(bool)
Y_hat = (Y_hat>.5).astype(bool)

def getConfusionMatrix(Y, Y_hat):

    TP = Y[Y_hat == True].sum()
    FP = (~Y[Y_hat == True]).sum()
    FN = Y[Y_hat == False].sum()
    TN = (~Y[Y_hat == False]).sum()

    confusionMatrix = np.array([[TP, FP], [FN, TN]])

    return confusionMatrix

#Get confusion matrix using our function
myConfusionMatrix = getConfusionMatrix(Y, Y_hat)

#Get confusion matrix using scickitlearn function
sklearnConfusionMatrix = confusion_matrix(Y, Y_hat)

#Accuracy: % de resultados que el modelo predijo correctamente.
clf.score(X, Y)

#Precision: Cuantos items seleccionados son relevantes
def getPrecision(Y, Y_hat):

    TP = Y[Y_hat == True].sum()
    FP = (~Y[Y_hat == True]).sum()
    precision = TP/(TP+FP)

    return precision

#Recall: Cuantos items relevantes son seleccionados
def getRecall(Y, Y_hat):

    TP = Y[Y_hat == True].sum()
    FN = Y[Y_hat == False].sum()
    recall = TP/(TP+FN)

    return recall


# Usamos las funciones basados en un thresholding de .5:

getPrecision(Y,Y_hat)
getRecall(Y,Y_hat)


# Ahora generamos un loop para ver como cambian las medidas cuando cambiamos el threshold:

def thresholdingPrecisionRecall(Y, Y_score):

    thresholds = np.unique(Y_score)
    precisionList = []
    recallList = []

    for tt in thresholds:

        Y_hat = Y_score > tt
        precision = getPrecision(Y,Y_hat)
        recall = getRecall(Y,Y_hat)
        precisionList.append(precision)
        recallList.append(recall)

    return precisionList, recallList


precisionList, recallList = thresholdingPrecisionRecall(Y, Y_score[:,1])


# Plot Results:
plt.plot(recallList, precisionList)


#Using sklearn module:

disp = plot_precision_recall_curve(LogisticRegression(random_state=0).fit(X,Y), X, Y)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


# The receiving operations curve and Area Under the Curve:

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


tprList, fprList = thresholdingAUC(Y, Y_score[:,1])
AUC = metrics.auc(fprList, tprList)

# Plot the figure:
plt.plot(fprList, tprList, label='ROC curve (area = %0.2f)' % AUC)
plt.plot(fprList,fprList, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
