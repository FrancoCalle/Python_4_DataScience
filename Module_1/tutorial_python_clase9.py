"""

Tercera Clase: Métodos de Machine Learning para Regressión
- Least Squares
- Ridge
- Lasso
- KNN regression
- R-squared, MSE

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

def generageData():
    np.random.seed(0)
    X = np.random.randn(100,2)
    beta = np.array([.9, -.3])
    e = np.random.randn(100,)
    Y = np.dot(X,beta.transpose()) + e
    return X, Y

X,Y = generageData()
sns.scatterplot(X[:,0], Y)


# Apply least square: https://scikit-learn.org/stable/modules/linear_model.html
reg = linear_model.LinearRegression()
reg.fit(X,Y)
reg.coef_
Y_hat_OLS = reg.predict(X)

sns.scatterplot(Y_hat,Y)
plt.plot(np.linspace(Y.min(), Y.max(), 8),np.linspace(Y.min(), Y.max(), 8),color='red', linestyle='--')
plt.xlabel('Estimated Y')
plt.ylabel('True Y')
plt.title("Linear Regression")
plt.show()


# Apply Lasso: https://scikit-learn.org/stable/modules/linear_model.html#lasso
alpha0 = 0.1
reg = linear_model.Lasso(alpha=alpha0)
reg.fit(X,Y)
reg.coef_
Y_hat_Lasso = reg.predict(X)

sns.scatterplot(Y_hat,Y)
plt.plot(np.linspace(Y.min(), Y.max(), 8),np.linspace(Y.min(), Y.max(), 8),color='red', linestyle='--')
plt.xlabel('Estimated Y')
plt.ylabel('True Y')
plt.title("Lasso Regression")
plt.show()


# Apply Ridge: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
alpha0 = 0.2
reg = linear_model.Ridge(alpha=alpha0)
reg.fit(X,Y)
reg.coef_
Y_hat_Ridge = reg.predict(X)

sns.scatterplot(Y_hat,Y)
plt.plot(np.linspace(Y.min(), Y.max(), 8),np.linspace(Y.min(), Y.max(), 8),color='red', linestyle='--')
plt.xlabel('Estimated Y')
plt.ylabel('True Y')
plt.title("Ridge Regression")
plt.show()



#Polynomial Regression: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X, Y)
Y_hat_KNN = reg.predict(X)

sns.scatterplot(Y_hat,Y)
plt.plot(np.linspace(Y.min(), Y.max(), 8),np.linspace(Y.min(), Y.max(), 8),color='red', linestyle='--')
plt.xlabel('Estimated Y')
plt.ylabel('True Y')
plt.title("K Nearest Neighbors Regression")
plt.show()


# R squared - MSE
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

MSE_ols = getMSE(Y, Y_hat_OLS)
MSE_lasso = getMSE(Y, Y_hat_Lasso)
MSE_ridge = getMSE(Y, Y_hat_Ridge)
MSE_KNN = getMSE(Y, Y_hat_KNN)


R2_ols = getR2(Y, Y_hat_OLS)
R2_lasso = getR2(Y, Y_hat_Lasso)
R2_ridge = getR2(Y, Y_hat_Ridge)
R2_KNN = getR2(Y, Y_hat_KNN)


plt.bar(('R2_ols', 'R2_lasso', 'R2_ridge', 'R2_KNN'),[R2_ols,R2_lasso,R2_ridge,R2_KNN])
plt.ylim([.5,.7])
