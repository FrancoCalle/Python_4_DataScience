import numpy as np
import pandas as pd
from scipy import stats, optimize, interpolate


n=1000
k=10


#DGP
X = np.random.rand(n, k)
Y = np.random.uniform(0,1,n) > .8
income = np.random.uniform(0,1,n)
# def probitModel(X,Y):

theta_init = np.zeros(X.shape[1]+2) + 0.01

def objFun(Y,X,parameters):

    theta = parameters[1:X.shape[1]+1]
    μ = parameters[-2]
    σ = parameters[-1]
    Xθ = np.matmul(X,theta)
    logL = np.mean(np.log(scipy.stats.norm.cdf((Xθ - μ)/σ))*Y + np.log(1-scipy.stats.norm.cdf((Xθ - μ)/σ))*(1-Y))

    return -logL

anon_fun = lambda θ: objFun(Y,X,θ)

results = scipy.optimize.minimize(anon_fun, theta_init, method='BFGS')

score = scipy.stats.norm.cdf(np.matmul(X,results.x[1:k+1]))

# apply k-nearest
score_t = score[Y == 1]
score_nt = score[Y == 0]

te_list = []
for ii in range(200):
    k = 5
    dist = (score_t[ii] - score_nt)**2
    idx = np.argpartition(dist, k)[0:3]
    te_i = np.mean(income[Y == 0][idx])
    te_list.append(te_i)


ATE = np.mean(income[Y == 1]) - np.mean(te_list)
