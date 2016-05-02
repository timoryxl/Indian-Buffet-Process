from original_functions import IBP_prior, calcInverse
import numpy as np
import time
#check whether the inversion method described by Griffiths and Ghahramani(2005) works
X=np.load('data/X_initialized.npy')
N=X.shape[0]
D=X.shape[1]
sigmaX=1.
sigmaA=1.
alpha=1.
i=2
k=1
Z,K = IBP_prior(alpha,N)
M = np.linalg.inv(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K))
Z[i,k] = 1 - Z[i,k]
loops = 1000
t_calculate_inverse = np.zeros(loops)
for l in range(loops):
    t0=time.time()
    calcInverse(Z,M,i,k)
    t1=time.time()
    t_calculate_inverse[l]=t1-t0
mean_t_calculate_inverse = round(np.mean(t_calculate_inverse),7)


t_linalg_inverse = np.zeros(loops)
for l in range(loops):
    t0=time.time()
    np.linalg.inv(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K))
    t1=time.time()
    t_linalg_inverse[l]=t1-t0
mean_t_linalg_inverse= round(np.mean(t_linalg_inverse),7)


aaaa = calcInverse(Z,M,i,k)
bbbb = np.linalg.inv(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K))

results = np.array([aaaa,bbbb,mean_t_calculate_inverse,t_linalg_inverse])
np.save("data/inverse_compare_results",results)