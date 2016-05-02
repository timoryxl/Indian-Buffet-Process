import numpy as np
from gibbs_sampler import Update_Z,Update_Kplus,Update_sigmaX,Update_sigmaA,Update_alpha
from original_functions import log_p, IBP_prior, log_p_origin
# Original sampler with either original likelihood or improved likelihood
def sampler(X, niter, sigmaX, sigmaA,alpha, N, D, maxNew, log_p):

#initialization
    count=0
    final_Z=np.zeros((niter,N,20))
    final_K=np.zeros((niter,1))
    final_sigma_X=np.zeros((niter,1))
    final_sigma_A=np.zeros((niter,1))
    final_alpha=np.zeros((niter,1))
    Z, Kplus = IBP_prior(alpha, N)
    
#repeat "niter" number of times
    for j in range(niter):
        final_Z[count,:,0:Kplus] = Z
        final_K[count] = Kplus
        final_sigma_X[count] = sigmaX
        final_sigma_A[count] = sigmaA
        final_alpha[count] = alpha
        count = count + 1

        for i in range(N):
            Z,Kplus = Update_Z(i,X,Kplus,Z,sigmaX,sigmaA,D,N,log_p)
            Kplus,Z = Update_Kplus(X,Z,maxNew,sigmaX, sigmaA,alpha,N,D,Kplus,log_p)
        Log_L1 = log_p(X, Z, sigmaX, sigmaA, Kplus, D, N )
        sigmaX = Update_sigmaX(X, Z, sigmaX, sigmaA, Kplus, D, N,Log_L1)
        sigmaA = Update_sigmaA(X, Z, sigmaX, sigmaA, Kplus, D, N,Log_L1)
        alpha = Update_alpha(N,Kplus)
 
    return(final_Z, final_K, final_sigma_A, final_sigma_X, final_alpha, Z)