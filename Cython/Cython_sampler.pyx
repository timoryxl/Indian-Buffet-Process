import numpy as np
cimport numpy as np
import math
import Cython_functions as func
def cy_sampler(double[:,:] X, int niter, double sigmaX, double sigmaA, double alpha, int N, int D, int maxNew):

    
    #initialization
    cdef int count=0
    cdef int j,i,kNew,kk,new,k
    cdef double[:,:] Z
    cdef double Log_L1,temp_unif
    cdef double Harmonic_N,alphaN, lik, pois, p
    cdef double temp_unif1, Log_L_New, sigmaX_a, U, sigmaA_new,sigmaX_new


    final_Z=np.zeros((niter,N,20))
    final_K=np.zeros((niter,1))
    final_sigma_X=np.zeros((niter,1))
    final_sigma_A=np.zeros((niter,1))
    final_alpha=np.zeros((niter,1))
    Z, Kplus = func.IBP_prior(alpha, N)
    
    #repeat "niter" number of times
    for j in range(niter):
        final_Z[count,:,0:Kplus] = Z
        final_K[count] = Kplus
        final_sigma_X[count] = sigmaX
        final_sigma_A[count] = sigmaA
        final_alpha[count] = alpha
        count = count + 1

        for i in range(N):
            for k in range(Kplus):
                if k >= Kplus:
                    break     
                if Z[i,k] > 0:
                    if (np.sum(Z[:,k])- 1) <= 0:
                        Z[:,k:(Kplus-1)] = Z[:,(k+1):Kplus]
                        Kplus = Kplus-1
                        Z = Z[:,0:Kplus]
                        continue
           

           #M-H algorithm for Z
                P = np.zeros(2)
                #set Z[i,k] = 0 and calculate posterior probability
                Z[i,k] = 0
                P[0] = func.log_p(X, Z, sigmaX, sigmaA, Kplus, D, N) + np.log(N-np.sum(Z[:,k])) - np.log(N)
                #set Z[i,k] = 1 and calculate posterior probability
                Z[i,k] = 1
                P[1] = func.log_p(X, Z,sigmaX, sigmaA, Kplus, D, N)  + np.log(np.sum(Z[:,k])- 1) - np.log(N)

                P = np.exp(P - max(P))
                U = np.random.uniform(0,1)
                if U<(P[1]/(np.sum(P))):
                    Z[i,k] = 1
                else:
                    Z[i,k] = 0   
  

            #M-H algorithm for k
            # Set the number of upper bound as 3
            maxNew = 3
            #Sample number of new features
            prob = np.zeros(maxNew)
            alphaN = alpha/N     
            for kNew in range(maxNew):
                Z_temp = Z
                if kNew > 0:
                    addCols = np.zeros((N,kNew))
                    addCols[i,:] = 1
                    Z_temp = np.hstack((Z_temp, addCols))

                pois = kNew*np.log(alphaN) - alphaN - np.log(math.factorial(kNew))
                kk = Kplus+kNew
                lik = func.log_p(X, Z_temp, sigmaX, sigmaA, kk, D, N)
                prob[kNew] = pois + lik
            prob = np.exp(prob - max(prob))
            prob = prob/sum(prob)

            U = np.random.uniform(0,1)
            p = 0
            kNew=0
            for new in range(maxNew):
                p = p + prob[new]
                if U < p:
                    kNew = new
                    break
            if kNew > 0:
                addCols = np.zeros((N,kNew))
                addCols[i,:] = 1
                Z = np.hstack((Z, addCols))
            Kplus = Kplus + kNew 
        Log_L1 = func.log_p(X, Z, sigmaX, sigmaA, Kplus, D, N )


        #update sigmaX  
        temp_unif = np.random.uniform(0,1)/30
        if np.random.uniform(0,1) < .5:
            sigmaX_new = sigmaX - temp_unif
        else:
            sigmaX_new = sigmaX + temp_unif
    
        Log_L_New = func.log_p(X, Z, sigmaX_new, sigmaA, Kplus, D, N)
        sigmaX_a = np.exp(min(0,Log_L_New-Log_L1))       
        U = np.random.uniform(0,1)
        if U < sigmaX_a:
            sigmaX = sigmaX_new


        #update sigmaA
        temp_unif1 = np.random.uniform(0,1)/30
        if np.random.uniform(0,1) < .5:
            sigmaA_new = sigmaA - temp_unif1
        else:
            sigmaA_new = sigmaA + temp_unif1
    
        Log_L_New = func.log_p(X, Z, sigmaX, sigmaA_new, Kplus, D, N)
        sigmaX_a = np.exp(min(0,Log_L_New-Log_L1))
        U = np.random.uniform(0,1)
        if U < sigmaX_a:
            sigmaA = sigmaA_new
        

        #update alpha
        Harmonic_N = 0.
        for i in range(1, N+1):
            Harmonic_N += 1.0/i
        alpha = np.random.gamma(1+Kplus, 1/(1+Harmonic_N))  
 
    return(final_Z, final_K, final_sigma_A, final_sigma_X, final_alpha, Z)