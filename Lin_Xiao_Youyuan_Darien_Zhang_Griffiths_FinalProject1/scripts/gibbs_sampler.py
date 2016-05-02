import numpy as np
import math
def Update_alpha(N,Kplus):
    Harmonic_N = 0.
    for i in range(1, N+1):
        Harmonic_N += 1.0/i
    alpha = np.random.gamma(1+Kplus, 1/(1+Harmonic_N))
    return alpha

def Update_sigmaA(X, Z, sigmaX, sigmaA, Kplus, D, N,Log_L1):
    temp_unif1 = np.random.uniform(0,1)/30
    if np.random.uniform(0,1) < .5:
        sigmaA_new = sigmaA - temp_unif1
    else:
        sigmaA_new = sigmaA + temp_unif1

    Log_L_New = log_p(X, Z, sigmaX, sigmaA_new, Kplus, D, N)
    sigmaX_a = np.exp(min(0,Log_L_New-Log_L1))
    U = np.random.uniform(0,1)
    if U < sigmaX_a:
        sigmaA = sigmaA_new
    return sigmaA

def Update_sigmaX(X, Z, sigmaX, sigmaA, Kplus, D, N,Log_L1):
    temp_unif = np.random.uniform(0,1)/30
    if np.random.uniform(0,1) < .5:
        sigmaX_new = sigmaX - temp_unif
    else:
        sigmaX_new = sigmaX + temp_unif
    Log_L_New = log_p(X, Z, sigmaX_new, sigmaA, Kplus, D, N)
    sigmaX_a = np.exp(min(0,Log_L_New-Log_L1))       
    U = np.random.uniform(0,1)
    if U < sigmaX_a:
        sigmaX = sigmaX_new
    return(sigmaX)

#M-H algorithm for Kplus
#Sample number of new features
def Update_Kplus(X,Z,maxNew,sigmaX, sigmaA,alpha,N,D,Kplus,log_p):
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
        lik = log_p(X, Z_temp, sigmaX, sigmaA, kk, D, N)
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
    return Kplus,Z

def Update_Z(i,X,Kplus,Z,sigmaX,sigmaA,D,N,log_p):
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
        P[0] = log_p(X, Z, sigmaX, sigmaA, Kplus, D, N) + np.log(N-np.sum(Z[:,k])) - np.log(N)
        #set Z[i,k] = 1 and calculate posterior probability
        Z[i,k] = 1
        P[1] = log_p(X, Z,sigmaX, sigmaA, Kplus, D, N)  + np.log(np.sum(Z[:,k])- 1) - np.log(N)

        P = np.exp(P - max(P))
        U = np.random.uniform(0,1)
        if U<(P[1]/(np.sum(P))):
            Z[i,k] = 1
        else:
            Z[i,k] = 0   
    return Z,Kplus