import numpy as np

cimport numpy as np

def IBP_prior(double alpha, int N):
    cdef double p
    cdef int i,j,K_plus,temp
    cdef np.ndarray res
    
    res = np.zeros((N, 1000))
    
    temp = np.random.poisson(alpha)
    if temp>0:
        res[0,0:temp] = np.ones(temp)

    K_plus = temp
    for i in range(1,N):
        for j in range(K_plus):
            p = np.sum(res[0:i,j])/(i+1)
            if np.random.uniform(0,1) < p:
                res[i,j] = 1
        temp = np.random.poisson(alpha/(i+1))
        if temp > 0:
            res[i, K_plus : K_plus + temp] = np.ones(temp)
            K_plus = K_plus + temp
    
    res = res[:,0:K_plus]
    return np.array((res, K_plus))

#log p(X|Z,σ_x ,σ_A )
def log_p(X, Z, double sigmaX, double sigmaA, int K, int D, int N):
    M = Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K)
    return (-0.5)*N*D*np.log(2*np.pi) - (N-K)*D*np.log(sigmaX) - K*D*np.log(sigmaA) -0.5*D*np.log(np.linalg.det(M)) -0.5/(sigmaX**2)*np.trace((X.T.dot(np.identity(N)-Z.dot(np.linalg.inv(M).dot(Z.T)))).dot(X))