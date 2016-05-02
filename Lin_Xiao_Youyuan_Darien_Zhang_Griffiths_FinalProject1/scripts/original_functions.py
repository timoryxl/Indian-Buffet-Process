#prior of IBP
import math
def IBP_prior(alpha, N):
    import numpy as np
    res = np.zeros((N, 1000))

    #First person
    temp = np.random.poisson(alpha)
    if temp > 0:
        res[0,0:temp] = np.ones(temp)

    #the rest with Bernoulli + Poisson
    K_plus = temp
    for i in range(1,N):
        for j in range(K_plus):
            p = np.sum(res[0:i,j])/(i+1)
    #sample as Bernoulli with rate m_k/i
            if np.random.uniform(0,1) < p:
                res[i,j] = 1
    #The "untouched" dishes are got from poisson distribution with rate alpha/i
        temp = np.random.poisson(alpha/(i+1))
    #None zero
        if temp > 0:
    #The "new dishes", silimar logic as the very first person
            res[i, K_plus : K_plus + temp] = np.ones(temp)
    #length of new person's first few "bernoulli" choice 
            K_plus = K_plus + temp
    
    res = res[:,0:K_plus]
    return np.array((res, K_plus))

# define a log likelihood function 
def log_p_origin(X, Z, sigmaX, sigmaA, K, D, N):
    import numpy as np
    return (-1)*np.log(2*np.pi)*N*D*.5 - np.log(sigmaX)*(N-K)*D - np.log(sigmaA)*K*D - .5*D*np.log(np.linalg.det(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K))) -.5/(sigmaX**2)*np.trace( (X.T.dot( np.identity(N)-Z.dot(np.linalg.inv(Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K)).dot(Z.T)) )).dot(X) )


# define a log likelihood function 
def log_p(X, Z, sigmaX, sigmaA, K, D, N):
    import numpy as np
    M = Z.T.dot(Z)+(sigmaX**2/sigmaA**2)*np.identity(K)
    return (-1)*np.log(2*np.pi)*N*D*.5 - np.log(sigmaX)*(N-K)*D - np.log(sigmaA)*K*D - .5*D*np.log(np.linalg.det(M)) -.5/(sigmaX**2)*np.trace( (X.T.dot( np.identity(N)-Z.dot(np.linalg.inv(M).dot(Z.T)) )).dot(X) )

def calcInverse(Z,M,i,k):
    import numpy as np
    """New inverse calculation as described in Griffiths and Ghahramani(2011)"""
    M_i = M - M.dot(Z[i,:].T.dot(Z[i,:].dot(M)))/(Z[i,:].dot(M.dot(Z[i,:].T))-1)
    M = M_i - M_i.dot(Z[i,:].T.dot(Z[i,:].dot(M_i)))/(Z[i,:].dot(M_i.dot(Z[i,:].T))+1)
    Inv = M
    return Inv