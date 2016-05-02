import time
import numpy as np
from Cython_sampler import cy_sampler
import math
import Cython_functions as func
from Original_sampler import sampler
from gibbs_sampler import Update_Z,Update_Kplus,Update_sigmaX,Update_sigmaA,Update_alpha
from original_functions import log_p, log_p_origin, IBP_prior
X=np.load('data/X_initialized.npy')
niter = 1500
N = 100 
D = 36 
sigmaX = 1.7
sigmaA = 0.5
alpha = 1.0
maxNew = 4
BURN_IN=200
SAMPLE_SIZE= niter-BURN_IN
np.random.seed(124)
t0=time.time()
chain1_Z, chain1_K, chain1_sigma_A, chain1_sigma_X, chain1_alpha, Z1= sampler(X, niter, sigmaX, sigmaA, alpha, N, D, maxNew, log_p_origin)  
t1=time.time()-t0

np.save("data/chain1/chain1Z", chain1_Z)
np.save("data/chain1/chain1K",chain1_K)
np.save("data/chain1/chain1SigmaX", chain1_sigma_X)
np.save("data/chain1/chain1SigmaA",chain1_sigma_A)
np.save("data/chain1/chain1Alpha", chain1_alpha) 
np.random.seed(124)
t0=time.time()
chain2_Z, chain2_K, chain2_sigma_A, chain2_sigma_X, chain2_alpha, Z2= sampler(X, niter, sigmaX, sigmaA, alpha, N, D, maxNew, log_p)  
t2=time.time()-t0

np.save("data/chain2/chain2Z", chain2_Z)
np.save("data/chain2/chain2K",chain2_K)
np.save("data/chain2/chain2SigmaX", chain2_sigma_X)
np.save("data/chain2/chain2SigmaA",chain2_sigma_A)
np.save("data/chain2/chain2Alpha", chain2_alpha) 
np.random.seed(124)
t0=time.time()
chain3_Z, chain3_K, chain3_sigma_A, chain3_sigma_X, chain3_alpha, Z3= cy_sampler(X, niter, sigmaX, sigmaA, alpha, N, D, maxNew)  
t3=time.time()-t0

np.save("data/chain3/chain3Z", chain3_Z)
np.save("data/chain3/chain3K",chain3_K)
np.save("data/chain3/chain3SigmaX", chain3_sigma_X)
np.save("data/chain3/chain3SigmaA",chain3_sigma_A)
np.save("data/chain3/chain3Alpha", chain3_alpha) 

np.save("data/sampler_time", np.array([t1,t2,t3]))