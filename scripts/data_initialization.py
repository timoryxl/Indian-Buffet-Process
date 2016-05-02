np.random.seed(123)
N = 100 
K_plus = 4 
D = 36 
sigmaX = 0.5
sigmaA = 0.5
# Simulated data
A = np.array(( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,1, 1, 1,0, 0, 0,  1, 0, 1, 0, 0, 0, 1, 1, 1,    \
             1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0 ,0, 0, \
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,0, 0, 0, 0,1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).reshape(4, D)

Z_zero = np.zeros((N, K_plus))
X = np.zeros((N, D))
I = np.identity(D) * sigmaX

for i in range(N):
    Z_zero[i,:] = (np.random.uniform(0,1,K_plus) > .5).astype(int)
    while (np.sum(Z_zero[i,:]) == 0):
        Z_zero[i,:] = (np.random.uniform(0,1,K_plus) > .5).astype(int)
    X[i,:] = np.random.normal(0,1, (1,D)).dot(I)+Z_zero[i,:].dot(A)

np.save("data/data_initialized", X)