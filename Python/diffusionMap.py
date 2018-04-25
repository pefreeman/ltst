### Diffusion Map Wrapper/Diffuse Function -- Python Implementation ###

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

# Class to format the output of the diffuse function for the diffusion map

class Dmap(object):
    def __init__(self, X, phi0, eigenvals, eigenmult, 
                 psi, phi, neigen, epsilon):
        self.X = X
        self.phi0 = phi0
        self.eigenvals = eigenvals
        self.eigenmult = eigenmult
        self.psi = psi
        self.phi = phi
        self.neigen = neigen
        self.epsilon = epsilon

# D - input distance matrix (represented as a 2D-list)
# neigen - the number of diffusion coordinates to return
# eps - a tuning parameter
# delta - a filtering parameter for creating a sparse matrix (Asp)
def diffuse(D,neigen,eps,delta=10^-10):
    n = D.shape[0]
    K = np.exp(-1 * (D**2) / eps)
    ## next two lines added to match Jaehyeok's DataLinker code
    v = K.sum(axis=1)
    K = K/np.outer(v,v.T)
    ##
    v = K.sum(axis=1)**0.5
    A = K / np.outer(v, v.T)

    ind = np.array([[row,col] 
        for col in range(len(A)) 
            for row in range(len(A[0]))
                if A[row][col] > delta
    ])
    row = ind[..., 0]
    col = ind[..., 1]
    data = A[row, col]
    Asp = csc_matrix((data, (row, col)), shape=(n, n)).toarray()

    neff = min(neigen+1,n)
    eigenvals, eigenvecs = eigsh(Asp, k=neff, which="LA", ncv=n)
    # Python eigsh sorts eigenvalues in order of increasing value, 
    # whereas R arpack sorts them in order of decreasing value.
    # This line will correct for this
    (eigenvals, eigenvecs) = (eigenvals[::-1], eigenvecs[..., ::-1])

    psi = eigenvecs / (eigenvecs[..., 0:1].dot(np.ones((1, neff))))
    phi = eigenvecs * (eigenvecs[..., 0:1].dot(np.ones((1, neff))))

    lam = eigenvals[1:] / (1 - eigenvals[1:])
    lam = np.outer(np.array([1]*n), lam.T)
    X = psi[..., 1:neigen+1] * lam[..., 0:neigen]

    y = Dmap(X=X, phi0=phi[...,0], eigenvals=eigenvals[1:],
             eigenmult=lam[0, 0:neigen], psi=psi, phi=phi,
             neigen=neigen, epsilon=eps)
    return y


## csvPath - path to get to csv data file *from the directory this file is in*
## k - number of nearest neighbors to use in computation
#
#def diffusionMap(csvPath, k, neigen, eps=1):
#
#	# Read in and format data
#	statistics = np.array(readFile(csvPath))
#	statScale = preprocessing.scale(statistics)
#	D = squareform(pdist(statScale))
#
#	nrow = D.shape[0]
#	sigma = np.array([0.0]*nrow)
#	for i in range(nrow):
#		j = np.argsort(D[..., i])[k + 1]
#		sigma[i] = pdist(np.array([statScale[i,...], statScale[j,...]]))
#
#	S = np.outer(np.array(1 / (sigma**0.5)), np.array(1 / (sigma**0.5)))
#	D = D * S
#	dmap = diffuse(D, neigen, eps)