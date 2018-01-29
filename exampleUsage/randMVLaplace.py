import numpy as np
from numpy.random import multivariate_normal, exponential
from scipy.linalg import det, sqrtm



def randMVLaplace(size, mean=[], variance=[]) :
    # size = (N,M)
    # generates M multivariate laplacian random variables in N dimensions
    if mean==[] :
        mean = np.zeros(size[0])
    else :
        assert (size[0],) == mean.shape
    if variance==[] :
        variance = np.identity(size[0])
    else :
        assert (size[0], size[0]) == variance.shape
    
    N,M = size
    X = np.zeros((N,M))
    for m in range(M) :
        X[:,m] = generateMVLaplace(N,mean,variance)
    
    return X

def generateMVLaplace(N,mean,variance) :
    
    X = multivariate_normal(np.zeros(N), np.identity(N))
    scalingFactor = det(variance) ** (1.0/N)
    variance /= scalingFactor
    Z = exponential()
    var = sqrtm(variance)

    return mean + np.sqrt(Z) * np.dot(var, X)

