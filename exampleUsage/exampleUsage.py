import numpy as np
from numpy import load, matmul
from numpy.random import rand
from diva import ddiva
from local_node import local_node
from joint_isi import joint_disi
from randMVLaplace import randMVLaplace


# Load source data (data), generate a random mixing matrix (A), and generate 
# an initial starting point for the algorithm (W)a
S = np.zeros((5,5000,4,3))
for i in range(5) :
    x = randMVLaplace((12,5000))
    S[i,:,0,0] = x[0,:]
    S[i,:,0,1] = x[1,:]
    S[i,:,0,2] = x[2,:]
    
    S[i,:,1,0] = x[3,:]
    S[i,:,1,1] = x[4,:]
    S[i,:,1,2] = x[5,:]
    
    S[i,:,2,0] = x[6,:]
    S[i,:,2,1] = x[7,:]
    S[i,:,2,2] = x[8,:]
    
    S[i,:,3,0] = x[9,:]
    S[i,:,3,1] = x[10,:]
    S[i,:,3,2] = x[11,:]



A = rand(5,5,4,3)
W = rand(5,5,4,3)
X = S * 0

for j in range(3) :
    X[:,:,0,j] = np.dot(A[:,:,0,j], S[:,:,0,j])
    X[:,:,1,j] = np.dot(A[:,:,1,j], S[:,:,1,j])
    X[:,:,2,j] = np.dot(A[:,:,2,j], S[:,:,2,j])
    X[:,:,3,j] = np.dot(A[:,:,3,j], S[:,:,3,j])

#siteData = matmul(A[:,:,:,0], S[:,:,:,0])
site1 = local_node(X[:,:,:,0], W[:,:,:,0])

#siteData = matmul(A[:,:,:,1], S[:,:,:,1])
site2 = local_node(X[:,:,:,1], W[:,:,:,1])

#siteData = matmul(A[:,:,:,2], S[:,:,:,2])
site3 = local_node(X[:,:,:,2], W[:,:,:,2])

sites = [site1, site2, site3]

# W is returned as a list of unmixing matrices, one for every site. Now we need
# to put A in the same form
W, cost, alpha = ddiva(sites,verbose=True)
A = [A[:,:,:,i] for i in range(3)]

# Test whether or not the unmixing matrices acutally unmix
print( joint_disi(W,A,np.identity(5)) )
