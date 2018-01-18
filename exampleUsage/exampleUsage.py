import numpy as np
from numpy import load, matmul
from numpy.random import rand
from diva import ddiva
from local_node import local_node
from joint_isi import joint_isi

# Load source data (data), generate a random mixing matrix (A), and generate 
# an initial starting point for the algorithm (W)
data = np.load("dataArray.npy")
A = rand(5,5,4,3)
W = rand(5,5,4,3)

for i in range(3) :
    siteData = matmul(data['site1'], A[:,:,:,0])
    site1 = local_node(data['site1'], W[:,:,:,0])
    
    siteData = matmul(data['site2'], A[:,:,:,1])
    site2 = local_node(data['site2'], W[:,:,:,1])
    
    siteData = matmul(data['site2'], A[:,:,:,2])
    site2 = local_node(data['site2'], W[:,:,:,2])

sites = [site1, site2, site3]

W, cost, alpha = ddiva(sites)

# Test whether or not the unmixing matrices acutally unmix
print( joint_isi(W,A,np.identity(5)) )
