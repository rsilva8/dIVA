import numpy as np
import joint_isi as isi

def main() :
    N = 20
    K = 2
    W = np.random.rand(N,N,K)
    A = W.copy()
    for k in range(K) : 
        A[:,:,k] = np.linalg.pinv(W[:,:,k])
    
    print "This should come out to zero \n", isi.joint_ISI(W,A)
    
    A = np.random.rand(N,N,K)
    print "Who knows what this should come out to \n", isi.joint_ISI(W,A)
    
if __name__ == "__main__" :
    main()