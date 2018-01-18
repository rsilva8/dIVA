import numpy as np
from numpy import dot, log, zeros, transpose, sum, diag
from numpy.random import rand
from numpy.linalg import det, qr
 
from multiprocessing.dummy import Pool

class local_node() :
    '''
    Attributes:
        X = The original data (replaced by pca whitening process)
        Y = the current approximation to S, X=A*S
        W = the current unmixing matrix
    '''
    def __init__(self, X, W) :
        self.X = X
        self.Y = X*0
        self.W = W
        self.POOL = Pool(X.shape[2])
    
    def initiate(self) :
        return self.X.shape
    
    def local_step_get_Y(self) :
        K = self.X.shape[2]
        #self.Y, YtY = compute_Y(self.X, self.Y, self.W)
        #YtY = compute_Y(self.X, self.Y, self.W)
        self.Y, YtY = compute_Y(self.X, self.W, self.POOL)
        w_value = 0
        for k in range(K) :
            w_value += np.linalg.slogdet(self.W[:,:,k])[1]
        return YtY, w_value
    
    def local_step_get_grad(self, sqrtYtYInv) :
        # Computes local gradient
        self.gW = gradient(self.Y, self.W, sqrtYtYInv)
        self.W_old = self.W.copy()
        norm = get_norm(self.gW)
        return norm
    
    def local_step_use_grad(self, alpha) :
        self.W = self.W_old + alpha * self.gW
        term = get_term(self.W, self.W_old)
        inft_norm = np.max(np.abs(self.gW))
        return term, inft_norm
    
    #def reset_W(self) :
    #    self.W = self.W_old.copy()
    
    def save(self) :
        return self.W
    
    def finish(self) :
        self.POOL.close()
        self.POOL.join()
        return self.W

def get_term(W, W_old) :
    diff = W - W_old
    K = diff.shape[2]
    term = max([np.linalg.norm(diff[:,:,k]) for k in range(K)])
    return term

def get_norm(gW) :
    return np.sum(gW*gW) 

def apply_compute_Y(info) :
    X = info[0]
    W = info[1]
    Y = np.dot(W,X)
    return Y

#def compute_Y(X, Y, W) :
def compute_Y(X, W, POOL) :
    #NOTE: Y is a method which is being modified. It will be 
    # # modified outside the function too.
    #N,R,K = X.shape
    #for k in range(K) :
    #    Y[:,:,k] = np.dot(W[:,:,k], X[:,:,k])
    y = POOL.map(apply_compute_Y, [(X[:,:,k],W[:,:,k]) for k in range(X.shape[2])])
    Y = np.zeros((X.shape))
    for k in range(X.shape[2]) :
        Y[:,:,k] = y[k]
    YtY = np.sum(Y*Y, 2)
    return Y, YtY
    #return YtY

def gradient(Y, W, sqrtYtYInv) :
    _, R, K = Y.shape
    dW = W.copy()
    for k in range(K) :
        phi = sqrtYtYInv * Y[:,:,k]
        dW[:,:,k] = W[:,:,k] - np.dot( np.dot(phi, transpose(Y[:,:,k]) / R), W[:,:,k])
    
    return dW
