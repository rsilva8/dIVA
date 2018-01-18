import numpy as np
from numpy import sqrt, sum, zeros, dot

class master_node() :
    '''
    N = number of components
    R = number of samples
    KK = The total number of subjects involved in algorithm
    '''
    def initiate(self, KK, verbose) :
        self.KK = sum(KK)
        self.verbose = verbose
    
    def master_step(self, YtY, w_value, it, term, alpha, backtrack) :
        '''
        Take in current values for YtY, log(abs(det(W))) and output
            cost, whether or not to terminate
        '''
        sqrtYtY = np.sqrt(YtY)
        #sqrtYtYInv = 1 / sqrtYtY
        cost = master_cost(w_value, sqrtYtY, self.KK)
        if self.verbose :
            if backtrack==True :
                print "  Backtrack : %4d  Cost : %.10f  Alpha : %.10f"%(it, cost, alpha)
            else :
                print "  Step : %4d  Cost : %.10f  Alpha : %.10f  W Change : %.10f" % (it, cost, alpha, term)
        #return sqrtYtYInv, cost
        return sqrtYtY, cost

def master_cost(w_value, sqrtYtY, KK):
    N,R  = sqrtYtY.shape
    cost = np.sum(sqrtYtY) / R
    cost -= w_value
    #cost += w_value
    #cost /= (N*KK)
    return cost
