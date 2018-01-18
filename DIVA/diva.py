import numpy as np
from numpy import NaN, zeros
from local_node import local_node
from master_node import master_node
from multiprocessing.dummy import Pool

from scipy.io import savemat

from time import time

def apply_local_step_get_Y(site) : 
    YtY, w_value = site.local_step_get_Y()
    return YtY, w_value, site

def get_site_info(X, POOL) :
    site_info = POOL.map(apply_local_step_get_Y, X)
    YtY = np.sum([site[0] for site in site_info], 0)
    w_value = np.sum([site[1] for site in site_info])
    return YtY, w_value, [site[2] for site in site_info]

def get_alpha(alpha, new_norm, old_norm, it) :
    if it > 0 :
        if it % 100 == 1 and it>1:
            return 1.0
        else :
            return alpha * (old_norm / new_norm)
    else :
        return alpha

def apply_get_site_grad(info) : 
    # info = (site, sqrtYtYInv)
    site = info[0]
    sqrtYtYInv = info[1]
    new_norm = site.local_step_get_grad(sqrtYtYInv)
    return new_norm, site

def get_site_grad(X, POOL, sqrtYtY, alpha, old_norm, it) :
    sqrtYtYInv = 1.0 / sqrtYtY
    X_sqrtInv = [(x, sqrtYtYInv) for x in X]
    site_info = POOL.map(apply_get_site_grad, X_sqrtInv)
    
    new_norm = np.sum([site[0] for site in site_info])
    alpha = get_alpha(alpha, new_norm, old_norm, it)
    return alpha, new_norm, [site[1] for site in site_info]

def apply_use_site_grad(info) : 
    site  = info[0]
    alpha = info[1]
    term, max_gW = site.local_step_use_grad(alpha)
    return term, max_gW, site

def use_site_grad(X, POOL, alpha) :
    X_alpha = [(x, alpha) for x in X]
    #site_info = POOL.map(apply_use_site_grad, X_alpha)
    site_info = []
    for info in X_alpha :
        site_info.append(apply_use_site_grad(info))
    
    term = max([site[0] for site in site_info])
    max_gW = max([site[1] for site in site_info])
    #return term, max_gW, [site[2] for site in site_info]
    return term, max_gW

def backtrack(X, POOL, c_curr, c_prev, alpha, master, sqrtYtY, old_norm, verbose) :
    back = 1
    while c_curr > c_prev - alpha * (1e-10 * old_norm) :
        alpha *= 0.75
        #term, X = use_site_grad(X, POOL, alpha)
        term, max_gW = use_site_grad(X, POOL, alpha)
        YtY, w_value, X = get_site_info(X, POOL)
        sqrtYtY, c_curr = master.master_step(YtY, w_value, back, 1.0,
                                             alpha, backtrack=True)
        if verbose :
            print " Backtracking : %4d  W Change : %.10f  Cost : %.10f  Alpha : %.10f  gW Norm : %f"%(back, term, c_curr, alpha, max_gW)
        back += 1
    return alpha, c_curr, sqrtYtY, X

def ddiva(X, alpha=1.0, term=1.0, max_iter=2048, term_thresh=1e-4, verbose=False, NAME='checkpoint') :
    '''
    X is a list of local_node objects
        ie X = local_sites
        Each local_node object has two user provided instances: 
            X, the site's data
            W, the site's initial unmixing matrix
    '''
    P = len(X)
    POOL = Pool(P)
    master = master_node()
    KK = []
    c = [np.NaN for it in range(max_iter)]
    
    
    for site in X :
        stuff = site.initiate()
        N =       stuff[0]
        R =       stuff[1]
        KK.append(stuff[2])
    
    master.initiate(KK, False)
    
    YtY, w_value, X = get_site_info(X, POOL)
    sqrtYtY, c[0] = master.master_step(YtY, w_value, 0, term,
                                       alpha, backtrack=False)
    alpha, old_norm, X = get_site_grad(X, POOL, sqrtYtY, alpha, 1.0, 0)
    #alpha, old_norm = get_site_grad(X, POOL, sqrtYtY, alpha, 1.0, 0)
    #term, X = use_site_grad(X, POOL, alpha)
    term, max_gW = use_site_grad(X, POOL, alpha)
    if verbose :
        print " Step : %4d  W Change : %.10f  Cost : %.10f  Alpha : %.10f  gW Norm : %.4f"%(0, term, c[0], alpha, max_gW)
    
    try :
        for it in range(1, max_iter) :
        #for it in range(1, 5) :
            YtY, w_value, X = get_site_info(X, POOL)
            sqrtYtY, c[it] = master.master_step(YtY, w_value, it, term,
                                                alpha, backtrack=False)
            alpha, c[it], sqrtYtY, X = backtrack(X, POOL, c[it], c[it-1], alpha,
                                                 master, sqrtYtY, old_norm,
                                                 verbose)
            alpha, old_norm, X = get_site_grad(X, POOL, sqrtYtY, alpha, old_norm, it)
            #alpha, old_norm = get_site_grad(X, POOL, sqrtYtY, alpha, old_norm, it)
            #term, max_gW, X = use_site_grad(X, POOL, alpha)
            term, max_gW = use_site_grad(X, POOL, alpha)
            if it % 100 == 0 :
                #W = []
                i = 0
                for site in X :
                    savemat(NAME + "_site%04d"%i, {'W':site.save()})
                    i += 1
                #save(NAME,[W,alpha,c[:it]])
            if verbose :
                print " Step : %4d  W Change : %.10f  Cost : %.10f  Alpha : %.10f  gW Norm : %.4f"%(it, term, c[it], alpha, max_gW)
            if term < term_thresh :
                break
        POOL.close()
        POOL.join()
        W = []
        for site in X :
            W.append(site.finish())
    except KeyboardInterrupt :
        POOL.close()
        POOL.join()
        W = []
        for site in X :
            W.append(site.finish())
    
    return W, c[:it], alpha
