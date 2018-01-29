import numpy as np
from numpy import dot, abs, max, sum, zeros, identity

def joint_disi(W, A, Wht) :
    # W.shape   = N,N,K,P
    # A.shape   = M,N,K,P
    # Wht.shape = N,M,K,P or N,M if only a single wht for every matrix
    P = len(W)
    N = W[0].shape[0]
    KK = [W[p].shape[2] for p in range(P)]
    B = zeros((N,N))
    try :
        for p in range(P) :
            for k in range(KK[p]) :
                B += abs(dot(W[p][:,:,k], dot(Wht[p][:,:,k], A[p][:,:,k])))
    except IndexError :
        for p in range(P) :
            for k in range(KK[p]) :
                B += abs(dot(W[p][:,:,k], dot(Wht, A[p][:,:,k])))
    
    row_sum = 0
    col_sum = 0
    
    for n in range(N) :
        row_max = max(B[n,:])
        col_max = max(B[:,n])
        
        row_sum += sum(B[n,:] / row_max) - 1
        col_sum += sum(B[:,n] / col_max) - 1
    tot_sum = (row_sum + col_sum) / (2 * N * (N-1))
    return tot_sum

def joint_isi(W, A, Wht=[]) :
    N, N, K = W.shape
    if Wht == [] :
        Wht = zeros((N,N,K))
        for k in range(K) :
            Wht[:,:,k] = identity(N)
    B = zeros((N,N))
    
    try :
        # If different wht for every subject
        for k in range(K) :
            B += abs(dot(W[:,:,k], dot(Wht[:,:,k], A[:,:,k])))
    
    except IndexError :
        # For same wht for every subject
        for k in range(K) :
            B += abs(dot(W[:,:,k], dot(Wht, A[:,:,k])))

    row_sum = 0
    col_sum = 0

    for n in range(N) :
        row_max = max(B[n,:])
        col_max = max(B[:,n])

        row_sum += sum(B[n,:] / row_max) - 1
        col_sum += sum(B[:,n] / col_max) - 1
    tot_sum = (row_sum + col_sum) / (2 * N * (N-1))
    return tot_sum
