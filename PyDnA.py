#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2018
@author: alexander.winkler@mpimet.mpg.de
Title: Optimal Fingerprinting after Ribes et al., 2009
"""

## import modules
import numpy as np
import scipy.linalg as spla
import scipy.stats as sps

## define functions
##
def eigvalvec(C):
    """
    Eigenvalue / Eigenvector calculation
    """
    ## Compute eigenvalues and eigenvectors
    eigval0, eigvec0 = spla.eigh(C) 
    ## Take real part (to avoid numeric noise, eg small complex numbers)
    if np.max(np.imag(eigval0))/np.max(np.real(eigval0)) > 1e-12:
        print("Matrix is not symmetric")
        return
    ## Check that C is symetric (<=> real eigen-values/-vectors)
    eigval1 = np.real(eigval0)
    eigvec1 = np.real(eigvec0)
    ## Sort in a descending order
    dorder = np.argsort(eigval1)[::-1]
    eigvec = np.flipud(eigvec1)[:, dorder]
    eigval = eigval1[dorder]
    return eigval, eigvec

##
def projfullrank(t, s):
    """
    Projection on full rank matrix
    """
    ## M: the matrix corresponding to the temporal centering
    M = np.eye(t, t) - np.ones([t, t])/float(t)
    ## Eigen-values/-vectors of M; note that rk(M)=T-1, so M has one eigenvalue equal to 0.
    eigval, eigvec = eigvalvec(M)
    ## (T-1) first eigenvectors (ie the ones corresponding to non-zero eigenvalues)
    eigvec = eigvec[:, :t-1].T
    ## The projection matrix P, which consists in S replications of U.
    P = np.zeros([(t-1)*s, t*s])
    for i in range(s):
        P[i:(t-1)*s:s, i:t*s:s] = eigvec
    return P

##
def total_wave_number(n):
    """
    Calculation wave number
    """
    nr = (n+1)**2
    l = np.zeros(nr)
    ir = 1
    l[:n+1] = range(1, n+2)
    ir = ir+n
    for i in range(2, n+2):
        for j in range(i, n+2):
            l[ir] = j
            l[ir+1] = j
            ir = ir+2
    return l

##
def regC(X):
    """
    Calculation regularized CoVariance matrix
    """
    # just to be sure it is a matrix object
    X = np.matrix(X)
    n, p = np.shape(X)
    # Sample covariance
    CE = X.T * X / float(n)		

    Ip = np.eye(p, p)
    # First estimate in L&W
    m = np.trace(CE * Ip) / float(p)	
    XP = CE - m * Ip
    # Second estimate in L&W
    d2 = np.trace(XP * XP.T) / float(p) 	

    bt = []
    for i in range(n):
        Mi = X[i, :].T * X[i, :]
        bt.append(np.trace((Mi - CE) * (Mi - CE).T) / float(p))
    bb2 = 1. / n**2 * np.sum(bt)
    # Third estimate in L&W
    b2 = np.min([bb2, d2])	
    # Fourth estimate in L&W
    a2 = d2 - b2		
    Cr = b2 * m / d2 * Ip + a2 / d2 * CE
    return Cr

##
def extract_Z2(NZ, frac_Z2, sampling_name):
    """
    Z1 and Z2 based on control
    """
    Ind_Z2 = np.zeros((int(NZ), 1))
    NZ2 = int(np.floor(NZ * frac_Z2))
    if sampling_name == 'segment':
        Ind_Z2[0:NZ2] = 1
        print('Z2 : segment 1-'+str(NZ2)+', fraction ~ '+str(NZ2/NZ))
    elif sampling_name == 'regular':
        ix = []
        a = 0
        while a <= NZ - 1./frac_Z2:
            a += 1./frac_Z2
            ix.append(np.int(np.floor(a)) - 1)
        Ind_Z2[ix] = 1
        print('Z2 : regular, fraction ~ '+str(sum(Ind_Z2)/NZ))
    elif sampling_name == 'random':
        u = np.random.normal(0, 1, size=(NZ, 1))
        z = np.argsort(u, axis=0)[::-1]
        Ind_Z2[z[0:NZ2]] = 1
        print('Z2 : random, fraction ~ '+str(sum(Ind_Z2)/NZ))
    else:
        print('Unknown sampling_name.')
    return Ind_Z2

##
def gke(d_H0, d):
    """
    Silverman's rule of Thumb
    """
    N = len(d_H0)
    h = 1.06 * np.std(d_H0, ddof=1) * N ** (-1./5)	# Silverman's rule of Thumb
    onem = sps.norm.cdf(d, d_H0, h)
    pvi = 1 - onem
    pv = np.sum(pvi)/N

    return pv

##
def tls(X, Y, Z2, nX, PROJ, Formule_IC_TLS):
    """
    TLS routine
    """
    n = Y.shape[1]
    m = X.shape[0]
    # Check sizes of X and Y
    if Y.shape[1] != X.shape[1]:
        print('Error in TLS: size of inputs X, Y.')
        return
    # Normalise the variance of X
    X = np.multiply(X, (np.sqrt(nX).T * np.ones((1, n))))
    if X.shape[0] == 1: # adjusted
        DnX = np.sqrt(nX).squeeze()
    else:
        DnX = np.diag(np.sqrt(nX).A1)
    # Computation of beta_hat
    #--------------------------
    # TLS fit via svd...
    M = np.vstack([X, Y])
    U, D, V = np.linalg.svd(M)
    V = V.T
    
    # Consider the "smallest" singular vector
    Uk = U[:, -1]
    Uk_proj = np.vstack([PROJ * DnX * Uk[:-1], Uk[-1]])
    # Computes beta_hat
    beta_hat = - Uk_proj[:-1] / Uk_proj[-1]
    # instantiate array for beta uncertainty estimates
    beta_hat_inf = np.zeros(beta_hat.shape)
    beta_hat_sup = np.zeros(beta_hat.shape)
    # Reconstructed data
    D_tilde = np.matrix(np.zeros(M.shape))
    np.fill_diagonal(D_tilde, D)
    D_tilde[m, m] = 0
    Z_tilde = U * D_tilde * V.T
    X_tilde = Z_tilde[0:m, :] / (np.dot(np.sqrt(nX).T, np.ones((1, n))))
    Y_tilde = Z_tilde[m, :]
    # Computation of Confidence Intervals
    #--------------------------------------
    # The square singular values (denoted by lambda in AS03)
    d = D**2
    # Computation of corrected singular value (cf Eq 34 in AS03)
    d_hat = np.zeros(d.shape)
    NAZv = Z2.shape[0]
    for i in range(len(d)):
        vi = V[:, i].T
        if Formule_IC_TLS == "AS03":
            # Formule Allen & Stott (2003)
            d_hat[i] = d[i] / np.dot(np.dot(np.dot(vi, Z2.T), Z2 / NAZv), vi.T)
        elif Formule_IC_TLS == "ODP": 
             # Formule ODP (Allen, Stone, etc)
            d_hat[i] = d[i] / np.dot(np.power(vi, 2), np.sum(np.power(Z2, 2), axis=0).T / NAZv)
        else:
            print('tls_v1.sci : unknown formula for computation of TLS CI.')
    # The "last" corrected singular value will be used in the Residual Consistency Check
    d_cons = d_hat[-1]
    # Threshold of the Fisher distribution, used for CI computation (cf Eq 36-37 in AS03)
    seuil_1d = np.sqrt(sps.f.ppf(0.9, 1, NAZv))
    # In order to compute CI, we need to run through the m-sphere (cf Eq 30 in AS03)
    # Nb of pts on the (m-)sphère...
    npt = 1000	
    if m == 1:
        Pts = np.array([[1], [-1]])
    else:
        Pts_R = np.random.normal(0, 1, size=(npt, m))
        # The points on the sphere
        Pts = Pts_R / (np.sqrt(np.sum(Pts_R ** 2, axis=1).reshape((npt, 1)) * np.ones((1, m))))
    # delta_d_hat provides the diagonal of the matrix used in Eq 36 in AS03
    delta_d_hat = d_hat - np.min(d_hat)
    # following notation of Eq 30 in AS03
    a = seuil_1d * Pts			
    arg_min = np.nan
    arg_max = np.nan
    # Check that 0 is not reached before the last index of delta_d_hat:
    if True not in (delta_d_hat[:-1] == 0):
        b_m1 = a / np.dot(np.ones((Pts.shape[0], 1)), np.sqrt(delta_d_hat[:-1]).reshape((1, delta_d_hat[:-1].shape[0])))
        # following notation of Eq 31 in AS03
        #b_m2 = np.sqrt(1 - np.sum(b_m1**2, axis=1))	
        b_m2 = np.matrix(np.sqrt(1 - np.sum(b_m1**2, axis=1))).T
        # b_m2 need to be strctly positive, otherwise the CI will be unbounded
        if (False in np.isreal(b_m2)) | (True in (b_m2 == 0)) | (True in np.isnan(b_m2)):
            print('Unbounded CI (2)', np.max(np.imag(b_m2)))
            beta_hat_inf += np.nan
            beta_hat_sup += np.nan
        else:
            # Then in order to CI that include +/- infinity, the computation are made in terms of angles,
            # based on complex numbers (this is a descrepancy with ODP)
            V_pts = np.dot(np.column_stack([b_m1, b_m2]), U.T)
            V_pts_proj = np.column_stack([np.dot(np.dot(V_pts[:, :-1], DnX), PROJ.T), V_pts[:, -1]])
            for i in range(m):
                Vc_2d_pts = V_pts_proj[:, i] + V_pts_proj[:, -1] * 1j 
                Vc_2d_ref = Uk_proj[i] + Uk_proj[-1] * 1j
                Vprod_2d = Vc_2d_pts / Vc_2d_ref
                arg = np.sort(np.imag(np.log(Vprod_2d)), axis=0)
                delta_arg_min = arg[0]
                delta_arg_max = arg[-1]
                Delta_max_1 = np.max(arg[1:] - arg[:-1])
                k1 = np.argmax(arg[1:] - arg[:-1])
                Delta_max = np.max([Delta_max_1, arg[0] - arg[-1] + 2 * np.pi])
                k2 = np.argmax([Delta_max_1, arg[0] - arg[-1] + 2 * np.pi])
                if Delta_max < np.pi:
                    beta_hat_inf[i] = np.nan
                    beta_hat_sup[i] = np.nan
                else:
                    if k2 != 1:
                        print("Warning k2")
                    arg_ref = np.imag(np.log(Vc_2d_ref))
                    arg_min = delta_arg_min + arg_ref
                    arg_max = delta_arg_max + arg_ref
                    beta_hat_inf[i] = -1 / np.tan(arg_min)
                    beta_hat_sup[i] = -1 / np.tan(arg_max)
    else:    
        # If 0 is reached before last index of delta_d_hat, the CI will be unbounded
        print('Unbounded CI (1)')
        beta_hat_inf += np.nan
        beta_hat_sup += np.nan
    return beta_hat, beta_hat_inf, beta_hat_sup, d_cons, X_tilde, Y_tilde

##
def consist_mc_tls(Sigma, X0, nb_runs_X, n1, n2, N, Formula):
    """
    Consistency check TLS
    """
    # Check that Sigma is a square matrix
    n = Sigma.shape[0]
    if (Sigma.shape[1] != n) | (X0.shape[0] != n):
        print("Error of size in consist_mc_tls.sci")
        return
    # Number of external forcings considered
    k = X0.shape[1]
    # Initial value of beta for the Monte Carlo simulations
    beta0 = np.ones((k,1))
    # Monte Carlo simulations
    #-------------------------
    Sigma12 = spla.sqrtm(Sigma)
    d_cons_H0 = np.zeros((N,1))
    for i in range(N):
        # Virtual observations Y
        Yt = np.dot(X0, beta0)
        Y = Yt + np.dot(Sigma12, np.random.normal(0, 1, size=(n, 1)))
        # Virtual noised response patterns X
        X = X0 + np.dot(Sigma12, np.random.normal(0, 1, size=(n, k)) / (np.ones(Yt.shape) * np.sqrt(nb_runs_X)))
        # Variance normalised X
        Xc = np.multiply((np.dot(np.ones(Yt.shape), np.sqrt(nb_runs_X))), X)
        # Virtual independent samples of pure internal variability, Z1 and Z2
        Z1 = np.dot(Sigma12, np.random.normal(0, 1, size=(n, n1)))
        Z2 = np.dot(Sigma12, np.random.normal(0, 1, size=(n, n2)))
        # Virtual estimated covariance matrix (based on Z1 only)
        C1_hat = regC(Z1.T)
        C12 = spla.inv(spla.sqrtm(C1_hat))
        # The following emulates the TLS algorithm and computes the variable used in the RCC (which is written in d_cons_H0). See also tls_v1.sci.
        # Xc and Y are prewhitened
        M = np.dot(C12, np.column_stack([Xc, Y]))
        U, D, V = np.linalg.svd(M.T)
        V = V.T
        d = D**2
        nd = len(d)
        vi = V[:,nd].T

        if Formula == "AS03":
            # Z2 is prewhitened
            Z2w = np.dot(C12, Z2).T
            # Formule Allen & Stott (2003)
            d_cons_H0[i,0] = d[nd-1] / np.dot(np.dot(np.dot(vi, Z2w.T), Z2w / n2), vi.T)
        elif Formula == "ODP":
            # Z2 is prewhitened
            Z2w = np.dot(C12, Z2).T
            # Formule ODP (Allen, Stone, etc)
            d_cons_H0[i,0] = d[nd-1] / np.dot(np.power(vi,2), np.sum(np.power(Z2w, 2), axis=0).T / n2)
        else:
            print("consist_mc_tls.sci : unknown formula for computation of RCC.")

    return d_cons_H0

