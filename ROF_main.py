#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 20 2020
@author: friederike.froeb@mpimet.mpg.de & alexander.winkler@mpimet.mpg.de 
Title: main.py script for PyDnA
"""

## import modules
import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import PyDnA as pda

def da(y, X, nb_runs_x, ctl, reg, cons_test, formule_ic_tls, sample_extr):
    """
    main detection and attribution routine
    
    """
    #%% Options
    # Spherical harmonics truncation
    trunc = 0
    # -> for extracting large sample Z into two samples Z1 and Z2
    sampling_name = sample_extr 
    # fraction of data used in Z2 (the remaining is used in Z1)
    frac_z2 = .5
    #%% Input parameters
    y = np.matrix(y).T
    X = np.matrix(X).T
    Z = np.transpose(np.matrix(ctl))
    nb_runs_x = np.matrix(nb_runs_x)
    # Number of Time steps
    nbts = y.shape[0]
    # Spatial dimension
    n_spa = (trunc+1)**2
    # Spatio_temporal dimension (ie dimension of y)
    n_st = n_spa * nbts
    # number of different forcings
    I = X.shape[1]
    if Z.shape[1] == 1:
        nle = len(Z)
        NZ = nle / n_st
        fl = int(np.floor(Z.shape[0] / y.shape[0]))
        Z = np.transpose(np.reshape(Z[:int(y.shape[0])*fl,], (fl, int(y.shape[0]))))
    else:
        NZ = Z.shape[1]
        
    ## Z1 and Z2 are taken from Z
    ind_z = pda.extract_Z2(NZ, frac_z2, sampling_name)
    ind_z1 = np.argwhere(ind_z == 0)[:, 0]
    ind_z2 = np.argwhere(ind_z == 1)[:, 0]
    Z1 = Z[:, ind_z1]
    Z2 = Z[:, ind_z2]

    #%%  Pre-processing
    #%% Weighting of spherical harmonics (cf Stott & Tett, 1998)
    l = pda.total_wave_number(trunc)
    p = 1. / np.sqrt(2*l-1)
    p.shape = (p.shape[0], 1)
    pml = np.matrix(np.diag(np.reshape(np.transpose(p*np.ones((1, nbts))), (n_st, 1)).squeeze()))
    y = np.dot(pml, y)
    Z1 = np.dot(pml, Z1)
    Z2 = np.dot(pml, Z2)
    X = pml*X

    # Removing of useless dimensions (equivalent to remove one time step;
    # see scientific documentation, Section 2)
    # Spatio-temporal dimension after reduction
    n_red = n_st - n_spa
    U = pda.projfullrank(nbts, n_spa)

    ## Project all input data
    yc = np.dot(U, y)
    Z1c = np.dot(U, Z1)
    Z2c = np.dot(U, Z2)
    Xc = np.dot(U, X)
    proj = np.identity(X.shape[1])

    #%% Statistical estimation
    ## Regularised covariance matrix
    Cf = pda.regC(Z1c.T)
    Cf1 = np.real(spla.inv(Cf))
    #Matrix is singular and may not have a square root. can be ignored
    Cf12 = np.real(spla.inv(spla.sqrtm(Cf)))
    #Matrix is singular and may not have a square root. can be ignored
    Cfp12 = np.real(spla.sqrtm(Cf))

    if reg == 'OLS':
        ## OLS algorithm
        pv_consist = np.nan
        Ft = np.transpose(np.dot(np.dot(spla.inv(np.dot(np.dot(Xc.T, Cf1), Xc)), Xc.T), Cf1))
        beta_hat = np.dot(np.dot(yc.T, Ft), proj.T)
        ## 1-D confidence intervals
        NZ2 = Z2c.shape[1]
        var_valid = np.dot(Z2c, Z2c.T) / NZ2
        var_beta_hat = np.dot(np.dot(np.dot(np.dot(proj, Ft.T), var_valid), Ft), proj.T)
        beta_hat_inf = beta_hat - sps.t.ppf(0.95, NZ2) * np.sqrt(np.diag(var_beta_hat))
        beta_hat_sup = beta_hat + sps.t.ppf(0.95, NZ2) * np.sqrt(np.diag(var_beta_hat))
        ## Consistency check
        # print('Residual Consistency Check')
        epsilon = yc - np.dot(np.dot(Xc, proj.T), beta_hat.T)
        if  cons_test == "OLS_AT99":
            # Formula provided by Allen & Tett (1999)
            d_cons = np.dot(np.dot(epsilon.T, np.linalg.pinv(var_valid)), epsilon) / (n_red - I)
            pv_cons = 1 - sps.f.cdf(d_cons, n_red - I, NZ2)
        elif cons_test == "OLS_Corr":
            # Hotelling Formula
            d_cons = np.dot(np.dot(epsilon.T, np.linalg.pinv(var_valid)), epsilon)/(NZ2*(n_red-I))*(NZ2-n_red+1)
            if NZ2-n_red + 1 > 0:
                pv_cons = 1 - sps.f.cdf(d_cons, n_red - I, NZ2 - n_red + 1)
            else:
                pv_cons = np.nan
        else:
            print('Unknown Cons_test : ', cons_test)

    elif reg == 'TLS':
        ## TLS algorithm
        c0, c1, c2, d_cons, x_tilde_white, y_tilde_white = pda.tls(np.dot(Xc.T, Cf12), np.dot(yc.T, Cf12), np.dot(Z2c.T, Cf12), nb_runs_x, proj, formule_ic_tls)
        x_tilde = np.dot(Cfp12, x_tilde_white.T)
        y_tilde = np.dot(Cfp12, y_tilde_white.T)
        beta_hat = c0.T
        beta_hat_inf = c1.T
        beta_hat_sup = c2.T

        # Consistency check
        print("Residual Consistency Check")
        NZ1 = Z1c.shape[1]
        NZ2 = Z2c.shape[1]

        if  cons_test == 'MC':
            ## Null-distribution sampled via Monte-Carlo simulations
            ## Note: input data here need to be pre-processed, centered, etc.
            ## First, simulate random variables following the null-distribution
            N_cons_mc = 1000
            d_H0_cons = pda.consist_mc_tls(Cf, Xc, nb_runs_x, NZ1, NZ2, N_cons_mc, formule_ic_tls)
            ## Evaluate the p-value from the H0 sample (this uses gke = gaussian kernel estimate)
            pv_cons = pda.gke(d_H0_cons, d_cons)
        elif cons_test == "AS03":
            ## Formula provided by Allen & Stott (2003)
            pv_cons = 1 - sps.f.cdf(d_cons / (n_red-I), n_red-I, NZ2)
        else:
            pv_cons = np.nan

    beta = np.zeros((4, I))
    beta[:-1, :] = np.concatenate((beta_hat_inf, beta_hat, beta_hat_sup))
    beta[-1, 0] = pv_cons
    
    return beta


