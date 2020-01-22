#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: friederike.froeb@mpimet.mpg.de
Title: load_fil_data.py script for PyDnA
"""

## import modules
import subprocess
import numpy as np
import xarray as xr
from scipy import signal, stats


def prefilt(var, in_pth, Ctype, sigma, base_per, start_year, bg, mem):
    """
    import and filter data

    var:        variable name
    in_pth:     path data storage
    Ctype:      Filter type; C0 or C1
    sigma:      length filter window
    bg:         stationary or transient null hypothesis
    start_year: start forcing, 2006 for CMIP5
    mem:        assign ensemble member as obs (1, 2, or 3)
    
    returns:
    Year:       time period of OBS, FP, CTL
    obs:        OBS for DA input
    fp:         FP for DA input
    nx:         number of ens member in FP for DA input
    ctl:        control climate for DA input

    """
    varlist = subprocess.check_output(['bash', '-c', "grep -w "+var+' '+in_pth+"/var_list.csv"]).decode().strip().split(',')
    varname = varlist[1]
    modname = varlist[4]
    
    ## load data
    aoa = xr.open_dataset(in_pth+'/MPI-ESM-LR_hist_AOA_'+modname+'.nc').squeeze()
    rcp = xr.open_dataset(in_pth+'/MPI-ESM-LR_hist_esmrcp85_'+modname+'.nc').squeeze()    
    if bg == 'stat':
        ctl = xr.open_dataset(in_pth+'/MPI-ESM-LR_esmControl_'+modname+'_r1i1p1.nc').squeeze()
    else:
        ctl = xr.open_dataset(in_pth+'/MPI-GE-LR_hist_rcp85_'+modname+'.nc').squeeze()
  
    Year = aoa['time.year'].values
    ind_0 = int(np.where(Year == start_year)[0])

    if bg == 'stat':        
        ## calculate anomaly w.r.t. period of time prior to start_year
        ctl_anom = ctl[varname] - np.nanmean(ctl[varname])
        aoa_anom = aoa[varname] - np.mean(rcp[varname][ind_0 - base_per:ind_0, :], axis=0)
        rcp_anom = rcp[varname] - np.mean(rcp[varname][ind_0 - base_per:ind_0, :], axis=0)
    else:    
        ## calculate anomaly by removing filtered ensemble mean / forced trend
        aoa_anom = aoa[varname] - np.tile(signal.filtfilt(np.ones((2))/2, 1, np.nanmean(rcp[varname], axis=1), axis=0, method='gust'), (aoa[varname].shape[1], 1)).T
        rcp_anom = rcp[varname] - np.tile(signal.filtfilt(np.ones((2))/2, 1, np.nanmean(rcp[varname], axis=1), axis=0, method='gust'), (aoa[varname].shape[1], 1)).T
        ctl_anom = ctl[varname] - np.tile(np.nanmean(ctl[varname], axis=1), (ctl[varname].shape[1],1)).T
        
    ## assign ensemble members as obs or fingerprint
    ctl = ctl_anom.values 
    if mem == 1:
        obs = aoa_anom[:, 0]
        fp1 = np.mean(aoa_anom[:, 1:], axis=1)
    elif mem == 2:
        obs = aoa_anom[:, 1]
        fp1 = (aoa_anom[:, 0] + aoa_anom[:, -1]) / 2
    else:
        obs = aoa_anom[:, -1]
        fp1 = np.mean(aoa_anom[:, :1], axis=1)
    fp2 = np.mean(rcp_anom, axis=1)

    ## filter data            
    obs_ad = np.empty((obs.shape))
    fp1_ad = np.empty((fp1.shape))
    fp2_ad = np.empty((fp2.shape))
    ctl_ad = np.empty((ctl.shape))    
    if Ctype == 'C0':
        ## moving average filter 
        ctl_ad = signal.lfilter(np.ones((sigma)) / sigma, 1, ctl)  
        obs_ad = signal.lfilter(np.ones((sigma)) / sigma, 1, obs)  
        fp1_ad = signal.lfilter(np.ones((sigma)) / sigma, 1, fp1)  
        fp2_ad = signal.lfilter(np.ones((sigma)) / sigma, 1, fp2)          
    elif Ctype == 'C1':
        ## trend-based filter
        for i in range(sigma, obs.shape[0]-1):
            slope, intercept, _, _, _ = stats.linregress(np.array(range(i - sigma, i)), obs[i - sigma:i])
            obs_ad[i+1] = slope * (i+1) + intercept           
            slope, intercept, _, _, _ = stats.linregress(np.array(range(i - sigma, i)), fp1[i - sigma:i])
            fp1_ad[i+1] = slope * (i+1) + intercept            
            slope, intercept, _, _, _ = stats.linregress(np.array(range(i - sigma, i)), fp2[i - sigma:i])
            fp2_ad[i+1] = slope * (i+1) + intercept          
        for j in range(sigma, ctl.shape[0]-1):
            if bg == 'stat':
                slope, intercept, _, _, _ = stats.linregress(np.array(range(j - sigma, j)), ctl[j - sigma:j])
                ctl_ad[j+1] = slope * (j+1) + intercept 
            else:
                for e in range(100):
                    slope, intercept, _, _, _ = stats.linregress(np.array(range(j - sigma, j)), ctl[j - sigma:j, e])
                    ctl_ad[j+1, e] = slope * (j+1) + intercept 
    obs = obs_ad
    fp1 = fp1_ad
    fp2 = fp2_ad
    ctl = ctl_ad
                  
    ## causal filter        
    obs[:int(sigma)+1] = np.nan
    fp1[:int(sigma)+1] = np.nan
    fp2[:int(sigma)+1] = np.nan
    if bg == 'stat':       
        fp = np.array(([fp1, fp2]))
        nx = np.array(([2, 3]))
        ctl = ctl[int(sigma)+1:]
    else:
        fp = fp1
        nx = np.array(([2]))
        ctl = np.reshape(ctl[int(sigma)+1:], (ctl[int(sigma)+1:].shape[0]*ctl[int(sigma)+1:].shape[1],1))
    return Year, obs, fp, nx, ctl

def timedec(beta, year, ind_0):
    """
    estimate time of detection on global scale

    """
    # estimate time of detection; i.e. beta_inf greater than zero
    ind_dect = np.empty((beta.shape[2]))
    tdect = np.empty((beta.shape[2]))
    ind_dect[:] = np.nan
    tdect[:] = np.nan
    for f in range(beta.shape[2]):
        temp1 = np.asarray(np.where(np.isnan(beta[:, 0, f])))
        temp2 = np.asarray(np.where(beta[:, 0, f] > beta[:, 2, f]))
        if temp1.size == 0:
            temp1 = np.empty((1, 1))
            temp1[:] = np.nan
        if temp2.size == 0:
            temp2 = np.empty((1, 1))
            temp2[:] = np.nan
        if np.isnan(temp1).all() and np.isnan(temp2).all():
            ind_remove = np.array([[0]])
        else:
            ind_remove = np.nanmax([np.nanmax(temp1), np.nanmax(temp2)]).astype(int) + np.array([[1]])                

        ind_remove = ind_remove.squeeze()
        inferior = beta[ind_remove:, 0, f]
        betacal = beta[ind_remove:, 1, f]
        if len(inferior[np.isnan(inferior)]) == beta.shape[0]:
            ind_dect[f] = np.nan
            tdect[f] = np.nan
        else:
            t = np.roll(inferior, -1) * inferior < 0
            # remove last entry
            cross_zero = np.asarray(np.where(t[:-1]))
            if cross_zero.shape[1] == 0 & np.all(inferior[~np.isnan(inferior)] > 0):
                # all entrys greater than zero
                if np.sum(betacal) == beta.shape[0]:
                    ind_dect[f] = np.nan
                    tdect[f] = -99
                else:
                    ind_dect[f] = 1 + ind_remove
                    tdect[f] = year[ind_0 + ind_remove - 1]
            else:
                ind_dect[f] = np.max(cross_zero) + 1
                tdect[f] = year[ind_remove + ind_0 + np.max(cross_zero).astype(int) + 1]
    return ind_dect, tdect

def timeattr(beta, year, ind_dect, ind_0):
    """
    estimate time of attribution

    """
    # estimate time of attribution; i.e. beta_inf greater than zero and includes 1
    ind_attr = np.empty((beta.shape[2]))
    tattr = np.empty((beta.shape[2]))
    ind_attr[:] = np.nan
    tattr[:] = np.nan
    for f in range(beta.shape[2]):
        superior = beta[:, 2, f]
        betacal = beta[:, 1, f]
        inferior = beta[:, 0, f]
        if np.isnan(ind_dect[f]):
            ind_attr[f] = np.nan
            tattr[f] = np.nan
        else:
            ind_attr_temp = np.array([all(t) for t in zip((superior > 1), (inferior > 0))])
            ind_attr_temp[0:int(ind_dect[f])] = False
            ind_attr_temp = np.array(np.where(ind_attr_temp))
            if ind_attr_temp.size == 0:
                ind_attr[f] = np.nan
                tattr[f] = np.nan
            else:
                ind_attr[f] = ind_attr_temp[0, 0]
                tattr[f] = year[ind_0+ind_attr_temp[0, 0]]
    return ind_attr, tattr
