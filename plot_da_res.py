#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: friederike.froeb@mpimet.mpg.de
Title: plot_da_res.py script 
"""

## import modules
import numpy as np
import subprocess
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import random

def plot_summary(var, Year, obs, fp, Ctl, beta, tdect,tattr, sigma, base_per, bg, reg, mem, Cons_test, Ctype, inpath, outpath, ind_0):
    """
    plot SUMMARY

    """
    varlist = subprocess.check_output(['bash', '-c', "grep -w "+var+' '+inpath+"/var_list.csv"]).decode().strip().split(',')
    varname = varlist[1]
    longname = varlist[5]
    modname = varlist[4]   

    ## load data
    aoa = xr.open_dataset(inpath+'/MPI-ESM-LR_hist_AOA_'+modname+'.nc').squeeze()
    rcp = xr.open_dataset(inpath+'/MPI-ESM-LR_hist_esmrcp85_'+modname+'.nc').squeeze()    
    if bg == 'stat':
        ctl = xr.open_dataset(inpath+'/MPI-ESM-LR_esmControl_'+modname+'_r1i1p1.nc').squeeze()       
        ctl = xr.merge([xr.concat([ctl[varname][0:250], ctl[varname][250:500].assign_coords(time=ctl[varname][0:250].coords['time']), ctl[varname][500:750].assign_coords(time=ctl[varname][0:250].coords['time']), ctl[varname][750:1000].assign_coords(time=ctl[varname][0:250].coords['time'])], dim='ens')])
    else:
        ctl = xr.open_dataset(inpath+'/MPI-GE-LR_hist_rcp85_'+modname+'.nc').squeeze()

    ## plotting routine
    plt.figure(facecolor='white', figsize=(14,14))
    ## plot raw data time series
    plt.subplot(221)
    if bg == 'stat':
        for c in range(ctl[varname].shape[0]):
            l1,=plt.plot(aoa['time.year'].values, ctl[varname][c, :], color=tuple(np.ones(3)*(0.4 + 0.2 * np.random.rand(1))), label='MPI-ESM-LR-esmcntl, 1000 years')
           
    else:
        for c in range(ctl[varname].shape[1]):
            l1,=plt.plot(aoa['time.year'].values, ctl[varname][:, c],color=tuple(np.ones(3)*(0.4 + 0.2 * np.random.rand(1))),label='MPI-GE-RCP8.5, 100 Ens.')
        
    for e in range(rcp[varname].shape[1]):
        l2,=plt.plot(aoa['time.year'].values, rcp[varname][:, e], color=tuple(np.array([1, 0.08 * np.random.rand(1).squeeze(), 0])), label='MPI-ESM-LR-RCP8.5, 3 Ens.')
        l3,=plt.plot(aoa['time.year'].values, aoa[varname][:, e], color=tuple(np.array([0, 0.08 * np.random.rand(1).squeeze(), 1])), label='MPI-ESM-LR-AOA, 3 Ens.')
    a=plt.ylim()   
    plt.ylim((a[0], a[1] + 0.1 * (a[1] - a[0])))
    plt.legend(handles=[l1, l2, l3], loc=2, frameon=False, fontsize=12)
    plt.xlim((1850, 2099))   
    plt.title('Global mean time series data', fontsize=12)
    plt.ylabel(longname)
    plt.xlabel('Year')
    
    ## plot anomalies and DA input
    plt.subplot(222)
    r = Ctl.shape[0] / len(range(Year[ind_0] - base_per, 2099))
    for c in range(int(r)):
         m1,=plt.plot(range(Year[ind_0]-base_per, 2099), Ctl[c*(len(range(Year[ind_0]-base_per, 2099))):((c+1)*(len(range(Year[ind_0]-base_per, 2099))))], color=tuple(np.ones(3)*(0.4 + 0.2 * random.random())), label='CTL')
    m2,=plt.plot(Year[:], obs, color=tuple(np.array([1, 0.08 * random.random(), 0])), label='OBS')
    if fp.shape[0]== 2:
        m3,=plt.plot(Year[:], fp[0, :], color=tuple(np.array([0, 0.08 * random.random(),1])), label='FP1')
        m4,=plt.plot(Year[:], fp[1, :], color=tuple(np.array([0.4, 0.08 * random.random(),1])), label='FP2')
        plt.legend(handles=[m1, m2, m3, m4],loc=3,frameon=False,fontsize=12)
    else:
        m3,=plt.plot(Year[:], fp, color=tuple(np.array([0, 0.08 * random.random(),1])), label='FP')
        plt.legend(handles=[m1, m2, m3], loc=3, frameon=False, fontsize=12)

    a=plt.ylim()
    plt.vlines(Year[ind_0], a[0], a[1], linestyles='dotted')    
    plt.xlim((2000, 2100))
    plt.ylim((a[0], a[1]))
    plt.title("Input data for DA analysis, sigma="+str(sigma).replace('[','').replace(']',''), fontsize=12)
    plt.ylabel("$\Delta$ "+longname)
    plt.text(2042, a[0]+(a[1]-a[0])*0.05, ("Base period: "+str(Year[ind_0]-base_per).replace('[','').replace(']','')+"-"+str(Year[ind_0])))
 
    ## plot beta and DA output
    plt.subplot(223)
            
    inferior=beta[:, 0, 0]
    superior=beta[:, 2, 0]
    beta_cor=beta[:, 1, 0]           
            
    plt.fill_between(Year[ind_0:], inferior, superior, where=superior>inferior, facecolor=(0.8,0.8,0.8))       
    plt.plot(Year[ind_0:], inferior, Year[ind_0:] , superior, color=(0.5,0.5,0.5))
    plt.plot(Year[ind_0:], beta_cor, color='r')
    plt.plot([Year[ind_0], Year[-1]], [0, 0], 'k--')   
    plt.plot([Year[ind_0], Year[-1]], [1, 1], 'k--')                
    plt.title('Spatio-temporal fingerprint amplitude $\\beta$', fontsize=12)
    plt.vlines(tdect[0], -2, 3, linestyles='dotted')
    plt.vlines(tattr[0], -2, 3, linestyles='dotted')

    plt.text(2052, -0.75,("Time of detection: "+str(int(tdect[0]))))
    plt.xlim((Year[ind_0], 2099))
    plt.ylabel("$\\beta$ ", fontsize=18) 
    plt.ylim((-2, 3))
            
    ## plot pval and summary
    plt.subplot(224)
    pval_cor=beta[:, 3, 0]
    plt.plot(Year[ind_0:], pval_cor, color='k')
    plt.plot([Year[ind_0], Year[-1]], [0.05, 0.05], 'k--')   
    plt.ylabel("p-value")        
    plt.ylim((0, 1))  
    plt.xlim((Year[ind_0], 2099))   
    plt.title('Residual Consistency Check', fontsize=12)     
    plt.text(2045, 0.9, "Type of regression: "+reg)    
    plt.text(2045, 0.7, "Cons_test: "+Cons_test)
    if Ctype == 'C0':
        plt.text(2045, 0.5, "Filter: Moving average") 
    else:
        plt.text(2045, 0.5, "Filter: Trend-based")          
    ## save figure
    plt.savefig(outpath+"/summary_"+varname+"_"+Ctype+"_"+reg+"_sigma_"+str(sigma).replace('[','').replace(']','')+"_base_"+str(base_per).replace('[','').replace(']','')+"_mem_"+str(mem).replace('[','').replace(']','')+".pdf", bbox_inches='tight')
    #plt.show() 

 

