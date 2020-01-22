#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: friederike.froeb@mpimet.mpg.de
Title: process_data.py script for PyDnA
"""

## import modules
import argparse
import multiprocessing
import subprocess
import numpy as np
from load_fil_data import prefilt, timedec, timeattr
from ROF_main import da

####
INPATH = './data'
OUTPATH = './out'

####
class SmartFormatter(argparse.HelpFormatter):
    """
    formatting of help function
    """
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)

PARSER = argparse.ArgumentParser(description='Run DA method', usage='use "python %(prog)s --help" for more information', formatter_class=argparse.RawTextHelpFormatter)
PARSER.add_argument("var", help="Variable name")
PARSER.add_argument("-s", "--sigma", help="Length of filter window", type=int)
PARSER.add_argument("-b", "--base_per", help="Length of base period", type=int)
PARSER.add_argument('-fil', "--Ctype", help="Type of filter: C0 or C1 (default: C0)", choices=['C0', 'C1'], default='C0')
PARSER.add_argument('-bg', "--background", help="Type of background climate: stationary or non-stationary (default: non-stationary)", choices=['stat', 'trans'], default='trans')
PARSER.add_argument('-m', "--member", help="Member siumlation in Obs: 1, 2 or 3 (default: 1)", default=1, type=int)
PARSER.add_argument('-r', "--reg", help="Type of regression: OLS or TLS (default: OLS)", choices=['OLS', 'TLS'], default='TLS')
PARSER.add_argument('-cons', "--Cons_test", choices=['OLS_AT99', 'OLS_Corr', 'AS03', 'MC'], default='OLS_Corr', help="Cons_test: the null-distribution used in the Residual Consistency Check (p-value calculation)\n"
         "In OLS, this may be:\n"
         "\t OLS_AT99: the formula provided by Allen & Tett (1999), \n"
         "\t OLS_Corr: the formula provided by Ribes et al. (2012), default.\n"
         "In TLS, this may be:   \n"
         "\t AS03: the null-distribution parametric formula provided by Allen & Stott (2003),\n"
         "\t MC: Null-distribution evaluated via Monte-Carlo Simulations (Ribes et al., 2012)")
PARSER.add_argument('-f', "--Formule_IC_TLS", choices=['AS03', 'ODP'], default='ODP', help="Formule_IC_TLS: the formula used for computing confidence intervals in TLS\n"
         "\t AS03: the formula provided by Allen & Stott (2003)\n"
         "\t ODP: the formula implemented in Optimal Detection Package (in Nov 2011), default.")
PARSER.add_argument('-sam', "--sample", help="Extracting Z into two samples Z1 and Z2 using regular, random or segment (default: regular)", choices=['regular', 'random', 'segment'], default='regular')         
ARGS = PARSER.parse_args()


####
print("DA ANALYSIS FOR "+ARGS.var+" USING "+ARGS.reg)

####
if ARGS.reg == 'TLS' and (ARGS.Cons_test == 'OLS_AT99' or ARGS.Cons_test == 'OLS_Corr'):
    ARGS.Cons_test = 'AS03'
    print("OBS: TLS requires Cons_test AS03 or MC, set to AS03 (default)")

####
##  set parameters
START_YEAR_FORCING = int(2006)
END_YEAR_FORCING = int(2100)
VARLIST = subprocess.check_output(['bash', '-c', 'grep -w '+ARGS.var+' '+INPATH+'/var_list.csv'])
VARLIST = VARLIST.decode().strip().split(',')
VARNAME = VARLIST[0]

####
##  preprocessing
##  calculate anomaly and filter data

YEAR, OBS, FP, NX, CNTL = prefilt(ARGS.var, INPATH, ARGS.Ctype, np.asarray(ARGS.sigma), np.asarray(ARGS.base_per), START_YEAR_FORCING, ARGS.background, ARGS.member)
IND_0 = int(np.where(YEAR == START_YEAR_FORCING)[0])

####
##  parallelization
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NewPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

####
def run_da(i):
    """
    call DA routine
    """
    if  ARGS.background == 'trans':
        ##  transient null hypothesis
        beta = da(OBS[IND_0-ARGS.base_per+1:IND_0+i+1], FP[IND_0-ARGS.base_per+1:IND_0+i+1], NX, CNTL.T, ARGS.reg, ARGS.Cons_test, ARGS.Formule_IC_TLS, ARGS.sample)
    else:
        ##  stationary null hypothesis
        beta = da(OBS[IND_0-ARGS.base_per+1:IND_0+i+1], FP[:, IND_0-ARGS.base_per+1:IND_0+i+1], NX, CNTL, ARGS.reg, ARGS.Cons_test, ARGS.Formule_IC_TLS, ARGS.sample)
               
    return beta

##  multiprocessing over number of years in FP, OBS and CTL
BETA = np.empty((len(YEAR[IND_0:]), 4, np.array(NX).size))
BETA[:] = np.nan
if __name__ == '__main__':
    POOL = NewPool(4)
    TEMP = POOL.map(run_da, range(len(YEAR[IND_0:])))
    POOL.close()
    POOL.join()
    BETA = np.array(TEMP)

##  this applies specifically to AOA-exp, scen. dependent  
if  ARGS.background == 'stat':
    ## assuming additive forcing 
    BETA[:, :, 1] = BETA[:, :, 1] + BETA[:, :, 0]

##  confidence interval check; infinite set to +/-10
for r in range(np.array(NX).size):
    INFERIOR = np.ones(BETA.shape[0])*(BETA[:, 0, r] > BETA[:, 2, r])*-10.0
    INFERIOR = np.where(INFERIOR != 0, INFERIOR, BETA[:, 0, r])
    INFERIOR = np.where(INFERIOR < BETA[:, 1, r], INFERIOR, np.nan)
    SUPERIOR = np.ones(BETA.shape[0])*(BETA[:, 2, r] < INFERIOR)*10
    SUPERIOR = np.where(SUPERIOR != 0, SUPERIOR, BETA[:, 2, r])
    SUPERIOR = np.where(SUPERIOR < BETA[:, 1, r], np.nan, SUPERIOR)

    BETA[:, 1, r] = np.where(np.isnan(SUPERIOR), np.nan, BETA[:, 1, r])
    BETA[:, 0, r] = INFERIOR
    BETA[:, 2, r] = SUPERIOR

## time of detection
IND_DEC, TDECT = timedec(BETA, YEAR, IND_0)
IND_ATTR, TATTR = timeattr(BETA, YEAR, IND_DEC, IND_0)

## save and plot results
np.savez(OUTPATH+'/'+ARGS.reg+'_'+ARGS.Ctype+'_'+VARNAME+'_'+ARGS.background+'_s_'+str(ARGS.sigma)+'_b_'+str(ARGS.base_per)+'_mem_'+str(ARGS.member), base_per=ARGS.base_per, mem=ARGS.member, ind_dect=IND_DEC, tdect=TDECT, Ctype=ARGS.Ctype, beta=BETA, Year=YEAR, var=VARNAME, Cons_test=ARGS.Cons_test, sigma=ARGS.sigma, reg=ARGS.reg, bg=ARGS.background)

from plot_da_res import plot_summary
plot_summary(ARGS.var, YEAR, OBS, FP, CNTL, BETA, TDECT, TATTR, ARGS.sigma, ARGS.base_per, ARGS.background, ARGS.reg, ARGS.member, ARGS.Cons_test, ARGS.Ctype, INPATH, OUTPATH, IND_0)

print("DA ANALYSIS DONE")
