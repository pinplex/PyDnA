# PyDnA
Detection and Attribution framework in python using the Optimal Fingerprinting Approach (Hasselmann, 1993; Ribes et al. 2013)

## Disclamer
This package is still under development and has not been fully tested yet.

## Overview
### Core framework
- ´PyDnA.py´               collection of functions needed by ROF_main.py (based on A. Ribes scilab code)

- ´ROF_main.py´            DA routine (based on A. Ribes scilab code)

### Helper functions (written by friederike.froeb@mpimet.mpg.de)
- ´load_fil_data.py´       load data and filter data

- ´plot_da_res.py´         plot results of da routine

- ´run_da_routine.py´      wrapper function, calls all other routines

## Dependencies
### Core framework
numpy (tested for version 1.17.4)
scipy (tested for version 1.3.1)

### Helper functions
argparse
multiprocessing
subprocess
xarray
pandas
matplotlib

## Run the example
1. [Download]() the example** data archive and unzip into the ´data´ directory
2. ´python run_da_routine.py ph -s 5 -b 15´
3. ´python run_da_routine.py --help´ to display all options.

   
**example data shows hydrogen ion concentration (measure of pH) for an ocean alkalinization experiment (Gonzalez & Ilyina, 2016)
