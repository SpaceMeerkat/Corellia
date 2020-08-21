#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:26:01 2020

@author: SpaceMeerkat

A helper script for loading in WISDOM data for non-blind testing. This way we
save the training script for the essentials regarding the training procedure
only. 

"""

# =============================================================================
# Import relevant packages
# =============================================================================

from math import pi as pi
import numpy as np
import torch
import sys
sys.path.append("../training_files_2D/")
from astropy.io import fits
import glob

# =============================================================================
# Define the WISDOM loader function
# =============================================================================

def WISDOM_loader(data_path):
   
# =============================================================================
# Collect all WISDOM filenames for testing all of them
# =============================================================================
    
    mom0_filenames = glob.glob(data_path + '*mom0.fits')
    mom0_filenames.sort()
    mom1_filenames = glob.glob(data_path + '*mom1.fits')
    mom1_filenames.sort()
    
    names = []
    for i in mom0_filenames:
        name = i.split('/')[-1]
        if name[0] == 'N':
            names.append(name[:7])
        else:
            names.append(name[:5])
    names = np.array(names)
    
# =============================================================================
# Collecting the mom0 and mom1 tensors for testing
# =============================================================================
    
    mom0s, mom1s, cdelts, sizes, vcirc = [], [], [], [], []
    
    for i in range(len(mom0_filenames)):
        f1 = fits.open(mom0_filenames[i])
        mom0s.append(f1[1].data.astype(float))
        cdelt = np.abs(f1[0].header['CDELT1']) * 60 * 60 
        cdelts.append(cdelt)
        size = cdelt * f1[0].header['NAXIS1'] / 2
        sizes.append(size)
        try:
            vcirc.append(f1[2].data)
        except:
            vcirc.append(np.zeros(32)*np.nan)
        f1.close()
        f2 = fits.open(mom1_filenames[i])
        mom1s.append(f2[1].data.astype(float))
        f2.close()
        
    mom0s, mom1s = np.array(mom0s), np.array(mom1s) 
    cdelts, sizes = np.array(cdelts), np.array(sizes)
    
    mom0s[mom0s<0.05] = 0
    mom1s[mom0s==0] = 0
    
    # mom0s[:,30:34,30:34]=0
    # mom1s[:,30:34,30:34]=0
    
    mom0s = torch.tensor(mom0s).to(torch.float).unsqueeze(1)
    mom1s = torch.tensor(mom1s).to(torch.float).unsqueeze(1)
    
    pos = torch.ones(mom0s.shape[0]).to(torch.float) * -pi/2
    
    return mom0s, mom1s, pos, cdelts, sizes, vcirc
    
# =============================================================================
# End of script
# =============================================================================
