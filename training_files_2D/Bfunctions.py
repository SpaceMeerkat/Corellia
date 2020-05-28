#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:24 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#

from astropy.nddata.utils import Cutout2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from cube_generator import cube_generator
from sauron_colormap import sauron

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def learning_rate(initial_lr, epoch):   
    """Sets the learning rate to the initial LR decayed by a factor of 10 every
    N epochs"""
    
    lr = initial_lr * (0.975 ** (epoch // 10))
    
    return lr 

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def return_cube(i):
    """ MULTIPROCESS CREATION OF CUBES WITH VARIOUS POSITION ANGLES AND
        INCLINATIONS """
        
    # pos_ang = random.uniform(-180,180)
    inc_ang = random.uniform(10,90)
    a = random.uniform(0.1,0.35)
    ah = random.uniform(0.1,0.5)
    Vh = random.uniform(50,500)
    
    pos_ang = 0
    
    cube = cube_generator(a=a,pos_ang=pos_ang,
                          inc_ang=inc_ang,resolution=1000,ah=ah,
                          Vh=Vh).cube_creation()  
    
    # cube[cube<3*np.std(cube[:,:,:10])] = np.nan
    mom0 = np.nansum(cube,axis=2) 
    mom0 -= np.nanmin(mom0)
    mom0 /= np.nanmax(mom0)
    
    pos_ang = np.deg2rad(pos_ang)
    inc_ang = np.deg2rad(inc_ang)
        
    return (mom0),pos_ang,inc_ang,a

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    

def plotter(s, mom0, inc, out_dir):    
    """ PLOTTING THE PREDICTED AND TRUE VELOCITY FIELDS """
    
    s = s.detach().cpu().numpy()
    mom0 = mom0.detach().cpu().numpy()
    inc = inc[:,0,0].detach().cpu().numpy()
    inc = np.rad2deg(inc)
    
    for i in range(s.shape[0]):
        plt.figure()
        
        plt.subplot(122)
        b = s[i,0,:,:]
        plt.imshow(b,cmap='magma')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('PREDICTED: %.2f' %inc[i])
    
        plt.subplot(121)
        b = mom0[i,0,:,:]
        plt.imshow(b,cmap='magma')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('TRUE')
        
        plt.tight_layout()
        plt.savefig(out_dir+str(i)+'.png')
    
    return

#_____________________________________________________________________________#
#_____________________________________________________________________________#






