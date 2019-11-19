#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:24 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#
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
    
    lr = initial_lr * (0.5 ** (epoch// 1))
    
    return lr 

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def return_cube(i):

    """ MULTIPROCESS CREATION OF CUBES WITH VARIOUS POSITION ANGLES AND
        INCLINATIONS """
        
    pos_ang = random.uniform(0,360)
    pos_ang =270
    inc_ang = random.uniform(0,90)
    inc_ang = 45
    scale_frac = random.uniform(0.1,0.4)
    ah = random.uniform(0.1,1)
    Vh = random.uniform(100,500)
    cube = cube_generator(scale_length_frac=scale_frac,pos_ang=pos_ang,
                          inc_ang=inc_ang,resolution=1000,ah=ah,Vh=Vh).cube_creation()
    
    cube[cube<=np.std(cube)]=np.nan
    cube = np.nansum(cube,axis=2)
#    cube = np.nansum(np.arange(-600,600,10)*cube,axis=2) / mom0
    cube -= np.nanmin(cube) 
    cube /= np.nanmax(cube)
    cube *= 2
    cube -= 1

    pos_ang = np.deg2rad(pos_ang)
        
    return cube,np.cos(pos_ang),np.sin(pos_ang),inc_ang,scale_frac,ah,Vh

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def pos_loss(output, target):
    
    """ LOSS FUNCTION FOR POSITION ANGLE BASED ON MINIMUM ARC SEPARATION """
        
    loss = torch.mean(torch.stack([ (output-target)**2,
                                   (1-torch.abs(output-target))**2] ).min(dim=0)[0]) 
    
    return loss

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def recover_pos(theta1,theta2):
    
    """ RECOVERING THE ANGLE AFTER A 2D ROTATION USING EULER MATRICES """
    
    angle = np.rad2deg(np.arctan2(theta1,theta2)) + 180
    
    return angle

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    

def plotter(rr_t, batch, batch2, out_dir):
    
    """ PLOTTING THE PREDICTED AND TRUE VELOCITY FIELDS """
       
    batch2 = batch2.detach().cpu().numpy()
    b = batch2
    plt.figure()
    plt.imshow(b,cmap=sauron)
    plt.colorbar()
    plt.savefig(out_dir+'predicted2.png')
    
    rr_t = rr_t.detach().cpu().numpy()
    b = rr_t
    plt.figure()
    plt.imshow(b,cmap=sauron)
    plt.colorbar()
    plt.savefig(out_dir+'rr_t.png')
    
    batch = batch.detach().cpu().numpy()
    b = batch[0,0,:,:]
    plt.figure()
    plt.imshow(b,cmap=sauron)
    plt.colorbar()
    plt.savefig(out_dir+'true.png')
    
    return











