#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:24 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from sauron_colormap import sauron
from cube_generator import cube_generator

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
    inc_ang = random.uniform(5,90)
    scale_frac = random.uniform(0.1,0.5)
    ah = random.uniform(0.01,0.1)
    Vh = random.uniform(100,500)
    sigma = random.uniform(0,20)
    moments = cube_generator(scale_length_frac=scale_frac,pos_ang=pos_ang,
                          inc_ang=inc_ang,resolution=1000,ah=ah,Vh=Vh,
                          sigma=sigma).cube_creation() 
    pos_ang = np.deg2rad(pos_ang)
           
    return moments,np.cos(pos_ang),np.sin(pos_ang),inc_ang,scale_frac,ah,Vh,sigma

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
    

def plotter(target, prediction, out_dir):
    
    """ PLOTTING THE PREDICTED AND TRUE VELOCITY FIELDS """
    
    target = target.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    a = target[0,:,:,:]
    b = prediction[0,:,:,:]
    plt.figure()
    plt.scatter(a,b)
    plt.tight_layout()
    plt.savefig(out_dir+'2D_stack/predicted.png')
    plt.close('all')
        
    return

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def regularise(array,maxims,minims):
        array = array / maxims
        array = array - minims
        array = array / maxims
        array = (array*2) - 1
        return array

#_____________________________________________________________________________#
#_____________________________________________________________________________#




