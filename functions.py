#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:24 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#

from cube_generator import cube_generator
import numpy as np

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
        
    pos_ang = np.random.uniform(0,360,1)
    inc_ang = np.random.uniform(0,90,1)
    scale_frac = np.random.uniform(0.1,0.4,1)
    cube = cube_generator(scale_length_frac=scale_frac,pos_ang=pos_ang[0],
                          inc_ang=inc_ang[0],resolution=1000).cube_creation()
    return cube,(pos_ang,inc_ang,scale_frac)

#_____________________________________________________________________________#
#_____________________________________________________________________________#