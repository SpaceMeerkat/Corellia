#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:20:55 2019

@author: SpaceMeerkat
"""

#=============================================================================#
#===   IMPORT RELEVANT MODULES   =============================================#
#=============================================================================#

import numpy as np
from KinMS import KinMS

#import matplotlib
#matplotlib.use('Agg')

#=============================================================================#
#~~~   START OF CLASS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#=============================================================================#

class cube_generator:
    """ TAKES PHYSICAL ARGUMENTS AND CALLS KinMS.py TO CREATE A FAKE DATA CUBE 
        FOR TRAINING A NN MODEL ON """
        
    def __init__(self,scale_length_frac,pos_ang,inc_ang,resolution,ah,Vh,sigma):
            self.scale_length_frac = scale_length_frac
            self.scale_length = None
            self.pos_ang = pos_ang
            self.inc_ang = inc_ang
            self.resolution = resolution
            self.ah = ah
            self.Vh = Vh
            self.gasSigma = sigma
            self.extent = 32
            self.xsize=64
            self.ysize=64
            self.vsize=1200
            self.cellsize=1.0
            self.dv=10
            self.beamsize=[4.,4.,0.]
            
    def surface_brightness_profile(self):
        """ CREATE A SURFACE BRIGHTNESS PROFILE THAT SCALES WITH GALACTIC 
            RADIUS """
        
        self.scale_length = self.extent * self.scale_length_frac
        radii = np.linspace(0,self.extent,self.resolution)
        sbProf = np.exp(-radii/self.scale_length)
        return radii,sbProf
    
    def velocity_profile(self,radii):
        """ CREATE A VELOCITY PROFILE THAT SCALES WITH GALACTIC RADIUS """
       
        self.ah = self.extent * self.ah
        vel = np.sqrt((self.Vh**2)*(1-((self.ah/radii[1:])*np.arctan(radii[1:]/self.ah))))
        vel = np.insert(vel,0,1)
        return vel
    
    def regularise(self,array):
        array = array / array.max((0,1))
        array = array - array.min((0,1))
        array = array / array.max((0,1))
        array = (array*2) - 1
        return array
    
    def create_moments(self,cube):
        mom0 = np.sum(cube,axis=2) + 1e-10
        channels = np.arange(-600,600,10)
        num = np.sum(cube*channels[None,None,:],axis=2)
        mom1 = num/mom0
        vel_narray = np.ones((cube.shape))
        vel_narray *= channels[None,None,:]
        mom2 = np.sqrt(np.sum(abs(cube) * (vel_narray - mom1[:,:,None])**2., axis=2) / (np.sum(abs(cube), axis=2)+1e-10))
        mom2 /= mom2.max()
        moments = np.dstack([mom0,mom1,mom2])
        moments = self.regularise(moments)
        return moments
    
    def cube_creation(self):
        """ CREATE THE CUBE USING KinMS.py """
        
        sbRad,sbProf = self.surface_brightness_profile()
        vel = self.velocity_profile(sbRad)
        f = KinMS()
        cube=f(xs=self.xsize,ys=self.ysize,vs=self.vsize,cellSize=self.cellsize,dv=self.dv,beamSize=self.beamsize,
               inc=self.inc_ang,sbProf=sbProf,sbRad=sbRad,velProf=vel,velRad=sbRad,posAng=self.pos_ang,
               gasSigma=self.gasSigma,verbose=False)
        moments = self.create_moments(cube)
        return moments

#=============================================================================#
#~~~   END OF CLASS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#=============================================================================#        

#=============================================================================#
#===   END OF SCRIPT   =======================================================#
#=============================================================================#
