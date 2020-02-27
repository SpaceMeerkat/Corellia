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

import matplotlib
matplotlib.use('Agg')

#=============================================================================#
#~~~   START OF CLASS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#=============================================================================#

class cube_generator:
    """ TAKES PHYSICAL ARGUMENTS AND CALLS KinMS.py TO CREATE A FAKE DATA CUBE 
        FOR TRAINING A NN MODEL ON """
        
    def __init__(self,scale_length_frac,pos_ang,inc_ang,resolution,ah,Vh):
            self.scale_length_frac = scale_length_frac
            self.scale_length = None
            self.pos_ang = pos_ang
            self.inc_ang = inc_ang
            self.resolution = resolution
            self.ah = ah
            self.Vh = Vh
            self.extent = 32
            self.xsize=64
            self.ysize=64
            self.vsize=1200
            self.cellsize=1
            self.dv=10
            self.beamsize=[4,4,0]
            
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
        vel = ((2*self.Vh)/np.pi)*(np.arctan(radii[1:]/self.ah))
        vel = np.insert(vel,0,1)
        return vel
       
    def cube_creation(self):
        """ CREATE THE CUBE USING KinMS.py """
        
        sbRad,sbProf = self.surface_brightness_profile()
        vel = self.velocity_profile(sbRad)
        cube=KinMS(xs=self.xsize,ys=self.ysize,vs=self.vsize,cellSize=self.cellsize,dv=self.dv,
                   beamSize=self.beamsize,inc=self.inc_ang,sbProf=sbProf,sbRad=sbRad,velProf=vel,
                   velRad=sbRad,posAng=self.pos_ang,intFlux=30,verbose=False).model_cube()
        return cube

#=============================================================================#
#~~~   END OF CLASS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#=============================================================================#        

#=============================================================================#
#===   END OF SCRIPT   =======================================================#
#=============================================================================#
