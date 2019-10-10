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

#=============================================================================#
#===   START OF CLASS   ======================================================#
#=============================================================================#

class cube_generator:
        
        def __init__(self,scale_length_frac,pos_ang,inc_ang,resolution):
                self.scale_length_frac = scale_length_frac
                self.scale_length = None
                self.pos_ang = pos_ang
                self.inc_ang = inc_ang
                self.resolution = resolution
                self.extent = 64
                self.xsize=128
                self.ysize=128
                self.vsize=128
                self.cellsize=1.0
                self.dv=10
                self.beamsize=[4.,4.,0.]
                
        def surface_brightness_profile(self):
                self.scale_length = self.extent / self.scale_length_frac
                radii = np.linspace(0,self.extent,self.resolution)
                sbProf = np.exp(-radii/self.scale_length)
                return radii,sbProf
        
        def velocity_profile(self):
                vel = np.ones(self.resolution)
                return vel
        
        def cube_creation(self):
                sbRad,sbProf = self.surface_brightness_profile()
                vel = self.velocity_profile()
                cube=KinMS().model_cube(xs=self.xsize,ys=self.ysize,vs=self.vsize,cellSize=self.cellsize,dv=self.dv,beamSize=self.beamsize,
                       inc=self.inc_ang,sbProf=sbProf,sbRad=sbRad,velProf=vel,velRad=sbRad,posAng=self.pos_ang,verbose=False)
                return cube

#=============================================================================#
#===   END OF CLASS   ========================================================#
#=============================================================================#        

f = cube_generator(scale_length_frac=4,pos_ang=45,inc_ang=70,resolution=1000).cube_creation()

#=============================================================================#
#===   END OF SCRIPT   =======================================================#
#=============================================================================#
