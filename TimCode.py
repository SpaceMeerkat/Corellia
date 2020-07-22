#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:04:03 2020

@author: james
"""
from scipy import ndimage
import numpy as np
from kinms import KinMS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

    
# =============================================================================
# This sets up the parameters needed for KinMS, calls KinMS 
# and returns the cube
# =============================================================================


def return_cube(Vh,FluxThresh,Mask=False):
    """ CREATION OF CUBES WITH VARIOUS POSITION ANGLES AND
        INCLINATIONS """
    
    Vh = Vh
    a = 0.26
    av = 0.18
    pos_ang = -90
    inc_ang = 35
    
    cube = cube_generator(a=a,pos_ang=pos_ang,
                          inc_ang=inc_ang,resolution=1000,av=av,
                          Vh=Vh,FluxThresh=FluxThresh).cube_creation(smoothmask=Mask)  
           
    return cube

# =============================================================================
# This bit does the KinMS cube creation with optional smooth masking
# =============================================================================

class cube_generator:
    """ TAKES PHYSICAL ARGUMENTS AND CALLS KinMS.py TO CREATE A FAKE DATA CUBE 
        FOR TRAINING A NN MODEL ON """
        
    def __init__(self,a,pos_ang,inc_ang,resolution,av,Vh,FluxThresh=None):
            self.a = a
            self.scale_length = None
            self.pos_ang = pos_ang
            self.inc_ang = inc_ang
            self.resolution = resolution
            self.av = av
            self.Vh = Vh
            self.extent = 32
            self.xsize=64
            self.ysize=64
            self.vsize=1200
            self.cellSize=1
            self.dv=1
            self.beamSize=[2,2,0]
            self.FluxThresh = FluxThresh
            
    def surface_brightness_profile(self):
        """ CREATE A SURFACE BRIGHTNESS PROFILE THAT SCALES WITH GALACTIC 
            RADIUS """
        
        self.scale_length = self.extent * self.a
        radii = np.linspace(0,self.extent*10,self.resolution)
        sbProf = np.exp(-radii/self.scale_length)
        return radii,sbProf
    
    def velocity_profile(self,radii):
        """ CREATE A VELOCITY PROFILE THAT SCALES WITH GALACTIC RADIUS """
        
        vel = self.Vh * (2/np.pi) * np.arctan(radii[1:]/(self.av*self.extent)) # arctan function option
        vel = np.insert(vel,0,1e-10)
        return vel
    
    def smoothmask(self,cube):
        dummy_cube = cube.copy()
        mask=ndimage.uniform_filter(dummy_cube, size=[1.5*(self.beamSize[0]/self.cellSize),1.5*(self.beamSize[0]/self.cellSize),
                                                      4], mode='constant', cval=0.0)
        scale = cube.max() / mask.max()
        mask*=scale
        mask[mask < self.FluxThresh]=0
        mask[mask > 0] =1
        return mask
       
    def cube_creation(self,smoothmask=False):
        """ CREATE THE CUBE USING KinMS.py """
        
        sbRad,sbProf = self.surface_brightness_profile()
        vel = self.velocity_profile(sbRad)
        cube=KinMS(xs=self.xsize,ys=self.ysize,vs=self.vsize,cellSize=self.cellSize,dv=self.dv,
                   beamSize=self.beamSize,inc=self.inc_ang,sbProf=sbProf,sbRad=sbRad,velProf=vel,
                   velRad=sbRad,posAng=self.pos_ang,intFlux=30,verbose=False, 
                   cleanOut=False).model_cube()
        
        if smoothmask == True:
            mask = self.smoothmask(cube)
            cube *= mask 
            mask = None
            del mask
        
        return cube
        
    
# =============================================================================
# Making the cube and moment maps with smooth masking (optional)
# =============================================================================

Vh = 320 # Left this outside the function to play around with 
FluxThresh = 0.009 # The smooth masking brightness cutoff
cube = return_cube(Vh=Vh,FluxThresh=FluxThresh,Mask=False) #Create the cube with masking

mom0 = np.nansum(cube,axis=2) # Create the moment maps
mom1 = np.nansum(cube*np.arange(-600,600,1)[None,None,:],axis=2) / mom0
# mom1 *= (Vh/500)/np.nanmax(mom1) # Normalise the moment 1 map
mom0 -= np.nanmin(mom0)# Normalise the moment 0 map
mom0 /= np.nanmax(mom0)

plt.figure()
plt.subplot(121)
plt.imshow(mom0); plt.colorbar()
plt.subplot(122)
plt.imshow(mom1); plt.colorbar()
plt.savefig('testimage.png')