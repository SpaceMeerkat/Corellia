#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:11:50 2020

@author: james
"""

import torch
from math import pi as pi
import matplotlib.pyplot as plt
plt.close('all')
from sauron_colormap import sauron

#_____________________________________________________________________________#

def rotation_all(x,y,z,X_ANGLE,Z_ANGLE):
    """ SCALE_LENGTH function for rotating 3d particles about the z and then x axes """
    
    _theta = Z_ANGLE + pi/2 # Correct Z_ANGLE ready for rotations
    
    xmax = x.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
    ymax = y.max(1)[0].max(1)[0].max(1)[0][:,None,None,None] 
    
    xx = ((x/xmax)*torch.cos(_theta) - (y/ymax)*torch.sin(_theta))*xmax # Z-Rotation of x coords
    _y = ((x/xmax)*torch.sin(_theta) + (y/ymax)*torch.cos(_theta))*ymax # Z-Rotation of y coords  
    
    _ymax = _y.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
    zmax = z.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]    
    
    yy = ((_y/_ymax)*torch.cos(X_ANGLE) - (z/zmax)*torch.sin(X_ANGLE))*_ymax # X-Rotation of y coords
    zz = ((_y/_ymax)*torch.sin(X_ANGLE) + (z/zmax)*torch.cos(X_ANGLE))*zmax # X-Rotation of z coords
    
    RADIUS_ARRAY = torch.sqrt((xx**2)+(yy**2)+(zz**2)) # Get the cube of radii R(x,y,z)
    XY_RADIUS_ARRAY = torch.sqrt((xx**2)+(yy**2)) # Get the cube of xy radii only R(x,y)
            
    return RADIUS_ARRAY, xx, _y, XY_RADIUS_ARRAY, torch.abs(zz)

def create_brightness_values(radius_tensor, scale_length, z_radius, scale_length_z):
    """ GET THE SURFACE BRIGHTNESS FOR EACH PIXEL IN THE RADIUS ARRAY """
    
    sbProf = torch.exp((-radius_tensor/scale_length)+(-torch.abs(z_radius)/scale_length_z))
    return sbProf
    
def create_velocity_values(radius_tensor,V_sin_i,a_v):
    """ GET THE LINE OF SIGHT VELOCITYOCITIES FOR EACH VOXEL IN THE RADIUS CUBE """

    vel = V_sin_i * (1-torch.exp(-torch.abs(radius_tensor)/a_v))
    return vel

def regularise(array):
    """ NORMALISES THE INPUT ARRAYS INTO THE RANGE [0,1] """
    
    array = array - array.min(1)[0].min(1)[0].min(1)[0][:,None,None,None]
    array = array + torch.tensor(1e-10)
    array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
    return array

#_____________________________________________________________________________#
    
batch_size = 1

### Create the auxiliary arrays
l = torch.arange(0 - 319/2., (319/2.)+1)
yyy, xxx, zzz = torch.meshgrid(l,l,l)

xxx, yyy, zzz = xxx.repeat(batch_size,1,1,1), yyy.repeat(batch_size,1,1,1), zzz.repeat(batch_size,1,1,1)
xxx = xxx.to(torch.float)
yyy = yyy.to(torch.float)
zzz = zzz.to(torch.float)

inc_ang = torch.tensor([pi/3]).to(torch.float)[:,None,None,None] 
a = torch.tensor([0.75]).to(torch.float)[:,None,None,None] * 160 / 2
ah = torch.tensor([0.75]).to(torch.float)[:,None,None,None] *160 / 2
Vh = torch.tensor([1]).to(torch.float)[:,None,None,None] 
pos = torch.tensor([pi/2]).to(torch.float)[:,None,None,None] 
a_z = a * 0 + 1

rr_t, xx, yy, XY_RADIUS_ARRAY, zz = rotation_all(xxx,yyy,zzz,inc_ang,pos) 

sbProf = create_brightness_values(XY_RADIUS_ARRAY, a, zz, a_z)
# sbProf = regularise(sbProf)

BRIGHTNESS = sbProf.sum(axis=3)
BRIGHTNESS = BRIGHTNESS.unsqueeze(1)

xmax = xx.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
ymax = yy.max(1)[0].max(1)[0].max(1)[0][:,None,None,None] 

# rr_t_v = rr_t.detach()
    
# Convert to line-of-sight VELOCITY ___________________________________

VELOCITY = create_velocity_values(XY_RADIUS_ARRAY,Vh,ah)                                     
thetas = torch.atan2(yy/ymax, xx/xmax) - pi/2
thetas = torch.sin(thetas)
VELOCITY = VELOCITY * thetas
# VELOCITY[zz>1] = 0
       
# Create moment 1 map _________________________________________________

BRIGHTNESS_COPY = sbProf.detach()
BRIGHTNESS_COPY = BRIGHTNESS_COPY 
BRIGHTNESS_COPY_FLAT = BRIGHTNESS_COPY.sum(axis=3).unsqueeze(1) #+ 1e-10
                
VELOCITY = VELOCITY * BRIGHTNESS_COPY
VELOCITY = VELOCITY.sum(axis=3)
VELOCITY = VELOCITY.unsqueeze(1)
VELOCITY = VELOCITY / BRIGHTNESS_COPY_FLAT
VELOCITY[BRIGHTNESS<0.1]=0
      
#_____________________________________________________________________________#

plt.figure()
plt.subplot(121)
plt.imshow(BRIGHTNESS[0,0,:,:])
plt.subplot(122)
plt.imshow(VELOCITY[0,0,:,:],cmap=sauron)







