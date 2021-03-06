#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:15:02 2019

@author: SpaceMeerkat
"""

import torch
from math import pi as pi
#from functions import makebeam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from kornia.geometry.transform import rotate
  
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class CAE(torch.nn.Module):
    """ PYTORCH CONVOLUTIONAL AUTOENCODER WITH A FUNCTIONAL DECODER """
               
    def __init__(self, nodes, xxx=None, yyy=None, zzz=None):
        super().__init__()
    
        self.xxx = xxx
        self.yyy = yyy
        self.zzz = zzz
        
        self.nodes = nodes
        self.conv0 = torch.nn.Conv2d(1,16,3,1,padding=1)
        self.conv1 = torch.nn.Conv2d(16,32,3,1,padding=1)
        self.conv2 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        self.conv4 = torch.nn.Conv2d(1,16,3,1,padding=1)
        self.conv5 = torch.nn.Conv2d(16,32,3,1,padding=1)
        self.conv6 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv7 = torch.nn.Conv2d(64,128,3,padding=1)
        self.pool = torch.nn.MaxPool2d(2)

        self.lc1 = torch.nn.Linear(2048,256)
        self.lc2 = torch.nn.Linear(256,2)
        self.lc3 = torch.nn.linear(2048,256)
        self.lc4 = torch.nn.linear(256,2)
        self.relu = torch.nn.ReLU()
        self.hardtanh = torch.nn.Hardtanh(min_val=-1,max_val=1.)
        
#        self.weights = makebeam(xxx.shape[1],yyy.shape[1],4)
#        self.weights = self.weights.unsqueeze(0).unsqueeze(0)
#        self.weights = torch.nn.Parameter(self.weights,requires_grad=False)
       
    def mom0_encoder(self,x):
        """ FEATURE EXTRACTION THROUGH CONVOLUTIONAL LAYERS """
        
        x = self.conv0(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(int(x.size()[0]),-1)
        return x
    
    def mom1_encoder(self,x):
        """ FEATURE EXTRACTION THROUGH CONVOLUTIONAL LAYERS """
        
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)        
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(int(x.size()[0]),-1)
        return x
    
    def encoder_linear(self,x,y):
        """ LINEARLY CONNECTED LAYERS FOR REGRESSION OF REQUIRED PARAMETERS """
        
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.hardtanh(x)

        y = self.lc3(y)
        y = self.relu(y)
        y = self.lc4(y)
        y = self.hardtanh(y)

        z = torch.cat((x,y),-1)
        return z
    
    def pos_ang(self, theta_1, theta_2):
        """ RECOVERING THE ANGLE AFTER A 2D ROTATION USING EULER MATRICES """

        pos = torch.atan2(theta_1,theta_2) 
        return pos
    
    def surface_brightness_profile(self, radius_tensor, scale_length, z, scale_length_z):
        """ GET THE SURFACE BRIGHTNESS FOR EACH PIXEL IN THE RADIUS ARRAY """
        
        sbProf = torch.exp((-radius_tensor/scale_length)+(torch.abs(-z)/scale_length_z))
        return sbProf
    
    def velocity_profile(self, V_sin_i, radius_tensor):
        """ GET THE LOS VELOCITIES FOR EACH PIXEL IN THE RADIUS ARRAY """

        vel = V_sin_i * (2/pi) * torch.atan(radius_tensor)
        return vel
    
    def de_regularise(self,tensor,minimum,maximum):
        """ RECOVER THE PARAMETERS OUT OF THE NORMALISED RANGE [-1, 1] """
        
        tensor=tensor+1.
        tensor=tensor*(maximum-minimum)/2.
        tensor=tensor+minimum
        return tensor
        
    def regularise2(self,array):
        """ NORMALISES THE INPUT ARRAYS INTO THE RANGE [0,1] """
        
        array = array - array.min(1)[0].min(1)[0].min(1)[0][:,None,None,None]
        array = array + torch.tensor(1e-10)
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        return array
       
    def rotation_all(self,x,y,z,X_ANGLE):
        """ A function for rotating about the z and then x axes """
        xmax = x.max(1)[0].max(1)[0].max(1)[0]
        zmax = z.max(1)[0].max(1)[0].max(1)[0]
        
        xx = ((x/xmax)*torch.cos(X_ANGLE) + (z/zmax)*torch.sin(X_ANGLE))*xmax # Z-Rotation of x coords
        zz = (-(x/xmax)*torch.sin(X_ANGLE) + (z/zmax)*torch.cos(X_ANGLE))*zmax # X-Rotation of z coords
        
        RADIUS_ARRAY = torch.sqrt(xx**2 + y**2 + zz**2) # Get the cube of radii R(x,y,z)
        XY_RADIUS_ARRAY = torch.sqrt(xx**2 + y**2) # Get the cube of xy radii only R(x,y)
                
        return RADIUS_ARRAY, xx, y, XY_RADIUS_ARRAY, torch.abs(zz)
    
    def cube_maker(self, x, original1, original2, shape):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """  

        # pos3 = self.pos_ang(x[:,2].clone(),x[:,3].clone())[:,None,None,None]    ### Poition angle
        inc3 = self.de_regularise(x[:,0].clone(),pi/18,pi/3)[:,None,None,None]  ### Inclination
        a = self.de_regularise(x[:,1].clone(),0.1,0.5)[:,None,None,None]        ### SBprof scale length
        a = a * shape / 2
        ah = self.de_regularise(x[:,2].clone(),0.1,0.5)[:,None,None,None]        ### DM halo scale length
        ah = ah * shape / 2
        Vh = self.de_regularise(x[:,3].clone(),0.2,1)[:,None,None,None]       ### Maximum velocity allowed
        a_z = a.clone()*0.2
        
        #a = a/a * 16
        #Vh = Vh/Vh * 500
        #pos3 = pos3/pos3 * -pi/2
#        inc3 = inc3/inc3 * 70/90 * pi/2
        #ah = ah/ah * 8
                                     
        # Create radius cube __________________________________________________

        rr_t, xx, yy, rr_xy, zz = self.rotation_all(self.xxx,self.yyy,self.zzz,inc3) 
        
        # Create sbProf cube __________________________________________________
               
        sbProf = self.surface_brightness_profile(rr_xy, a, zz, a_z)
                      
        # Create moment 0 map _________________________________________________
        
        mom0 = sbProf.sum(axis=3)
        mom0 = mom0.unsqueeze(1)
        mom0 = self.regularise2(mom0)
        
#        mom0 = torch.nn.functional.conv2d(mom0,self.weights,padding=4)
#        mom0 = torch.nn.functional.interpolate(mom0,size=64,mode='bilinear')
        
        # Create velocity cube ________________________________________________
               
        rr_t_v = rr_t.clone()
        rr_t_v = rr_t_v / ah
            
        # Convert to LOS velocities ___________________________________________
        
        vel = self.velocity_profile(Vh, rr_t_v)                                     
        thetas = torch.atan2(yy, xx) 
        vel = vel * torch.sin(thetas)
               
        # Create moment 1 map _________________________________________________
        
        sbProf_v = sbProf.clone()
        sbProf_v = sbProf_v 
        sbProf_v = sbProf_v + 1e-10
        mom0_v = sbProf_v.sum(axis=3).unsqueeze(1)

        vel = vel * sbProf_v 
        vel = vel.sum(axis=3)
        vel = vel.unsqueeze(1)
        vel = vel/mom0_v
        
#        vel = torch.nn.functional.conv2d(vel,self.weights,padding=4)
#        vel = torch.nn.functional.interpolate(vel,size=64,mode='bilinear')
        
        # Mask moment maps ____________________________________________________
                
        vel[original2==0] = 0
        mom0[original1==0] = 0
        
        vmax = vel[-2,0,:,:].max()
        
        return mom0, vel, inc3, pos3, (a, pos3, inc3, ah, Vh), vmax
    
    def test_encode(self,x1,x2):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        
        output1 = self.mom0_encoder(x1)
        output2 = self.mom1_encoder(x2)
        output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output)
        pos = self.pos_ang(output[:,2].clone(),output[:,3].clone())[:,None,None] ### Poition angle
        inc = self.de_regularise(output[:,0].clone(),pi/18,pi/3)[:,None,None]  ### Inclination
        a = self.de_regularise(output[:,1].clone(),0.1,0.5)[:,None,None]         ### SBprof scale length
        ah = self.de_regularise(output[:,5].clone(),0.1,0.5)[:,None,None]       ### DM halo scale length 
        Vh = self.de_regularise(output[:,4].clone(),100,500)[:,None,None]        ### Maximum velocity allowed 
        output = torch.tensor([pos,inc,a,ah,Vh])
        return output
        
    def forward(self, x1, x2):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        
        output1 = self.mom0_encoder(x1)
        output2 = self.mom1_encoder(x2)
        #output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output1,output2)
        output = self.cube_maker(output,x1,x2,shape=64)
        return output
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
