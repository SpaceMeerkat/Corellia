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

        self.lc1 = torch.nn.Linear(2048*2,256)
        self.lc2 = torch.nn.Linear(256,self.nodes)
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
    
    def encoder_linear(self,x):
        """ LINEARLY CONNECTED LAYERS FOR REGRESSION OF REQUIRED PARAMETERS """
        
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.hardtanh(x)
        return x
    
    def pos_ang(self, theta_1, theta_2):
        """ RECOVERING THE ANGLE AFTER A 2D ROTATION USING EULER MATRICES """

        pos = torch.atan2(theta_1,theta_2) 
        return pos
    
    def surface_brightness_profile(self, radius_tensor, scale_length, z, scale_length_z):
        """ GET THE SURFACE BRIGHTNESS FOR EACH PIXEL IN THE RADIUS ARRAY """
        
        sbProf = torch.exp((-radius_tensor/scale_length)+(-z/scale_length_z))
        return sbProf
    
    def velocity_profile(self, radius_tensor):
        """ GET THE LOS VELOCITIES FOR EACH PIXEL IN THE RADIUS ARRAY """

        vel = (2/pi)*(1.273239)*(torch.atan(pi*(radius_tensor)))
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
       
    def rotation_all(self,x,y,z,phi,theta):
        """ A function for rotating about the z and then x axes """
        _theta = theta + pi/2
        xmax = x.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        ymax = y.max(1)[0].max(1)[0].max(1)[0][:,None,None,None] 
        xx = ((x/xmax)*torch.cos(_theta) - (y/ymax)*torch.sin(_theta))*xmax
        _y = ((x/xmax)*torch.sin(_theta) + (y/ymax)*torch.cos(_theta))*ymax    
        _ymax = _y.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        zmax = z.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]    
        yy = ((_y/_ymax)*torch.cos(phi) - (z/zmax)*torch.sin(phi))*_ymax
        zz = ((_y/_ymax)*torch.sin(phi) + (z/zmax)*torch.cos(phi))*zmax
        rr = torch.sqrt((xx**2)+(yy**2)+(zz**2))
        rr_xy = torch.sqrt((xx**2)+(yy**2))
                
        return rr, xx, _y, rr_xy, torch.abs(zz)
    
    def cube_maker(self, x, original1, original2, shape):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """  
       
#        pos3 = -t[:,0].clone()[:,None,None,None]    ### Poition angle
#        inc3 = t[:,1].clone()[:,None,None,None]  ### Inclination
#        a = t[:,2].clone()[:,None,None,None]        ### SBprof scale length
#        a = a * shape / 2
#        ah = t[:,3].clone()[:,None,None,None] 
#        ah = ah * shape/2
#        Vh = t[:,4].clone()[:,None,None,None]       ### Maximum velocity allowed
#        ah_z = ah.clone()*0.2

        pos3 = self.pos_ang(x[:,0].clone(),x[:,1].clone())[:,None,None,None]    ### Poition angle
        inc3 = self.de_regularise(x[:,2].clone(),pi/18,pi/2)[:,None,None,None]  ### Inclination
        a = self.de_regularise(x[:,3].clone(),0.1,0.5)[:,None,None,None]        ### SBprof scale length
        a = a * shape / 2
        ah = self.de_regularise(x[:,4].clone(),0.1,0.5)[:,None,None,None]        ### DM halo scale length
        ah = ah * shape / 2
        Vh = self.de_regularise(x[:,5].clone(),100,500)[:,None,None,None]       ### Maximum velocity allowed
        a_z = a.clone()*0.2
        
        ah = ah/ah * 32
        Vh = Vh/Vh * 500
        inc3 = inc3/inc3 * 70/90 * pi/2
                                     
        # Create radius cube __________________________________________________

        rr_t, xx, yy, rr_xy, zz = self.rotation_all(self.xxx,self.yyy,self.zzz,inc3,pos3) 
        
        # Create sbProf cube __________________________________________________
               
        sbProf = self.surface_brightness_profile(rr_xy, a, zz, a_z)
                      
        # Create moment 0 map _________________________________________________
        
        mom0 = sbProf.sum(axis=3)
        mom0 = mom0.unsqueeze(1)
        mom0 = self.regularise2(mom0)
        
#        mom0 = torch.nn.functional.conv2d(mom0,self.weights,padding=4)
#        mom0 = torch.nn.functional.interpolate(mom0,size=64,mode='bilinear')
        
        # Create velocity cube ________________________________________________
        
        xmax = xx.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        ymax = yy.max(1)[0].max(1)[0].max(1)[0][:,None,None,None] 
        
        rr_t_v = rr_t.clone()
        rr_t_v = rr_t_v/ah
        
        # Convert to LOS velocities ___________________________________________
        
        vel = self.velocity_profile(rr_t_v)                                     
        thetas = torch.atan2(yy/ymax, xx/xmax) - pi/2
        thetas = torch.sin(thetas)
        thetas = thetas * torch.sin(inc3) 
        vel = vel * Vh * thetas
               
        # Create moment 1 map _________________________________________________
        
        sbProf_v = sbProf.clone()
        sbProf_v = sbProf_v 
        sbProf_v = sbProf_v + 1e-10
        mom0_v = sbProf_v.sum(axis=3).unsqueeze(1)
                
        vel = vel * sbProf_v 
        vel = vel.sum(axis=3)
        vel = vel.unsqueeze(1)
        vel = vel/mom0_v
        vel = vel / 500
        
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
        pos = self.pos_ang(output[:,0].clone(),output[:,1].clone())[:,None,None] ### Poition angle
        inc = self.de_regularise(output[:,2].clone(),pi/18.,pi/2.)[:,None,None]  ### Inclination
        a = self.de_regularise(output[:,3].clone(),0.1,0.5)[:,None,None]         ### SBprof scale length
        ah = self.de_regularise(output[:,4].clone(),0.1,0.5)[:,None,None]       ### DM halo scale length
        Vh = self.de_regularise(output[:,5].clone(),100,500)[:,None,None]        ### Maximum velocity allowed
#        ah_z = self.de_regularise(output[:,6].clone(),0.1,1)[:,None,None] 
        output = torch.tensor([pos,inc,a,ah,Vh])
        return output
        
    def forward(self, x1, x2):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        
        output1 = self.mom0_encoder(x1)
        output2 = self.mom1_encoder(x2)
        output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output)
        output = self.cube_maker(output,x1,x2,shape=64)
        return output
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
