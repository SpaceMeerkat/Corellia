#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:15:02 2019

@author: SpaceMeerkat
"""

import torch
from math import pi as pi
#from functions import makebeam
  
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class CAE(torch.nn.Module):
    """ PYTORCH CONVOLUTIONAL AUTOENCODER WITH A FUNCTIONAL DECODER """
               
    def __init__(self, nodes, xxx=None, yyy=None, zzz=None, xx=None, yy=None):
        super().__init__()
        
        self.xx = xx
        self.yy = yy
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
        
#        self.weights = makebeam(xx.shape[1],yy.shape[1],40)
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

        pos = torch.atan2(theta_1,theta_2) + pi
        return pos
    
    def surface_brightness_profile(self, radius_tensor, scale_length):
        """ GET THE SURFACE BRIGHTNESS FOR EACH PIXEL IN THE RADIUS ARRAY """
        
        sbProf = torch.exp(-radius_tensor/scale_length)
        return sbProf
    
    def velocity_profile(self, radius_tensor, Vh, ah):
        """ GET THE LOS VELOCITIES FOR EACH PIXEL IN THE RADIUS ARRAY """

        vel = (2/pi)*(torch.atan(radius_tensor/ah))
        return vel
    
    def de_regularise(self,tensor,minimum,maximum):
        """ RECOVER THE PARAMETERS OUT OF THE NORMALISED RANGE [-1, 1] """
        
        tensor=tensor+1.
        tensor=tensor*(maximum-minimum)/2.
        tensor=tensor+minimum
        return tensor
    
    def regularise(self,array):
        """ NORMALISES THE INPUT ARRAYS INTO THE RANGE [-1,1] """
        
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        array = array - array.min(1)[0].min(1)[0].min(1)[0][:,None,None,None]
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        array = (array*2) - 1
        return array
    
    def regularise2(self,array):
        """ NORMALISES THE INPUT ARRAYS INTO THE RANGE [0,1] """
        
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        array = array - array.min(1)[0].min(1)[0].min(1)[0][:,None,None,None]
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        return array
       
    def rotation_all(self,x,y,z,phi,theta):
        """ A function for rotating about the x and then z axes """
        
        xmax = x.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        ymax = y.max(1)[0].max(1)[0].max(1)[0][:,None,None,None] 
        xx = ((x/xmax)*torch.cos(theta) - (y/ymax)*torch.sin(theta))*xmax
        _y = ((x/xmax)*torch.sin(theta) + (y/ymax)*torch.cos(theta))*ymax    
        ymax = _y.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        zmax = z.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]    
        yy = ((_y/ymax)*torch.cos(phi) - (z/zmax)*torch.sin(phi))*ymax
        zz = ((_y/ymax)*torch.sin(phi) + (z/zmax)*torch.cos(phi))*zmax
        rr = torch.sqrt((xx**2)+(yy**2)+(zz**2))
        return rr, yy, xx
    
    def cube_maker(self, x, original1, original2, shape):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """  

        pos = self.pos_ang(x[:,0].clone(),x[:,1].clone())[:,None,None]          ### Poition angle
        inc = self.de_regularise(x[:,2].clone(),pi/18.,pi/2.)[:,None,None]      ### Inclination
        a = self.de_regularise(x[:,3].clone(),0.1,0.5)[:,None,None,None]        ### SBprof scale length
        a = a * shape / 2
        ah = self.de_regularise(x[:,4].clone(),0.01,0.5)[:,None,None,None]      ### DM halo scale length
        ah = ah * shape / 2
        Vh = self.de_regularise(x[:,5].clone(),100,500)[:,None,None,None]       ### Maximum velocity allowed
        
        inc3 = inc.clone()[:,None]
        pos3 = pos.clone()[:,None]

        rr_t,yy,xx = self.rotation_all(self.xxx,self.yyy,self.zzz,inc3,pos3)   
               
        sbProf = self.surface_brightness_profile(rr_t, a)
        mom0 = sbProf.sum(axis=3)
        mom0 = mom0.unsqueeze(1)
                                      
        vel = self.velocity_profile(rr_t, Vh, ah) 
        vel = vel * torch.sin(torch.atan2(yy, xx)) * torch.sin(inc3)         ### Convert to LOS velocities
        vel = vel * sbProf

        vel = vel/vel.max()
        vel = vel*Vh*torch.sin(inc3)
        vel = vel.sum(axis=3)
        vel = vel.unsqueeze(1)
        vel = vel/mom0.clone()
        
        vel[vel == float('inf')]=0      
        vel[vel == -float('inf')]=0
        vel[vel!=vel] = 0
        
        mom0[mom0 == float('inf')]=0      
        mom0[mom0 == -float('inf')]=0
        mom0[mom0!=mom0] = 0
        
        vel[original2==0] = 0
        mom0[original1==0] = 0
        
#        vel = torch.nn.functional.conv2d(vel,self.weights,padding=4)
#        vel = torch.nn.functional.interpolate(vel,size=64,mode='bilinear')        
#        sbProf = torch.nn.functional.conv2d(sbProf,self.weights,padding=4)
#        sbProf = torch.nn.functional.interpolate(sbProf,size=64,mode='bilinear')
                                      
        return mom0, vel, inc3
    
    def test_encode(self,x1,x2):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        
        output1 = self.mom0_encoder(x1)
        output2 = self.mom1_encoder(x2)
        output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output)
        pos = self.pos_ang(output[:,0].clone(),
                           output[:,1].clone())[:,None,None]                     ### Poition angle
        inc = self.de_regularise(output[:,2].clone(),pi/18.,pi/2.)[:,None,None]  ### Inclination
        a = self.de_regularise(output[:,3].clone(),0.1,0.5)[:,None,None]         ### SBprof scale length
        ah = self.de_regularise(output[:,4].clone(),0.01,0.5)[:,None,None]       ### DM halo scale length
        Vh = self.de_regularise(output[:,5].clone(),100,500)[:,None,None]        ### Maximum velocity allowed
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
