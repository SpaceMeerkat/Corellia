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
               
    def __init__(self, nodes, xx=None, yy=None, cube=None):
        super().__init__()
        
        self.xx = xx
        self.yy = yy
        self.cube = cube
        
        self.nodes = nodes
        self.conv4 = torch.nn.Conv2d(1,16,3,1,padding=1)
        self.conv5 = torch.nn.Conv2d(16,32,3,1,padding=1)
        self.conv6 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv7 = torch.nn.Conv2d(64,128,3,padding=1)
        self.pool = torch.nn.MaxPool2d(2)

        self.lc1 = torch.nn.Linear(2048,256)
        self.lc2 = torch.nn.Linear(256,self.nodes)
        self.relu = torch.nn.ReLU()
        self.hardtanh = torch.nn.Hardtanh(min_val=-1,max_val=1.)
           
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
        
        #vel = torch.sqrt((Vh**2)*(1-((ah/radius_tensor)*torch.atan(radius_tensor/ah))))
#        vel = ((2*Vh)/pi)*(torch.atan(radius_tensor/ah))
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
    
    def cube_maker(self, x, original, vlim, shape, dv, v_size):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """  

        pos = self.pos_ang(x[:,0].clone(),x[:,1].clone())[:,None,None]          ### Poition angle
        inc = self.de_regularise(x[:,2].clone(),pi/18.,pi/2.)[:,None,None]      ### Inclination
        a = self.de_regularise(x[:,3].clone(),0.1,0.5)[:,None,None]             ### SBprof scale length
        a = a * shape / 2
        ah = self.de_regularise(x[:,4].clone(),0.01,0.5)[:,None,None]           ### DM halo scale length
        ah = ah * shape / 2
        Vh = self.de_regularise(x[:,5].clone(),100,500)[:,None,None]            ### Maximum velocity allowed
                
        xx_t = -self.xx*torch.sin(pos) + self.yy*torch.cos(pos)
        yy_t = self.xx*torch.cos(pos) + self.yy*torch.sin(pos)
        yy_t = yy_t / torch.sin((pi/2)-inc)
        rr_t = torch.sqrt((xx_t**2) + (yy_t**2))
        
        vel = self.velocity_profile(rr_t, Vh, ah) 
        vel = vel * torch.cos(torch.atan2(yy_t, xx_t)) * torch.sin(inc)         ### Convert to LOS velocities
        vel = vel/vel.max()
        vel = vel*Vh*torch.sin(inc)
        
#        vel[vel<-600] = 0
#        vel[vel>600] = 0        
        
        vel = vel.unsqueeze(1)
        vel = vel/500
        vel[original==0] = 0
                                      
        return vel
    
    def test_encode(self,x):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        
        output = self.mom1_encoder(x)
        output = self.encoder_linear(output)
        pos = self.pos_ang(output[:,0].clone(),
                           output[:,1].clone())[:,None,None]                     ### Poition angle
        inc = self.de_regularise(output[:,2].clone(),pi/18.,pi/2.)[:,None,None]  ### Inclination
        a = self.de_regularise(output[:,3].clone(),0.1,0.5)[:,None,None]         ### SBprof scale length
        ah = self.de_regularise(output[:,4].clone(),0.01,0.5)[:,None,None]       ### DM halo scale length
        Vh = self.de_regularise(output[:,5].clone(),50,500)[:,None,None]         ### Maximum velocity allowed
        output = torch.tensor([pos,inc,a,ah,Vh])
        return output
        
    def forward(self, x):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        
        output = self.mom1_encoder(x)
        output = self.encoder_linear(output)
        output = self.cube_maker(output,x,vlim=1,shape=64,dv=10,v_size=120)
        return output
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
