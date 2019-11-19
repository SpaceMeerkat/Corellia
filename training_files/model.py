#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:15:02 2019

@author: SpaceMeerkat
"""

import torch
from math import pi as pi
   
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class CAE(torch.nn.Module):
    
    """ PYTORCH CONVOLUTIONAL AUTOENCODER WITH A FUNCTIONAL DECODER """
               
    def __init__(self, nodes, xx, yy, cube):
        super().__init__()
        
        self.xx = xx
        self.yy = yy
        self.cube = cube
        
        self.nodes = nodes
        self.conv0 = torch.nn.Conv2d(120,16,3,1,padding=1)
        self.conv1 = torch.nn.Conv2d(16,32,3,1,padding=1)
        self.conv2 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        self.pool = torch.nn.MaxPool2d(2)

        self.lc1 = torch.nn.Linear(2048,1024)
        self.lc2 = torch.nn.Linear(1024,256)
        self.lc3 = torch.nn.Linear(256,self.nodes) 
        self.relu = torch.nn.ReLU()
        self.hardtanh = torch.nn.Hardtanh(min_val=-1,max_val=1.)
        
    def encoder_conv(self,x):
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
        return x
    
    def encoder_linear(self,x):
        """ LINEARLY CONNECTED LAYERS FOR REGRESSION OF REQUIRED PARAMETERS """
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.hardtanh(x)
        return x
    
    def pos_ang(self, theta_1, theta_2):
        """ RECOVERING THE ANGLE AFTER A 2D ROTATION USING EULER MATRICES """
        pos = torch.atan2(theta_1,theta_2)
        return pos
    
    def surface_brightness_profile(self, radius_tensor, scale_length):
        """ GET THE SURFACE BRIGHTNESS FOR EACH PIXEL IN THE RADIUS ARRAY """
        sbProf = torch.exp(-radius_tensor/scale_length)
        return sbProf
    
    def velocity_profile(self, radius_tensor, Vh, ah):
        """ GET THE LOS VELOCITIES FOR EACH PIXEL IN THE RADIUS ARRAY """
        vel = torch.sqrt(torch.pow(Vh, 2)*(1-((ah/radius_tensor)*torch.atan(radius_tensor/ah))))
        return vel
    
    def de_regularise(self,tensor,minimum,maximum):
        """ RECOVER THE PARAMETERS OUT OF THE NORMALISED RANGE [-1, 1] """
        tensor+=1.
        tensor*=(maximum-minimum)/2.
        tensor+=minimum
        return tensor
    
    def cube_maker(self, x, vlim, shape, dv, v_size):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """ 
        
        ### Define the variables needed for cube modelling from input x
        pos = self.pos_ang(x[:,0].clone(),x[:,1].clone()) ### Poition angle
        inc = self.de_regularise(x[:,2].clone(),5,90) ### Inclination
        a = self.de_regularise(x[:,3].clone(),0.1,0.4) ### SBprof scale length
        ah = self.de_regularise(x[:,4].clone(),0.1,1) ### DM halo scale length
        Vh = self.de_regularise(x[:,5].clone(),50,500) ### Maximum velocity allowed
        
        ### Create 2D arrays of x,y, and r values
        xx_t = self.xx*torch.cos(pos) + self.yy*torch.sin(pos)
        yy_t = -self.xx*torch.sin(pos) + (self.yy*torch.cos(inc))*torch.cos(pos)
        rr_t = torch.sqrt(torch.pow(xx_t,2) + torch.pow(yy_t,2))
        
        ### Create 2D array of SBprof given some 2D radius array
        sbProf = self.surface_brightness_profile(rr_t, a)
        
        ### Define the 2D V(r) given some 2D radius array
        vel = self.velocity_profile(rr_t, Vh, ah)  
        vel *= -torch.cos(pi-torch.atan2(xx_t,yy_t)+pos)*torch.sin(inc) ### Convert to LOS velocities
        v = vel.clone() ### Velocity array is returned for later comparison to true velocity map
        vel = vel//dv ### Get channel indices of each pixel
        vel += int(v_size/2.) ### Redistribute channel values to lie in range 0-N
        vel = torch.clamp(vel,0,v_size) ### Restrict to existing channel range 
       
        cube = self.cube.clone()
        cube[torch.unique(vel).type(torch.LongTensor),:,:] = torch.stack([*map(vel.__eq__,torch.unique(vel))]).type(torch.float)*sbProf.type(torch.float) ### Fill cube
        cube = cube.unsqueeze(0) ### Resize cube
        
        cube = cube/torch.max(cube)

        return cube, v
        
    def forward(self, x):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        output = self.encoder_conv(x)
        output = self.encoder_linear(output)
        output, v = self.cube_maker(output,vlim=500,shape=64,dv=10,v_size=120)
        return output, v
    
#=============================================================================#
#///   MODEL BELOW IF WANTING TO DO A CNN REGRESSION TASK INSTEAD   //////////#
#=============================================================================#
        
class CovNet(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.conv1 = torch.nn.Conv2d(120,32,3,1,padding=1)
        self.conv2 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        self.pool = torch.nn.MaxPool2d(2)

        self.lc1 = torch.nn.Linear(128*8*8,2048)
        self.lc2 = torch.nn.Linear(2048,1024)
        self.lc3 = torch.nn.Linear(1024,512)
        self.lc4 = torch.nn.Linear(512,256)
        self.lc5 = torch.nn.Linear(256,self.nodes) 
        self.relu = torch.nn.ReLU()
        self.hardtanh = torch.nn.Hardtanh(min_val=-1,max_val=1.)
        
    def feature_extraction(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    
    def linear(self,x):
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.lc4(x)
        x = self.relu(x)
        x = self.lc5(x)
        x = self.hardtanh(x)
        return x
        
    def forward(self,x):
        output = self.feature_extraction(x)
        output = self.linear(output)
        return output
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#