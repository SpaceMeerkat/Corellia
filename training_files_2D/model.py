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
        self.conv0 = torch.nn.Conv2d(1,16,3,1,padding=1)
        self.conv1 = torch.nn.Conv2d(16,32,3,1,padding=1)
        self.conv2 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        self.pool = torch.nn.MaxPool2d(2)

        self.lc1 = torch.nn.Linear(2048,256)
        self.lc2 = torch.nn.Linear(256,128)
        self.lc3 = torch.nn.Linear(128,self.nodes) 
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
        pos = torch.atan2(theta_1,theta_2) + pi
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
        tensor=tensor+1.
        tensor=tensor*(maximum-minimum)/2.
        tensor=tensor+minimum
        return tensor
    
    def cube_maker(self, x, vlim, shape, dv, v_size):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """ 
        
        ### Define the variables needed for cube modelling from input x
        pos = self.pos_ang(x[:,0].clone(),x[:,1].clone()) ### Poition angle
        inc = self.de_regularise(x[:,2].clone(),0,pi/2.) ### Inclination
        a = self.de_regularise(x[:,3].clone(),0.1,0.4) ### SBprof scale length
        a = a * shape / 2
        
        ### Create 2D arrays of x,y, and r values
        xx_t = self.xx*torch.cos(pos) + self.yy*torch.sin(pos)
        yy_t = (-self.xx*torch.sin(pos) + self.yy*torch.cos(pos))*torch.cos(inc)
        rr_t = torch.sqrt((xx_t**2) + (yy_t**2))
        
        ### Create 2D array of SBprof given some 2D radius array
        sbProf = self.surface_brightness_profile(rr_t, a)
        sbProf = sbProf / torch.max(sbProf)
        sbProf = sbProf - torch.min(sbProf)
        sbProf = sbProf / torch.max(sbProf)
        sbProf = (sbProf*2) - 1
       
        return sbProf, rr_t
    
    def test_encode(self,x):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        output = self.encoder_conv(x)
        output = self.encoder_linear(output)
        return output
        
    def forward(self, x):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        output = self.encoder_conv(x)
        output = self.encoder_linear(output)
        output, rr_t = self.cube_maker(output,vlim=1,shape=64,dv=10,v_size=120)
        return output, rr_t
    
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