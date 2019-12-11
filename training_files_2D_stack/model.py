#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:15:02 2019

@author: SpaceMeerkat
"""

import torch
#import kornia
from math import pi as pi
   
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class CAE(torch.nn.Module):
    
    """ PYTORCH CONVOLUTIONAL AUTOENCODER WITH A FUNCTIONAL DECODER """
               
    def __init__(self, nodes, xx, yy, cube, dv):
        super().__init__()
        
        self.xx = xx
        self.yy = yy
        self.cube = cube
        self.width = cube.shape[1]
        self.shape = cube.shape[2]
        self.dv = dv
        self.vlim = (cube.shape[1]*dv)/2        

        self.nodes = nodes
        self.conv0 = torch.nn.Conv2d(1,16,3,1,padding=1)
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
        pos = torch.atan2(theta_1,theta_2) + pi
        return pos
    
    def de_regularise(self,tensor,minimum,maximum):
        """ RECOVER THE PARAMETERS OUT OF THE NORMALISED RANGE [-1, 1] """
        tensor=tensor+1.
        tensor=tensor*(maximum-minimum)/2.
        tensor=tensor+minimum
        return tensor
    
    def cube_maker(self, x, shape):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """ 
        
        ### Define the variables needed for cube modelling from input x
        pos_a = x[:,0].clone()[:,None,None] ### Poition angle
        pos_b = x[:,1].clone()[:,None,None] ### Poition angle
        inc = self.de_regularise(x[:,2].clone(),5,pi/2.)[:,None,None] ### Inclination
        a = self.de_regularise(x[:,3].clone(),0.1,0.5)[:,None,None] ### SBprof scale length
        a = a * shape / 2
        ah = self.de_regularise(x[:,4].clone(),0.01,0.1)[:,None,None] ### DM halo scale length
        ah = ah * shape / 2
        Vh = self.de_regularise(x[:,5].clone(),50,500)[:,None,None] ### Maximum velocity allowed
        sigma = self.de_regularise(x[:,5].clone(),0,20)[:,None,None]
        params = torch.stack([pos_a,pos_b,inc,a,ah,Vh,sigma]).T

        return params
    
    def test_encode(self,x):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        output = self.encoder_conv(x)
        output = self.encoder_linear(output)
        output = self.cube_maker(output,self.shape)
        
        return output
        
    def forward(self, x):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        output = self.encoder_conv(x)
        output = self.encoder_linear(output)
        output = self.cube_maker(output, self.shape)
        return output
       
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#