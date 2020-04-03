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
    """ PYTORCH CONVOLUTIONAL AUTOENCODER WITH SCALE_LENGTH FUNCTIONAL DECODER """
               
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
        self.lc3 = torch.nn.Linear(2048,256)
        self.lc4 = torch.nn.Linear(256,4)
        self.relu = torch.nn.ReLU()
        self.hardtanh = torch.nn.Hardtanh(min_val=-1,max_val=1.)
       
    def BRIGHTNESS_encoder(self,x):
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
    
    def VELOCITY_encoder(self,x):
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
    
    def encoder_linear(self,x1,x2):
        """ LINEARLY CONNECTED LAYERS FOR REGRESSION OF REQUIRED PARAMETERS """
        
        x1 = self.lc1(x1)
        x1 = self.relu(x1)
        x1 = self.lc2(x1)
        x1 = self.hardtanh(x1)

        x2 = self.lc3(x2)
        x2 = self.relu(x2)
        x2 = self.lc4(x2)
        x2 = self.hardtanh(x2)

        x = torch.cat((x1,x2),-1)
        return x
    
    def pos_ang(self, theta_1, theta_2):
        """ RECOVERING THE ANGLE AFTER SCALE_LENGTH 2D ROTATION USING EULER MATRICES """

        pos = torch.atan2(theta_1,theta_2) 
        return pos
    
    def create_brightness_values(self, radius_tensor, scale_length, z, scale_length_z):
        """ GET THE SURFACE BRIGHTNESS FOR EACH PIXEL IN THE RADIUS ARRAY """
        
        sbProf = torch.exp((-radius_tensor/scale_length)+(-z/scale_length_z))
        return sbProf
    
    def create_velocity_values(self, radius_tensor):
        """ GET THE LINE OF SIGHT VELOCITYOCITIES FOR EACH VOXEL IN THE RADIUS CUBE """

        vel = (2/pi)*(1.273239)*(torch.atan(pi*radius_tensor)) # 1.273... = 1/arctan(1)
        return vel
    
    def de_regularise(self,tensor,minimum,maximum):
        """ RECOVER THE PARAMETERS OUT OF THE NORMALISED RANGE [-1, 1] """
        
        tensor=tensor+1.
        tensor=tensor*(maximum-minimum)/2.
        tensor=tensor+minimum
        return tensor
        
    def regularise(self,array):
        """ NORMALISES THE INPUT ARRAYS INTO THE RANGE [0,1] """
        
        array = array - array.min(1)[0].min(1)[0].min(1)[0][:,None,None,None]
        array = array + torch.tensor(1e-10)
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        return array
       
    def rotation_all(self,x,y,z,X_ANGLE,Z_ANGLE):
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
    
    def cube_maker(self, x, BRIGHTNESS_INPUT, VELOCITY_INPUT, shape):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """  

        Z_ANGLE = self.pos_ang(x[:,2].clone(),x[:,3].clone())[:,None,None,None]    ### Poition angle
        X_ANGLE = self.de_regularise(x[:,0].clone(),pi/18,pi/2)[:,None,None,None]   ### Inclination
        SCALE_LENGTH = self.de_regularise(x[:,1].clone(),0.1,0.5)[:,None,None,None]       ### SBprof scale length
        SCALE_LENGTH = SCALE_LENGTH * shape / 2
        V_SCALE_LENGTH = self.de_regularise(x[:,5].clone(),0.1,0.5)[:,None,None,None]     ### DM halo scale length
        V_SCALE_LENGTH = V_SCALE_LENGTH * shape / 2
        MAX_VELOCITY = self.de_regularise(x[:,4].clone(),0.1,1)[:,None,None,None]       ### Maximum velocity allowed
        a_z = SCALE_LENGTH.clone()*2
                                             
        # Create radius cube __________________________________________________

        rr_t, xx, yy, XY_RADIUS_ARRAY, zz = self.rotation_all(self.xxx,self.yyy,self.zzz,X_ANGLE,Z_ANGLE) 
        
        # Create brightness cube ______________________________________________
               
        sbProf = self.create_brightness_values(XY_RADIUS_ARRAY, SCALE_LENGTH, zz, a_z)
                      
        # Create moment 0 map ________________________________________________
        
        BRIGHTNESS = sbProf.sum(axis=3)
        BRIGHTNESS = BRIGHTNESS.unsqueeze(1)
        BRIGHTNESS = self.regularise(BRIGHTNESS)
        
        # Create VELOCITY cube ________________________________________________
        
        xmax = xx.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        ymax = yy.max(1)[0].max(1)[0].max(1)[0][:,None,None,None] 
        
        rr_t_v = rr_t.clone()
        rr_t_v = rr_t_v / V_SCALE_LENGTH
            
        # Convert to line-of-sight VELOCITY ___________________________________
        
        VELOCITY = self.create_velocity_values(rr_t_v)                                     
        thetas = torch.atan2(yy/ymax, xx/xmax) - pi/2
        thetas = torch.sin(thetas)
        thetas = thetas * torch.sin(X_ANGLE) 
        VELOCITY = VELOCITY * thetas * MAX_VELOCITY
        #VELOCITY = (VELOCITY*MAX_VELOCITY)/VELOCITY.sum(3).max(1)[0].max(1)[0][:,None,None,None]
               
        # Create moment 1 map _________________________________________________
        
        BRIGHTNESS_COPY = sbProf.clone()
        BRIGHTNESS_COPY = BRIGHTNESS_COPY 
        BRIGHTNESS_COPY = BRIGHTNESS_COPY + 1e-10
        BRIGHTNESS_COPY_FLAT = BRIGHTNESS_COPY.sum(axis=3).unsqueeze(1)
                
        VELOCITY = VELOCITY * BRIGHTNESS_COPY
        #VELOCITY = (VELOCITY*MAX_VELOCITY)/VELOCITY.sum(3).max(1)[0].max(1)[0][:,None,None,None]
        VELOCITY = VELOCITY.sum(axis=3)
        VELOCITY = VELOCITY.unsqueeze(1)
        VELOCITY = VELOCITY / BRIGHTNESS_COPY_FLAT
        #VELOCITY = VELOCITY / 500
               
        # Mask moment maps ____________________________________________________
                
        VELOCITY[VELOCITY_INPUT==0] = 0
        BRIGHTNESS[BRIGHTNESS_INPUT==0] = 0
        
        vmax = VELOCITY.max(1)[0].max(1)[0].max(1)[0]

        return BRIGHTNESS, VELOCITY, X_ANGLE, Z_ANGLE, vmax
    
    def test_encode(self,x1,x2):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        
        output1 = self.BRIGHTNESS_encoder(x1)
        output2 = self.VELOCITY_encoder(x2)
        #output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output1,output2)
        Z_ANGLE = self.pos_ang(output[:,2].clone(),output[:,3].clone())[:,None,None] 
        X_ANGLE = self.de_regularise(output[:,0].clone(),pi/18,pi/2)[:,None,None]  
        SCALE_LENGTH = self.de_regularise(output[:,1].clone(),0.1,0.5)[:,None,None]         
        V_SCALE_LENGTH = self.de_regularise(output[:,5].clone(),0.1,0.5)[:,None,None]       
        MAX_VELOCITY = self.de_regularise(output[:,4].clone(),0.1,1)[:,None,None]       
        output = torch.tensor([Z_ANGLE,X_ANGLE,SCALE_LENGTH,V_SCALE_LENGTH,MAX_VELOCITY])
        return output
        
    def forward(self, x1, x2):
        """ CREATE SCALE_LENGTH FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        
        output1 = self.BRIGHTNESS_encoder(x1)
        output2 = self.VELOCITY_encoder(x2)
        #output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output1,output2)
        output = self.cube_maker(output,x1,x2,shape=64)
        return output
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
