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
               
    def __init__(self, xxx=None, yyy=None, zzz=None, pvd=None, mask=None):
        super().__init__()
    
        self.xxx = xxx
        self.yyy = yyy
        self.zzz = zzz
        self.pvd = pvd
        self.mask = mask
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        self.conv0 = torch.nn.Conv2d(1,16,3,1,padding=1)
        self.conv1 = torch.nn.Conv2d(16,32,3,1,padding=1)
        self.conv2 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        
        self.conv4 = torch.nn.Conv2d(1,16,3,1,padding=1)
        self.conv5 = torch.nn.Conv2d(16,32,3,1,padding=1)
        self.conv6 = torch.nn.Conv2d(32,64,3,1,padding=1)
        self.conv7 = torch.nn.Conv2d(64,128,3,padding=1)
        
        self.pool = torch.nn.MaxPool2d(2)

        self.lc1 = torch.nn.Linear(8192,256)
        self.lc2 = torch.nn.Linear(256,2)
        
        self.lc3 = torch.nn.Linear(8192,256)
        self.lc4 = torch.nn.Linear(256,2)
        
        self.lc5 = torch.nn.Linear(128,32)
        self.lc6 = torch.nn.Linear(64,8)
        self.lc7 = torch.nn.Linear(8,2)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        self.conv8 = torch.nn.Conv1d(1, 16, 3, padding=1)
        self.conv9 = torch.nn.Conv1d(16, 32, 3, padding=1)
        self.conv10 = torch.nn.Conv1d(32, 64, 3, padding=1)
        self.conv11 = torch.nn.Conv1d(64, 128, 3, padding=1)
        
        self.pool2 = torch.nn.MaxPool1d(2)        
        
        self.lc8 = torch.nn.Linear(1024, 1024)
        self.lc9 = torch.nn.Linear(1024, 512)
        self.lc10 = torch.nn.Linear(512, 256)
        self.lc11 = torch.nn.Linear(256, 64)
        self.lc12 = torch.nn.Linear(64, 16)
        self.lc13 = torch.nn.Linear(16, 2)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
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
    
    def conv_vel(self,x):
        
        x = self.conv8(x)
        x = self.pool2(x)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.pool2(x)        
        x = self.conv10(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv11(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(int(x.size()[0]),-1)
        return x
    
    def linear_vel(self,x):
        
        x = self.lc8(x)
        x = self.relu(x)
        x = self.lc9(x)
        x = self.relu(x)
        x = self.lc10(x)
        x = self.relu(x)
        x = self.lc11(x)
        x = self.relu(x)
        x = self.lc12(x)
        x = self.relu(x)
        x = self.lc13(x)
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

        vel = (2/pi)*(1.273239)*(torch.atan(pi*radius_tensor))
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
    
    def rotator(self,tensor,angle):
        """ DEROTATE A 2D TENSOR GIVEN SOME POSITION ANGLE"""
        
        tensor = rotate(tensor,-angle)
        return tensor
    
    def cube_maker(self, x, original1, original2, pos, shape):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """  

        inc3 = self.de_regularise(x[:,0].clone(),pi/18,pi/2)[:,None,None,None]  ### Inclination
        a = self.de_regularise(x[:,1].clone(),0.2,1)[:,None,None,None]        ### SBprof scale length
        a = a * shape / 2
        # pos3 = self.pos_ang(x[:,2].clone(),x[:,3].clone())[:,None,None,None]    ### Poition angle
        a_z = a.clone()*0.1
        pos3 = -pos[:,None,None,None]
        
        # pos3 = pos3/pos3 * (pi/4)
        # inc3 = (inc3/inc3) * (pi/4)
               
        # Create radius cube __________________________________________________
        
        rr_t, xx, yy, rr_xy, zz = self.rotation_all(self.xxx,self.yyy,self.zzz,
                                                    inc3, pos3) 
                
        # Create sbProf cube __________________________________________________
               
        sbProf = self.surface_brightness_profile(rr_xy, a, zz, a_z)
                      
        # Create moment 0 map _________________________________________________
        
        mom0 = sbProf.sum(axis=3)
        mom0 = mom0.unsqueeze(1)
        mom0 = self.regularise2(mom0)
               
        # mask = self.rotator(self.mask, pos3.view(-1) * (180/pi))                  # Rotate the mask
        # mom0 = mom0*mask
                      
        # Create velocity slice _______________________________________________
        
        mom1 = original2.clone()
        pos_deg = -90 + pos3.clone().detach().view(-1) * (180/pi)
        mom1 = self.rotator(mom1,pos_deg)                                       # ROtate the mom1 map
        
        # major_slice = mom1[:,:,31:33,:].clone()
        major_slice = mom1[:,:,63:65,:].clone()
        major_slice = major_slice.sum(2)/2
        # major_slice = torch.abs(major_slice)       
                
        # Pass PVD through linear network _____________________________________
        
        params = self.conv_vel(major_slice)
        params = self.linear_vel(params)
        Vh = self.de_regularise(params[:,0].clone(),0.2,1)[:,None]
        ah = self.de_regularise(params[:,1].clone(),0.1,1)[:,None]
        ah = ah * shape / 2
                
        # Recreate PVD ________________________________________________________
        
        inc_pvd = inc3.clone().detach().squeeze(1).squeeze(1)
                       
        PVD_r = self.pvd / ah
        PVD = self.velocity_profile(PVD_r)
        PVD = PVD * Vh * torch.sin(inc_pvd)
        # PVD = torch.abs(PVD)
        
        major_slice = major_slice.squeeze(1)
               
        # Mask moment maps ____________________________________________________
                       
        mom0[original1==0] = 0
        
        return mom0, major_slice, PVD, inc3, pos3, mom1
    
    def test_encode(self,x1,x2, pos):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        
        output1 = self.mom0_encoder(x1)
        output2 = self.mom1_encoder(x2)
        # output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output1,output2)
        
        inc = self.de_regularise(output[:,0].clone(),pi/18,pi/2)[:,None,None,None]  ### Inclination
        a = self.de_regularise(output[:,1].clone(),0.1,1)[:,None,None,None]        ### SBprof scale length
        # pos = self.pos_ang(output[:,2].clone(),output[:,3].clone())[:,None,None,None]    ### Poition angle
        pos = -pos[:,None,None,None]
        
        mom1 = x2.clone()
        pos_deg = -90 + pos.view(-1) * (180/pi)
        mom1 = self.rotator(mom1,pos_deg)
        
        major_slice = mom1[:,:,30:32,:].clone()
        major_slice = major_slice.sum(2)/2
        # major_slice = torch.abs(major_slice)
        
        params = self.conv_vel(major_slice)
        params = self.linear_vel(params)
        Vh = self.de_regularise(params[:,0].clone(),100,500)[:,None]
        ah = self.de_regularise(params[:,1].clone(),0.1,1)[:,None]

        output = torch.tensor([pos,inc,a,ah,Vh])
        return output
        
    def forward(self, x1, x2, pos):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        
        output1 = self.mom0_encoder(x1)
        output2 = self.mom1_encoder(x2)
        #output = torch.cat((output1,output2),-1)
        output = self.encoder_linear(output1,output2)
        output = self.cube_maker(output,x1,x2,pos,shape=128)
        return output
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
