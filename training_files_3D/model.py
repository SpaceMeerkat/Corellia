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
               
    def __init__(self, nodes, xx, yy, zz, cube, dv):
        super().__init__()
        
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.cube = cube
        self.width = cube.shape[1]
        self.shape = cube.shape[2]
        self.dv = dv
        self.vlim = (cube.shape[1]*dv)/2        

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
        pos = torch.atan2(theta_1,theta_2) + pi
        return pos
    
    def surface_brightness_profile(self, radius_tensor, scale_length):
        """ GET THE SURFACE BRIGHTNESS FOR EACH PIXEL IN THE RADIUS ARRAY """
        sbProf = torch.exp(-radius_tensor/scale_length)
        return sbProf
    
    def velocity_profile(self, radius_tensor, Vh, ah):
        """ GET THE LOS VELOCITIES FOR EACH PIXEL IN THE RADIUS ARRAY """
        vel = torch.sqrt((Vh**2)*(1-((ah/radius_tensor)*torch.atan(radius_tensor/ah))))
        return vel
    
    def de_regularise(self,tensor,minimum,maximum):
        """ RECOVER THE PARAMETERS OUT OF THE NORMALISED RANGE [-1, 1] """
        tensor=tensor+1.
        tensor=tensor*(maximum-minimum)/2.
        tensor=tensor+minimum
        return tensor
    
    def regularise(self,array):
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        array = array - array.min(1)[0].min(1)[0].min(1)[0][:,None,None,None]
        array = array / array.max(1)[0].max(1)[0].max(1)[0][:,None,None,None]
        array = (array*2) - 1
        return array
    
    def normal(self,mu,std,auxiliary,amplitude):
        mu = mu.view(-1)
        amplitude = amplitude.view(-1)
        a = 1/(std*torch.sqrt(2*torch.tensor(pi)))
        b = torch.exp(-0.5*(((auxiliary-mu[:,None])/std)**2))
        return amplitude[:,None]*(a*b)       
    
    def cube_maker(self, x, original):
        """ GENERATE THE OUTPUT CUBE USING PYTORCH FUNCTIONS """ 
        
        ### Define the variables needed for cube modelling from input x
        pos = self.pos_ang(x[:,0].clone(),x[:,1].clone())[:,None,None] ### Poition angle
        inc = self.de_regularise(x[:,2].clone(),5,pi/2.)[:,None,None] ### Inclination
        a = self.de_regularise(x[:,3].clone(),0.1,0.5)[:,None,None] ### SBprof scale length
        a = a * self.shape / 2
        ah = self.de_regularise(x[:,4].clone(),0.01,0.1)[:,None,None] ### DM halo scale length
        ah = ah * self.shape / 2
        Vh = self.de_regularise(x[:,5].clone(),50,500)[:,None,None] ### Maximum velocity allowed
        cube = self.cube.clone()
                       
        ### Create 2D arrays of x,y, and r values
        xx_t = -self.xx*torch.sin(pos) + self.yy*torch.cos(pos)
        yy_t =  self.xx*torch.cos(pos) + self.yy*torch.sin(pos)
        yy_t = yy_t / torch.sin((pi/2)-inc)
        rr_t = torch.sqrt((xx_t**2) + (yy_t**2)) # Now a cube of radii as needed for batches
        zz_t = self.zz
        
        ### Create 2D array of SBprof given some 2D radius array
        sbProf = self.surface_brightness_profile(rr_t, a)
        sbProf = sbProf-sbProf.min()
        sbProf = sbProf / sbProf.max()
        
        ### Define the 2D V(r) given some 2D radius array
        
        vel = self.velocity_profile(rr_t, Vh, ah) # Get velocities V(r)
        vel = vel * torch.cos(torch.atan2(yy_t, xx_t)) * torch.sin(inc) ### Convert to line-of-sight velocities
        vel[vel<-self.vlim] = 0 # Clip velocities out of range
        vel[vel>self.vlim] = 0 # Clip velocities out of range
    

        vel = vel//self.dv # Get channel indices of each pixel
        v = vel.clone() # clone the index array for visualising in trainging script
        vel += self.width/2. # Redistribute channel values to lie in range 0-N
        vel = vel.unsqueeze(1).to(torch.long)
        sbProf = sbProf.unsqueeze(1)
        
        for i in range(cube.shape[0]):
            flattened_cube = self.normal(vel[i],3,zz_t[i], sbProf[i])
            cube[i,:,:,:] = flattened_cube.T.view(120,64,64)
              
        ### Choose between mapping and looping (so far can't see a difference other than speed)
        ### MAPPING
#        for k in range(cube.shape[0]):cube[k,torch.unique(vel[k]).type(torch.LongTensor),:,:] = \
#        torch.stack([*map(vel[k].__eq__,torch.unique(vel[k]))]).type(torch.float)*sbProf[k].type(torch.float) ### Fill cube
        
        ### ANOTHER WAY THAT DOESN'T REQUIRE *MAP FUNCTIONALITY
#        for k in range(cube.shape[0]): cube[k,:,:,:].scatter_(0,vel[:,k,:,:],sbProf[:,k,:,:])
                
        ### LOOPING
#        for k in range(cube.shape[0]):
#            for i in range(self.shape): # Populate channels with sbProf
#                for j in range(self.shape):
#                    v_ind = vel[k,i,j].to(torch.int)
#                    cube[k,v_ind,i,j] = sbProf[k,i,j] 
        #cube[original==0] = 0 # Mask the output by where the input=0

        return cube, v, sbProf
    
    def test_encode(self,x):
        """ CREATE ENCODINGS FOR TESTING AND DEBUGGING """
        output = self.encoder_conv(x)
        output = self.encoder_linear(output)
        return output
        
    def forward(self, x):
        """ CREATE A FORWARD PASS THROUGH THE REQUIRED MODEL FUNCTIONS """
        output = self.encoder_conv(x)
        output = self.encoder_linear(output)
        output, vel, sbProf = self.cube_maker(output, x)
        return output, vel, sbProf
       
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#