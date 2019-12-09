#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:24 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from sauron_colormap import sauron
from cube_generator import cube_generator

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def learning_rate(initial_lr, epoch):
    
    """Sets the learning rate to the initial LR decayed by a factor of 10 every
    N epochs"""
    
    lr = initial_lr * (0.5 ** (epoch// 1))
    
    return lr 

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def return_cube(i):

    """ MULTIPROCESS CREATION OF CUBES WITH VARIOUS POSITION ANGLES AND
        INCLINATIONS """
        
    pos_ang = random.uniform(0,360)
    inc_ang = random.uniform(5,90)
    scale_frac = random.uniform(0.1,0.5)
    ah = random.uniform(0.01,0.1)
    Vh = random.uniform(100,500)
    cube = cube_generator(scale_length_frac=scale_frac,pos_ang=pos_ang,
                          inc_ang=inc_ang,resolution=1000,ah=ah,Vh=Vh).cube_creation()
       
    cube/=cube.max()
    cube[cube<np.std(cube)]=0
    pos_ang = np.deg2rad(pos_ang)
        
    return cube,np.cos(pos_ang),np.sin(pos_ang),inc_ang,scale_frac,ah,Vh

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def pos_loss(output, target):
    
    """ LOSS FUNCTION FOR POSITION ANGLE BASED ON MINIMUM ARC SEPARATION """
        
    loss = torch.mean(torch.stack([ (output-target)**2,
                                   (1-torch.abs(output-target))**2] ).min(dim=0)[0]) 
    
    return loss

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    
def recover_pos(theta1,theta2):
    
    """ RECOVERING THE ANGLE AFTER A 2D ROTATION USING EULER MATRICES """
    
    angle = np.rad2deg(np.arctan2(theta1,theta2)) + 180
    
    return angle

#_____________________________________________________________________________#
#_____________________________________________________________________________#
    

def plotter(batch, prediction, vel, sbProf, out_dir):
    
    """ PLOTTING THE PREDICTED AND TRUE VELOCITY FIELDS """
    
    prediction = prediction.detach().cpu().numpy()
    for i in range(prediction.shape[0]):
        b = prediction[i,:,:,:]
        mom0 = np.sum(b,axis=0) + 1e-10
        channels = np.arange(-600,600,10)
        num = np.sum(b*channels[:,None,None],axis=0)
        mom1 = num/mom0

        plt.figure()
        plt.imshow(mom1)
        plt.colorbar()
        plt.savefig(out_dir+'3D/predicted'+str(i)+'.png')
    plt.close('all')
       
    batch = batch.detach().cpu().numpy()
    for i in range(prediction.shape[0]):
        b = batch[i,:,:,:]
        mom0 = np.sum(b,axis=0) + 1e-10
        channels = np.arange(-600,600,10)
        num = np.sum(np.transpose(b,(1,2,0))*channels,axis=2)
        mom1 = num/mom0
        plt.figure()
        plt.imshow(mom1)
        plt.colorbar()
        plt.savefig(out_dir+'3D/True'+str(i)+'.png')
    plt.close('all')
    
#    vel = vel[0,:,:].detach().cpu().numpy()
#    plt.figure()
#    plt.imshow(vel)
#    plt.colorbar()
#    plt.savefig(out_dir+'vel.png')
#    
#    sbProf = sbProf[0,:,:].detach().cpu().numpy()
#    plt.figure()
#    plt.imshow(np.log(sbProf+(1e-10)))
#    plt.colorbar()
#    plt.savefig(out_dir+'sbProf.png')
    
    return


def moment_one(cube):
        plt.close('all')
        mom0 = np.sum(cube,axis=2) + 1e-10
        channels = np.arange(-600,600,10)
        num = np.sum(cube*channels[None,None,:],axis=2)
        mom1 = num/mom0
        mom1+= np.random.uniform(0,0.01*np.std(mom1),(64,64))
        
        plt.figure()
        plt.imshow(mom1,cmap=sauron)
        plt.colorbar()
        plt.show()
        
        return mom1
    
def moment_two(cube):
        #plt.close('all')
        mom0 = np.sum(cube,axis=2) + 1e-10
        channels = np.arange(-600,600,10)
        num = np.sum(cube*channels[None,None,:],axis=2)
        mom1 = num/mom0
        
        vel_narray = np.ones((cube.shape))
        vel_narray *= channels[None,None,:]
                       
        mom2 = np.sqrt(np.sum(abs(cube) * (vel_narray - mom1[:,:,None])**2., axis=2) / (np.sum(abs(cube), axis=2)+1e-10))

        fig, axes = plt.subplots(nrows=1, ncols=2)
        im1 = axes[0].imshow(mom1,cmap=sauron)
        plt.colorbar(im1, ax=axes[0],fraction=0.046, pad=0.04)
        im2 = axes[1].imshow(mom2,cmap=sauron)
        plt.colorbar(im2, ax=axes[1],fraction=0.046, pad=0.04)
        plt.tight_layout()
        return mom1






