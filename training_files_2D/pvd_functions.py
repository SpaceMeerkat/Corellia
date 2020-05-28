#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:59:24 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#

from astropy.nddata.utils import Cutout2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from cube_generator import cube_generator
from sauron_colormap import sauron

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def learning_rate(initial_lr, epoch):   
    """Sets the learning rate to the initial LR decayed by a factor of 10 every
    N epochs"""
    
    lr = initial_lr * (0.975 ** (epoch// 1))
    
    return lr 

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def return_cube(i):
    """ MULTIPROCESS CREATION OF CUBES WITH VARIOUS POSITION ANGLES AND
        INCLINATIONS """
        
    pos_ang = random.uniform(-180,180)
    inc_ang = random.uniform(10,90)
    a = random.uniform(0.1,1)
    ah = random.uniform(0.1,1)
    Vh = random.uniform(100,500)
    
    # Vh = 500
    # a = 0.5
    # ah = 0.25
    # pos_ang = 45
    # inc = [10,20,30,40,50,60,70,80]
    # inc_ang = 45
    
    cube = cube_generator(a=a,pos_ang=pos_ang,
                          inc_ang=inc_ang,resolution=1000,ah=ah,
                          Vh=Vh).cube_creation()  
    
    #cube[cube<=3*np.std(cube[:10,:,:])]=np.nan
    mom0 = np.nansum(cube,axis=2) 
    mom1 = np.nansum(cube*np.arange(-600,600,10)[None,None,:],axis=2) / mom0
    mom1 /= 500
    mom0 -= np.nanmin(mom0)
    mom0 /= np.nanmax(mom0)
    
    # mom0[mom1<0] *= -1
           
    pos_ang = np.deg2rad(pos_ang)
    inc_ang = np.deg2rad(inc_ang)
        
    return (mom0,mom1),pos_ang,inc_ang,a,ah,Vh

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
    

def plotter(mom0_p, PVD_p, mom0, PVD, inc, pos, out_dir, mom1):    
    """ PLOTTING THE PREDICTED AND TRUE VELOCITY FIELDS """
    
    mom0_p = mom0_p.detach().cpu().numpy()
    PVD_p = PVD_p.detach().cpu().numpy()
    mom0 = mom0.detach().cpu().numpy()
    PVD = PVD.detach().cpu().numpy()
    inc = np.rad2deg(inc.detach().cpu().numpy())
    pos = np.rad2deg(pos.detach().cpu().numpy())
    mom1 = mom1.detach().cpu().numpy()

    for i in range(PVD_p.shape[0]):
        fig, axs = plt.subplots(3, 2, figsize=(5.8,7))
        
        b2 = PVD[i]
        axs[1,0].plot(b2)
        axs[1,0].set_ylim([-1,1])
        
        b1 = PVD_p[i]
        axs[1,1].plot(b1)
        axs[1,1].set_ylim([-1,1])
        
        b = mom0_p[i,0,:,:]
        im1 = axs[0,1].imshow(b,cmap='magma',aspect="auto")
        plt.colorbar(im1, fraction=0.046, pad=0.04, ax = axs[0,1])
        axs[0,1].title.set_text('PREDICTED %.2f' %inc[i])
    
        b = mom0[i,0,:,:]
        im2 = axs[0,0].imshow(b,cmap='magma',aspect="auto")
        plt.colorbar(im2, fraction=0.046, pad=0.04, ax=axs[0,0])
        axs[0,0].title.set_text('TRUE %.2f' %pos[i])
        
        im3 = axs[2,0].imshow(mom1[i,0,:,:],aspect="auto")
        plt.colorbar(im3,ax=axs[2,0])
        
        plt.tight_layout()
        plt.savefig(out_dir+str(i)+'.png')
    
    return

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def makebeam(xpixels, ypixels, beamSize, cellSize=1, cent=None):
        """
        Creates a psf with which one can convolve their cube based on the beam provided.
        
        :param xpixels:
                (float or int) Number of pixels in the x-axis
        :param ypixels:
                (float or int) Number of pixels in the y-axis
        :param beamSize:
                (float or int, or list or array of float or int) Scalar or three element list for size of convolving beam (in arcseconds). If a scalar then beam is
                assumed to be circular. If a list/array of length two. these are the sizes of the major and minor axes,
                and the position angle is assumed to be 0. If a list/array of length 3, the first 2 elements are the
                major and minor beam sizes, and the last the position angle (i.e. [bmaj, bmin, bpa]).
        :param cellSize:
                (float or int) Pixel size required (arcsec/pixel)
        :param cent: 
            (array or list of float or int) Optional, default value is [xpixels / 2, ypixels / 2].
                Central location of the beam in units of pixels.
        :return psf or trimmed_psf:
                (float array) psf required for convlution in self.model_cube(). trimmed_psf returned if self.huge_beam=False, 
                otherwise default return is the untrimmed psf.              
        """

        if not cent: cent = [xpixels / 2, ypixels / 2]

        beamSize = np.array(beamSize)

        try:
            if len(beamSize) == 2:
                beamSize = np.append(beamSize, 0)
            if beamSize[1] > beamSize[0]:
                beamSize[1], beamSize[0] = beamSize[0], beamSize[1]
            if beamSize[2] >= 180:
                beamSize[2] -= 180
        except:
            beamSize = np.array([beamSize, beamSize, 0])

        st_dev = beamSize[0:2] / cellSize / 2.355

        rot = beamSize[2]

        if np.tan(np.radians(rot)) == 0:
            dirfac = 1
        else:
            dirfac = np.sign(np.tan(np.radians(rot)))

        x, y = np.indices((int(xpixels), int(ypixels)), dtype='float')

        x -= cent[0]
        y -= cent[1]

        a = (np.cos(np.radians(rot)) ** 2) / (2 * st_dev[1] ** 2) + (np.sin(np.radians(rot)) ** 2) / \
            (2 * (st_dev[0] ** 2))

        b = (dirfac * (np.sin(2 * np.radians(rot)) ** 2) / (4 * st_dev[1] ** 2)) + ((-1 * dirfac) * \
            (np.sin(2 * np.radians(rot)) ** 2) / (4 * st_dev[0] ** 2))

        c = (np.sin(np.radians(rot)) ** 2) / (2 * st_dev[1] ** 2) + (np.cos(np.radians(rot)) ** 2) / \
            (2 * st_dev[0] ** 2)

        psf = np.exp(-1 * (a * x ** 2 - 2 * b * (x * y) + c * y ** 2))

        ### Trim around high values in the psf, to speed up the convolution ###

        psf[psf < 1e-5] = 0  # set all kernel values that are very low to zero

        # sum the psf in the beam major axis
        if 45 < beamSize[2] < 135:
            flat = np.sum(psf, axis=1)
        else:
            flat = np.sum(psf, axis=0)

        idx = np.where(flat > 0)[0]  # find the location of the non-zero values of the psf

        newsize = (idx[-1] - idx[0])  # the size of the actual (non-zero) beam is this

        if newsize % 2 == 0:
            newsize += 1  # add 1 pixel just in case
        else:
            newsize += 2  # if necessary to keep the kernel size odd, add 2 pixels

        trimmed_psf = Cutout2D(psf, (cent[1], cent[0]), newsize).data  # cut around the psf in the right location
        trimmed_psf = torch.tensor(trimmed_psf)

        return trimmed_psf









