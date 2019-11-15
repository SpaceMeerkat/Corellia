#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:32:41 2019

@author: SpaceMeerkat
"""

#_____________________________________________________________________________#
#_____________________________________________________________________________#

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from tqdm import tqdm

def recover_pos(theta1,theta2):
    """ RECOVERING THE ANGLE AFTER A 2D ROTATION USING EULER MATRICES """
    angle = np.rad2deg(np.arctan2(theta1,theta2)) + 180
    return angle

#_____________________________________________________________________________#
#_____________________________________________________________________________#

path = '/home/corona/c1307135/Semantic_ML/pickle_files/'

#_____________________________________________________________________________#
#_____________________________________________________________________________#

names = [os.path.basename(x) for x in glob.glob(path+'*.pkl')]

#_____________________________________________________________________________#
#_____________________________________________________________________________#

fig, axs = plt.subplots(2, 2, figsize=(12,10))

for n in tqdm(names):

    df = pd.read_pickle(path+n)
    
    predictions = df[[0,1,2,3,4]].values.astype(float)
    target = df[[5,6,7,8,9]].values.astype(float)
    
    phi = target[:,2]
    index = np.where(phi<0.25)[0]
    
    axs[0, 0].scatter(recover_pos(target[:,0],target[:,1]),
                recover_pos(predictions[:,0],predictions[:,1]),s=1,c='k')
    axs[0, 0].scatter(recover_pos(target[index,0],target[index,1]),
                recover_pos(predictions[index,0],predictions[index,1]),s=1,c='r')
    axs[0, 1].scatter(target[:,2]*90,predictions[:,2]*90,s=1,c='k')
    axs[1, 0].scatter(target[:,3],predictions[:,3],s=1,c='k')
#    residuals = (target[:,4]*700*np.sin(np.deg2rad(target[:,2]*90))) - (predictions[:,4]*700*np.sin(np.deg2rad(target[:,2]*90)))
#    one_sigma = np.std(residuals)
#    mean = np.mean(residuals)
    axs[1, 1].scatter(target[:,4]*700*np.sin(np.deg2rad(target[:,2]*90)), predictions[:,4]*700*np.sin(np.deg2rad(target[:,2]*90)),s=1,c='k')
#    axs[1, 1].fill_between(np.linspace(0,700,1000),mean+one_sigma,mean-one_sigma,alpha=0.1,color='r')
    
axs[0, 0].set(xlabel= r'$\theta_{pos} \, (^{\circ}) $', ylabel= r'$\theta_{pos,pred} \, (^{\circ}) $')   
axs[0, 1].set(xlabel= r'$\phi_{inc} \, (^{\circ})$', ylabel= r'$\phi_{inc,pred} \, (^{\circ})$') 
axs[1, 0].set(xlabel= r'$a$', ylabel= r'$a_{pred}$') 
axs[1, 1].set(xlabel= r'$v \, sin(\phi_{inc}) \, (km\,s^{-1})$', ylabel= r'$v_{pred} \, sin(\phi_{inc,pred}) \, (km\,s^{-1})$') 
    
plt.tight_layout()
plt.savefig('/home/corona/c1307135/Semantic_ML/Test_images/cube_velocity_test3.png')
#_____________________________________________________________________________#
#_____________________________________________________________________________#
