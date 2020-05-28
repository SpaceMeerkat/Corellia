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

#_____________________________________________________________________________#
#_____________________________________________________________________________#

path = '/home/corona/c1307135/Semantic_ML/Corellia/pickle_files/'

#_____________________________________________________________________________#
#_____________________________________________________________________________#

names = [os.path.basename(x) for x in glob.glob(path+'*.pkl')]

#_____________________________________________________________________________#
#_____________________________________________________________________________#

fig, axs = plt.subplots(2, 2, figsize=(10,10))

for n in tqdm(range(len(names))):

    df = pd.read_pickle(path+names[n])
    
    predictions = df[[0,1,2,3,4]].values.astype(float)
    target = df[[5,6,7,8,9]].values.astype(float)
    
    pos_t = np.rad2deg(target[:,0])
    pos_p = -np.rad2deg(predictions[:,0])
        
    phi_t = np.rad2deg(target[:,1])
    index = np.where(phi_t<40)[0]
    phi_p = np.rad2deg(predictions[:,1])
    
    # axs[0, 0].scatter(pos_t,pos_p,s=1,c='k')
    # axs[0, 0].scatter(pos_t[index],pos_p[index],s=1,c='r')
    axs[0, 0].scatter(phi_t,phi_p,s=1,c='k')
    axs[0, 0].plot([10,90],[10,90],'b--')
    axs[0, 1].scatter(target[:,2],predictions[:,2],s=1,c='k')
    axs[0, 1].scatter(target[index,2],predictions[index,2],s=3,c='r',label='i < 45$^{\circ}$')
    if n == 0:
        axs[0, 1].legend(loc='best')
    axs[0, 1].plot([0.1,0.35],[0.1,0.35],'b--')
    axs[1, 0].scatter(target[:,4], predictions[:,4],s=1,c='k')
    axs[1, 0].scatter(target[index,4], predictions[index,4],s=1,c='r',label='i < 45$^{\circ}$')
    axs[1, 0].plot([50,500],[50,500],'b--')
    if n == 0:
        axs[1, 0].legend(loc='best')
    axs[1, 1].scatter(target[:,3],predictions[:,3],s=1,c='k')
    axs[1, 1].scatter(target[index,3],predictions[index,3],s=1,c='r',label='i < 45$^{\circ}$')
    axs[1, 1].plot([0.1,0.8],[0.1,0.8],'b--')
    if n == 0:
        axs[1, 1].legend(loc='best')
    
# axs[0, 0].set(xlabel= r'$\theta_{pos} \, (^{\circ}) $', ylabel= r'$\theta_{pos,pred} \, (^{\circ}) $')   
axs[0, 0].set(xlabel= r'i $(^{\circ})$', ylabel= r'i$_{pred} \, (^{\circ})$') 
axs[0, 1].set(xlabel= r'$a_{I}$', ylabel= r'$a_{I,pred}$') 
axs[1, 0].set(xlabel= r'$V_{max} \, sin(i) \, (km\,s^{-1})$', 
   ylabel= r'$V_{max,pred} \, sin(i) \, (km\,s^{-1})$') 
axs[1, 1].set(xlabel= r'$V_{scale}$', ylabel= r'$V_{scale,pred}$') 
    
plt.tight_layout()
plt.savefig('/home/corona/c1307135/Semantic_ML/Corellia/Test_images/2D/semantic_AE_REMOTE.png')
#_____________________________________________________________________________#
#_____________________________________________________________________________#
