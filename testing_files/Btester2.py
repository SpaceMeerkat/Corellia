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

fig, axs = plt.subplots(1, 2, figsize=(10,5))

for n in tqdm(names):

    df = pd.read_pickle(path+n)
    
    predictions = df[[0,1,2]].values.astype(float)
    target = df[[3,4,5]].values.astype(float)
    
    a_t = target[:,2]
    a_p = predictions[:,2]
    index_a = np.where(a_t>0.5)
    
    pos_t = np.rad2deg(target[:,0])
    pos_p = np.rad2deg(predictions[:,0])
        
    phi_t = np.rad2deg(target[:,1])
    index = np.where(phi_t<40)[0]
    phi_p = np.rad2deg(predictions[:,1])
    
    axs[1].scatter(phi_t,phi_p,s=1,c='k')
    axs[1].scatter(phi_t[index_a],phi_p[index_a],s=1,c='r')
    axs[0].scatter(a_t,a_p,s=1,c='k')
    axs[0].scatter(target[index,2],predictions[index,2],s=3,c='r')
    
axs[1].set(xlabel= r'$\phi_{inc} \, (^{\circ})$', ylabel= r'$\phi_{inc,pred} \, (^{\circ})$') 
axs[0].set(xlabel= r'$a$', ylabel= r'$a_{pred}$') 
    
plt.tight_layout()
plt.savefig('/home/corona/c1307135/Semantic_ML/Corellia/Test_images/2D/semantic_AE_REMOTE.png')
#_____________________________________________________________________________#
#_____________________________________________________________________________#
