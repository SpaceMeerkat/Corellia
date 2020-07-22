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

ew=0.5
ms = 1

use_errors = False

beam_incs = []

a_p = []
a_t = []

for n in tqdm(range(len(names))):

    df = pd.read_pickle(path+names[n])
    
    predictions = df[[0,1,2,3,4]].values.astype(float)
    target = df[[5,6,7,8,9]].values.astype(float)
    
    if use_errors == True:
        errors = df[[10,11,12,13,14]].values.astype(float)
    else:
        errors = np.zeros(predictions.shape)
    
    pos_t = np.rad2deg(target[:,0])
    pos_p = -np.rad2deg(predictions[:,0])
    pos_e = np.rad2deg(errors[:,0])
        
    phi_t = np.rad2deg(target[:,1])
    index = np.where(phi_t<40)[0]
    phi_p = np.rad2deg(predictions[:,1])
    phi_e = np.rad2deg(errors[:,1])
    
    vs_p = predictions[:,3]
    vs_t = target[:,3]
    
    out = np.where(vs_p-vs_t > 0.05)[0]
    
    beam1 = target[:,2]*2.3*np.cos(target[:,1])*32
    beam1 = np.exp(-2/beam1)
    beam2 = np.sin(target[:,1])
    beam2 = np.exp(-2/beam2)
    beams = np.vstack([beam1,beam2]).max(0)
    beam = np.where(beams<=1/np.exp(1))[0]
    beam_incs.append(phi_t[beam]) 
    
    a_p.append(np.array(predictions[:,2]))
    a_t.append(np.array(target[:,2]))   
    
    # axs[0, 0].scatter(pos_t,pos_p,s=1,c='k')
    # axs[0, 0].scatter(pos_t[index],pos_p[index],s=1,c='r')
    
    axs[0, 0].errorbar(phi_t,phi_p,yerr=phi_e,fmt='k.',ecolor='k',elinewidth=ew,ms = ms)
    axs[0, 0].errorbar(phi_t[beam],phi_p[beam],yerr=phi_e[beam],fmt='r.',ecolor='r',elinewidth=ew,ms = ms)
    axs[0, 0].plot([10,90],[10,90],'b--')
    axs[0, 0].plot([70,80],[80,70],'g--')
    axs[0, 1].errorbar(target[:,2],predictions[:,2],yerr=errors[:,2],fmt='k.',ecolor='k',elinewidth=ew,ms = ms)
    axs[0, 1].errorbar(target[beam,2],predictions[beam,2],yerr=errors[beam,2],
                      fmt='r.',ecolor='r',elinewidth=ew,ms = ms,label='i < 45$^{\circ}$')
    if n == 0:
        axs[0, 1].legend(loc='best')
    axs[0, 1].plot([0.1,0.35],[0.1,0.35],'b--')
    axs[1, 0].errorbar(target[:,4], predictions[:,4],yerr=errors[:,4],fmt='k.',ecolor='k',elinewidth=ew,ms = ms)
    axs[1, 0].errorbar(target[out,4], predictions[out,4],yerr=errors[out,4],
                       fmt='r.',ecolor='r',elinewidth=ew,ms = ms,label='i < 45$^{\circ}$')
    axs[1, 0].plot([50,500],[50,500],'b--')
    if n == 0:
        axs[1, 0].legend(loc='best')
    axs[1, 1].errorbar(target[:,3],predictions[:,3],yerr=errors[:,3],fmt='k.',ecolor='k',elinewidth=ew,ms = ms)
    axs[1, 1].errorbar(target[out,3],predictions[out,3],yerr=errors[out,3],
                       fmt='r.',ecolor='r',elinewidth=ew,ms = ms,label='i < 45$^{\circ}$')
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


plt.figure()
plt.hist(np.hstack(beam_incs),bins=15)
plt.xlabel(r'i $(^{\circ})$')
plt.ylabel('Counts')
plt.savefig('/home/corona/c1307135/Semantic_ML/Corellia/Test_images/2D/beam_incs.png')

a_t, a_p = np.array(a_t).reshape(-1),np.array(a_p).reshape(-1)
offset = a_p - a_t

all_a = np.vstack([a_t,offset]).T
all_a = sorted(all_a,key=lambda x: x[0])
df_a = pd.DataFrame(all_a)

beam = 10

scale_means = df_a.rolling(50).median().drop(0)
scale_stds = df_a.rolling(50).std().drop(0)
scale_df = pd.concat([scale_means,scale_stds[1]],axis=1)
scale_df.columns = ['a_'+str(beam),'avg_'+str(beam),'std_'+str(beam)]

# scale_means = []
# scale_stds = []

# bounds = [0.1,0.15,0.2,0.25,0.3,0.35]
# for _ in range(len(bounds)-1):
#     offset = a_p - a_t
#     inds = np.where((a_t>bounds[_])&(a_t<bounds[_+1]))[0]
#     scale_means.append(np.median(offset[inds]))
#     scale_stds.append(np.std(offset[inds]))

df_t = pd.read_pickle('/home/corona/c1307135/Semantic_ML/Corellia/scale_length_pickle_files/scale_lengths3.pkl')
df = pd.concat([df_t,scale_df],axis=1)
df.to_pickle('/home/corona/c1307135/Semantic_ML/Corellia/scale_length_pickle_files/scale_lengths3.pkl')

plt.figure()
plt.plot(scale_df['a_'+str(beam)].values,scale_df['avg_'+str(beam)].values,'b.-')
plt.fill_between(scale_df['a_'+str(beam)].values,scale_df['avg_'+str(beam)].values - scale_df['std_'+str(beam)].values,
                 scale_df['avg_'+str(beam)].values + scale_df['std_'+str(beam)].values, alpha=0.5,color='b')
plt.xlabel('bin number')
plt.ylabel('mean offset')
plt.savefig('/home/corona/c1307135/Semantic_ML/Corellia/Test_images/2D/offsets.png')
#_____________________________________________________________________________#
#_____________________________________________________________________________#
