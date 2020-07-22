#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:30:48 2020

@author: james
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
import numpy as np

import matplotlib

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['legend.fontsize'] = 17.5
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 20;
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['xtick.major.size'] = 10;
matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['xtick.major.width'] = 2;
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['xtick.minor.size'] = 5;
matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['xtick.minor.width'] = 1;
matplotlib.rcParams['ytick.minor.width'] = 1
matplotlib.rcParams['xtick.direction'] = 'in';
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['ytick.left'] = True
params = {'mathtext.default': 'regular'}
matplotlib.rcParams.update(params)
matplotlib.rcParams['axes.labelsize'] = 30

# df = pd.read_pickle('/home/corona/c1307135/Semantic_ML/Corellia/scale_length_pickle_files/scale_lengths.pkl')
df = pd.read_pickle('/home/james/Documents/mountdir/scale_length_pickle_files/scale_lengths3.pkl')
beams = [4,5,6,7,8,9,10]
colors = ['black','green','orange','chocolate','orangered','red','maroon']

plt.figure(figsize=(16,11))
plt.xlabel(r'$a_{I,true}$')
plt.ylabel('mean offset')
for beam in beams:
    plt.plot(df['a_'+str(beam)].values,df['avg_'+str(beam)].values,'-',color=colors[beam-4],label=str(beam))
    plt.fill_between(df['a_'+str(beam)].values, df['avg_'+str(beam)].values - df['std_'+str(beam)].values,
                     df['avg_'+str(beam)].values + df['std_'+str(beam)].values, alpha=0.2,color=colors[beam-4])
    # const = np.sqrt( (df['a_'+str(beam)].values**2) + ((beam/32)**2))  - df['a_'+str(beam)].values
    # const = [(beam/32) **2 * 0.8] * len(const) 
    # plt.plot(df['a_'+str(beam)].values,const,'k--')

plt.ylabel(r'med ($a_{I,pred} - a_{I,true}$)')
plt.legend(title='Beamsize',ncol=2)#,fontsize='small')
plt.tight_layout()
plt.savefig('/home/james/Documents/mountdir/Test_images/2D/offsets.png')

