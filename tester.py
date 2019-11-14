#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:33:59 2019

@author: SpaceMeerkat
"""
#_____________________________________________________________________________#
#_____________________________________________________________________________#

import numpy as np
import torch
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import CovNet
from functions import return_cube, recover_pos
import pandas as pd

#_____________________________________________________________________________#
#_____________________________________________________________________________#

model = CovNet(5)
model.load_state_dict(torch.load('CovNet_pilot4.pt'))
model = model.cpu()
model.train(False)
print("Model cast to CPU")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

index = np.arange(0,1000,1)
num_cores = 14
pool = mp.Pool(num_cores,maxtasksperchild=100)
results = list(pool.imap(return_cube,index))
pool.close()

batch = np.array([r[0] for r in results])
target = np.array([r[1:] for r in results])

print("Test data created")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

batch = np.transpose(batch,(0,3,1,2))

predictions = []
for j in range(batch.shape[0]):    
    prediction1, prediction2 = model(torch.tensor(batch[j]).to(torch.float).unsqueeze(0))  
    prediction1, prediction2 = prediction1.detach().numpy(), prediction2.detach().numpy()
    predictions.append(np.append(prediction1, prediction2))
      
print("Testing data complete")

predictions = np.vstack(predictions)

dfp = pd.DataFrame(predictions)
dft = pd.DataFrame(np.vstack(target))

df = pd.concat([dfp,dft],axis=1)
df.columns = [0,1,2,3,4,5,6,7,8,9]

df.to_pickle('/home/corona/c1307135/Semantic_ML/pickle_files/'+str(np.random.uniform(0,1,1)[0])+'.pkl')

#_____________________________________________________________________________#
#_____________________________________________________________________________#

plotter = False

if plotter ==True:

    phi = target[:,2]
    index = np.where(phi<0.25)[0]
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(221)
    plt.scatter(recover_pos(target[:,0],target[:,1]),
                recover_pos(predictions[:,0],predictions[:,1]),s=1,c='k')
    plt.scatter(recover_pos(target[index,0],target[index,1]),
                recover_pos(predictions[index,0],predictions[index,1]),s=1,c='r')
    plt.xlabel(r'$\theta_{pos}$'); plt.ylabel(r'$\theta_{pos,pred}$')
    
    plt.subplot(222)
    plt.scatter(target[:,2]*90,predictions[:,2]*90,s=1,c='k')
    plt.xlabel(r'$\phi_{inc}$'); plt.ylabel(r'$\phi_{inc,pred}$')
    
    plt.subplot(223)
    plt.scatter(target[:,3],predictions[:,3],s=1,c='k')
    plt.xlabel(r'$a$'); plt.ylabel(r'$a_{pred}$')    
    
    plt.subplot(224)
    plt.scatter(target[:,4]*700*np.sin(np.deg2rad(target[:,2]*90)),predictions[:,4]*700*np.sin(np.deg2rad(target[:,2]*90)),s=1,c='k')
    #plt.scatter(target[:,4]*700,predictions[:,4]*700,s=1,c='k')
    plt.xlabel(r'$v (km\,s^{-1})$'); plt.ylabel(r'$v_{pred} (km\,s^{-1})$')  
    
    plt.tight_layout()
    plt.savefig('/home/corona/c1307135/Semantic_ML/Test_images/cube_velocity_test.png')

#_____________________________________________________________________________#
#_____________________________________________________________________________#

