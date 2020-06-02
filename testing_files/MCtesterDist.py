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
import sys
sys.path.append("../training_files_2D/")
from model2 import CAE
from functions import return_cube
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#_____________________________________________________________________________#
#_____________________________________________________________________________#

model_path = '/home/corona/c1307135/Semantic_ML/Corellia/models/'

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train(True)

model = CAE()
model.load_state_dict(torch.load(model_path+'semantic_AE3_.pt'))
model = model.cpu()
model.train(False)
model.apply(apply_dropout)
print("Model cast to CPU")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

index = np.array([1])
num_cores = 14
pool = mp.Pool(num_cores,maxtasksperchild=100)
results = list(pool.imap(return_cube,index))
pool.close()

batch = np.array([r[0] for r in results])
target = np.array([r[1:] for r in results])

batch = torch.tensor(batch).to(torch.float)
batch[batch!=batch]=0
target = torch.tensor(target).to(torch.float)

pos = target[:,0]
ah = target[:,-2]

mom0s, mom1s = batch[:,0,:,:].unsqueeze(1), batch[:,1,:,:].unsqueeze(1)

print("Test data created")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

predictions = []
errors = []
temp_pred = []
for _ in range(10000):
    prediction1 = model.test_encode(mom0s,mom1s,pos)
    prediction1 = prediction1.detach().numpy()
    temp_pred.append(prediction1)
temp_pred = np.vstack(temp_pred)
mean_pred = np.mean(temp_pred,0)
predictions.append(mean_pred)
errors.append(np.sum(np.abs(temp_pred-mean_pred[None,:]),0)/len(temp_pred))

fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0, 0].hist(np.rad2deg(temp_pred[:,1]),bins=50)
axs[0, 0].set_xlim(10,90)
axs[0, 1].hist(temp_pred[:,2],bins=50)
axs[0, 1].set_xlim(0.1,0.35)
axs[1, 0].hist(temp_pred[:,4],bins=50)
axs[1, 0].set_xlim(50,500)
axs[1, 1].hist(temp_pred[:,3],bins=50)
axs[1, 1].set_xlim(0.1,0.8)
    
axs[0, 0].set(xlabel= r'i $(^{\circ})$') 
axs[0, 1].set(xlabel= r'$a_{I}$') 
axs[1, 0].set(xlabel= r'$V_{max} \, sin(i) \, (km\,s^{-1})$') 
axs[1, 1].set(xlabel= r'$V_{scale}$') 
    
plt.tight_layout()
plt.savefig('/home/corona/c1307135/Semantic_ML/Corellia/Test_images/2D/param_dists.png')

#_____________________________________________________________________________#
#_____________________________________________________________________________#


