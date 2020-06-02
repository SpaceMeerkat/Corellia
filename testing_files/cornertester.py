#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:40:54 2020

@author: james
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
import corner

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
    
p = temp_pred
print(p.shape)
p = p[:,1:]
_labels = ['i ($^{\circ}$)','$a_{I}$','$V_{scale} \, (km\,s^{-1})$','$V_{max} \, sin(i) \, (km\,s^{-1})$']

figure = corner.corner(p,bins=20,labels=_labels)   
plt.tight_layout()
plt.savefig('/home/corona/c1307135/Semantic_ML/Corellia/Test_images/2D/corner_single.png')
#_____________________________________________________________________________#
#_____________________________________________________________________________#