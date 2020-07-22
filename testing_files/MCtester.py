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
import pandas as pd

#_____________________________________________________________________________#
#_____________________________________________________________________________#

model_path = '/home/corona/c1307135/Semantic_ML/Corellia/models/'

#_____________________________________________________________________________#
#_____________________________________________________________________________#

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train(True)

model = CAE()
model.load_state_dict(torch.load(model_path+'semantic_AE_test_.pt'))
model = model.cpu()
model.train(False)
model.apply(apply_dropout)
print("Model cast to CPU")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

index = np.arange(0,100,1)
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
a = target[:,-3]

mom0s, mom1s = batch[:,0,:,:].unsqueeze(1), batch[:,1,:,:].unsqueeze(1)

print("Test data created")

#_____________________________________________________________________________#
#_____________________________________________________________________________#

predictions = []
errors = []
for j in range(batch.shape[0]): 
    temp_pred = []
    for _ in range(100):
        prediction1 = model.test_encode(mom0s[j].unsqueeze(0),mom1s[j].unsqueeze(0),
                                        pos[j].unsqueeze(0))#,a[j].unsqueeze(0))
        prediction1 = prediction1.detach().numpy()
        temp_pred.append(prediction1)
    temp_pred = np.vstack(temp_pred)
    mean_pred = np.mean(temp_pred,0)
    predictions.append(mean_pred)
    errors.append(np.sum(np.abs(temp_pred-mean_pred[None,:]),0)/len(temp_pred))
        
print("Testing data complete")

predictions = np.vstack(predictions)
errors = np.vstack(errors)

dfp = pd.DataFrame(predictions)
dft = pd.DataFrame(np.vstack(target))
dfe = pd.DataFrame(errors)

df = pd.concat([dfp,dft,dfe],axis=1)
df.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

df.to_pickle('/home/corona/c1307135/Semantic_ML/Corellia/pickle_files/'+str(np.random.uniform(0,1,1)[0])+'.pkl')

#_____________________________________________________________________________#
#_____________________________________________________________________________#


