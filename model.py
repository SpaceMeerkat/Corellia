#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:15:02 2019

@author: SpaceMeerkat
"""

import torch

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class CovNet(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.conv1 = torch.nn.Conv1d(1,32,3,padding=1)
        self.conv2 = torch.nn.Conv1d(32,64,3,padding=1)
        self.conv3 = torch.nn.Conv1d(64,128,3,padding=1)
        self.mp = torch.nn.AvgPool1d(2)
        self.mp2 = torch.nn.AvgPool1d(3)
        self.lc1 = torch.nn.Linear(128*275,2048)
        self.lc2 = torch.nn.Linear(2048,1024)
        self.lc3 = torch.nn.Linear(1024,512)
        self.lc4 = torch.nn.Linear(512,256)
        self.lc5 = torch.nn.Linear(261,self.nodes) #257,self.nodes
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x):
        x = x.view(int(x.size()[0]),1,-1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.lc4(x)
        x = self.relu(x)
        return x
        
    def forward(self,x,y):
        output = self.encoder(x)
        y = y.unsqueeze(1)
        output = torch.cat((output,y),dim=1)
        output = self.lc5(output)
        return output  