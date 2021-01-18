# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:22:58 2020

@author: seungjun
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import math


# 가중치 초기화

def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
"""


def weights_init(m): 
    if isinstance(m, nn.Conv2d): 
        nn.init.xavier_normal_(m.weight.data) 
        nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias, 0)
"""


class CNN_model(nn.Module):
    
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(4,32, 3,stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32,48, 3,stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(48,64,3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(64,96,3, stride = 1, padding = 1)
        self.max = nn.MaxPool2d(2,stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(864, 256)
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax()

        

        
    def forward(self, input):
        output = self.conv1(input)
        #print(output.shape)
        output = self.relu(output)
        #print(output.shape)
        output = self.max(output)
        #print(output.shape)
        output = self.conv2(output)
        #print(output.shape)
        output = self.relu(output)
        #print(output.shape)
        output = self.max(output)
        #print(output.shape)
        output = self.dropout(output)
        #print(output.shape)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.max(output)
        output = self.conv4(output)
        output = self.relu(output)
        output = self.max(output)
        output = self.dropout(output)
        output = self.flat(output)
        #print(output.shape)
        output = self.fc1(output)
        
        output = self.relu(output)
        output = self.dropout(output)
        #print(output.shape)
        output = self.fc2(output)
        output = self.softmax(output)
        output = output.view(output.size(0), -1)
        
        
        return output
    
"""
modelss = CNN_model()
modelss.apply(weights_init)

a=[np.random.randint(0,225) for x in range(10*3*48*48)]
a=np.reshape(a,(10,3,48,48))
a=torch.tensor(next(train_gen)[0])

output=modelss(a.float())


"""