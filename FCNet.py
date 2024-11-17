#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import torch
import torch.nn as nn


class DenseNet(torch.nn.Module):
    """Simple DenseNet emotional regressor."""
    
    def __init__(self, input_dim, output_dim):
        super(DenseNet, self).__init__()
        
        # STL-Dense-layers
        self.dense_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
    def forward(self, x):
        y = self.dense_net(x)
        return y
