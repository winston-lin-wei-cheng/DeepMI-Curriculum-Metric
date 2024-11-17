#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)


__all__ = [ 'DenseResNet', 'resnet']

class residual_block(nn.Module):
    """Base residual block."""
    
    def __init__(self, nodes_num):
        super(residual_block, self).__init__()
        self.dense1 = nn.Linear(nodes_num, nodes_num)
        self.dense2 = nn.Linear(nodes_num, nodes_num)

    def forward(self,x):
        residual = x
        out = F.relu(self.dense1(x))
        out = F.dropout(out, p=0.3)
        out = self.dense2(out)
        out += residual    
        out = F.relu(out)
        return out


class DenseResNet(nn.Module):
    """Backbone DeepEmoCluster model in the paper."""

    def __init__(self, input_dim, num_classes):
        super(DenseResNet, self).__init__()
        # ResNet Deep Feature Representations
        self.dnn_features = nn.Sequential(
            # resnet-block1
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.3),
            residual_block(1024),
            residual_block(1024),
            # resnet-block2
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            residual_block(512),
            residual_block(512),
            # resnet-block3
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            residual_block(256),
            residual_block(256),
        )
        
        # Unsupervised Cluster Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )              
        self.top_layer_class = nn.Linear(256, num_classes)
        
        # Supervised STL-Emotional Regressors
        self.emo_regressor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )              

    def forward(self, x):
        # shared ResNet feature extraction model
        x = self.dnn_features(x)
        # for deep-cluster classification
        x_class = self.classifier(x)
        if self.top_layer_class:
            x_class = self.top_layer_class(x_class)
        # for emotion STL regression
        x_attri = self.emo_regressor(x)
        return x_class, x_attri


def resnet(inp, out):
    model = DenseResNet(input_dim=inp, num_classes=out)
    return model

