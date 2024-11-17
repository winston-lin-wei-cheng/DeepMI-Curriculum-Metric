#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import torch
import os
import numpy as np
import json
from scipy.io import loadmat
from utils import getPaths_attri, load_model
from tqdm import tqdm
import argparse


def attri_to_level(score):
    """Manually quantifies original continuous emotion scores into E=6-levels."""
    emo_level = np.zeros(len(score))
    emo_level[np.where(score>=6)[0]]=5
    emo_level[np.where((score>=5)&(score<6))[0]]=4
    emo_level[np.where((score>=4)&(score<5))[0]]=3
    emo_level[np.where((score>=3)&(score<4))[0]]=2
    emo_level[np.where((score>=2)&(score<3))[0]]=1
    emo_level[np.where(score<2)[0]]=0
    return emo_level.astype('int')


def compute_jointprob_matrix(X, Y, num_clusters):
    """Compute P(X,Y) table for MI."""
    domain_x = num_clusters # X: acoustic clusters
    domain_y = 6            # Y: emotional levels
    joint_prob_table = np.zeros((domain_y, domain_x))
    for i in range(domain_x):
        for j in range(domain_y):
            joint_freq = len(np.where((X==i)&(Y==j))[0])
            joint_prob_table[j, i] = joint_freq/len(X)
    return joint_prob_table 
##############################################################################



argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-nc", "--num_clusters", required=True)
args = vars(argparse.parse_args())

# Dirs & Parameters
root_dir = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Features/OpenSmile_func_IS13ComParE/feat_mat/'
label_dir = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv'
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']
num_clusters = int(args['num_clusters'])

# checking/creating output directory
if not os.path.exists('./curriculum_metric/'):
    os.mkdir('./curriculum_metric/')

# loading de-normalize parameters
Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']
if emo_attr == 'Act':
    Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Dom':
    Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Val':
    Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]    

# loading entire train set (the curriculum metric is only extracted for the training set)
_paths, _gt_labels = getPaths_attri(label_dir, split_set='Train', emo_attr=emo_attr)

# loading the trained DeepEmoCluster model
MODEL_PATH = './trained_SSLmodels_v1.8/ResNetDeepEmoCluster_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+str(num_clusters)+'clusters_'+emo_attr+'.pth.tar'
model = load_model(MODEL_PATH, num_clusters)
model.cuda().eval()

# compute DeepMI-metric for each file in the training set
FileName = []
X, Y, Y_p = [], [], []
for i in tqdm(range(len(_paths))):
    # Loading data
    data = loadmat(root_dir + _paths[i].replace('.wav','.mat'))['Audio_data']
    
    # Z-normalization
    data = (data - Feat_mean) / Feat_std
    data = data.reshape(-1)
    
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]= -3
    
    # model flow
    with torch.no_grad():
        data = torch.from_numpy(data).view(1, -1).cuda().float()
        pred_cluster, pred_attri = model(data)
        pred_cluster = torch.argmax(pred_cluster)
        pred_attri = torch.squeeze(pred_attri)
        pred_attri = (Label_std * pred_attri) + Label_mean # de-norm
    
    # collect outputs
    FileName.append(str(_paths[i]))
    X.append(pred_cluster.data.cpu().numpy())
    Y.append(_gt_labels[i])
    Y_p.append(pred_attri.data.cpu().numpy())

# weighting parameter alpha for I(X;Y) and I(Y';Y)
alpha = 0.5 if emo_attr == 'Val' else 0

# obtain discrete rvs
X = np.array(X)    
Y = attri_to_level(np.array(Y))
Y_p = attri_to_level(np.array(Y_p))

# obtain JointProb(X,Y), MarginProb(X) and MarginProb(Y)
p_x_y = compute_jointprob_matrix(X, Y, num_clusters)
p_x = np.sum(p_x_y, axis=0)
p_y = np.sum(p_x_y, axis=1)

# obtain JointProb(Y',Y), MarginProb(Y') and MarginProb(Y)
p_yp_y = compute_jointprob_matrix(Y_p, Y, 6)
p_yp = np.sum(p_yp_y, axis=0)
p_y = np.sum(p_yp_y, axis=1)

# calculate DeepMI values
DeepMI = []
for i in range(len(X)):
    # definition formula of mutual information
    mi_xy = p_x_y[Y[i], X[i]] * np.log(p_x_y[Y[i], X[i]] / (p_x[X[i]] * p_y[Y[i]]))
    mi_ypy = p_yp_y[Y[i], Y_p[i]] * np.log(p_yp_y[Y[i], Y_p[i]] / (p_yp[Y_p[i]] * p_y[Y[i]]))
    mi = (1 - alpha) * mi_xy + alpha * mi_ypy
    mi = np.round(mi, decimals=5) # rounded value
    DeepMI.append(mi)
assert sum(DeepMI) >= 0 # MI should be non-negative
print('Global MI: '+str(sum(DeepMI)))

# output JSON file (dict) for the DeepMI-metric with respect to filenames
curriculum_metric = {}
for i in range(len(FileName)):
    curriculum_metric[FileName[i]] = DeepMI[i]
with open('./curriculum_metric/DeepMI_'+str(num_clusters)+'clusters@'+'_'+emo_attr+'.json', 'w') as fp:
    json.dump(curriculum_metric, fp)

