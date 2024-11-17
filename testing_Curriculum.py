#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import torch
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from utils import getPaths_attri, evaluation_metrics
from FCNet import DenseNet
import argparse
torch.manual_seed(1)



def model_evaluation(root_dir, paths, gt_labels):
    Pred_Rsl, GT_Label = [], []
    for i in tqdm(range(len(paths))):
        # Loading data
        data = loadmat(root_dir+paths[i].replace('.wav','.mat'))['Audio_data']
        
        # Z-normalization
        data = (data - Feat_mean) / Feat_std
        
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3
        
        # models flow
        with torch.no_grad():
            data = torch.from_numpy(data).float().cuda()
            pred = model(data)
            pred = torch.squeeze(pred, dim=1)
            
        # output
        GT_Label.append(gt_labels[i])
        Pred_Rsl.append(pred.data.cpu().numpy()[0])
    Pred_Rsl, GT_Label = np.array(Pred_Rsl), np.array(GT_Label)
    
    # Regression Task => De-Normalize Target and Prediction
    Pred_Rsl = (Label_std * Pred_Rsl) + Label_mean
    
    # Output Predict Reulst
    pred_CCC_Rsl = evaluation_metrics(GT_Label, Pred_Rsl)[0]
    
    return pred_CCC_Rsl
###############################################################################



argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
args = vars(argparse.parse_args())

# Dirs & Parameters
root_dir = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Features/OpenSmile_func_IS13ComParE/feat_mat/'
label_dir = '/YOUR/PATH/TO/MSP-PODCAST-Publish-1.8/Labels/labels_concensus.csv'
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
emo_attr = args['emo_attr']

# loading test set Paths & GTs
_paths, _gt_labels = getPaths_attri(label_dir, split_set='Test1', emo_attr=emo_attr)

# de-normalize parameters
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
    
# load the trained model
MODEL_PATH = './Models/SimpleEmoRegressor_DeepMI@_'+emo_attr+'.pt.tar'
model = DenseNet(input_dim=6373, output_dim=1)
model.load_state_dict(torch.load(MODEL_PATH))
model.cuda().eval()

# testing process & recognition results
recog_rsl = model_evaluation(root_dir, _paths, _gt_labels)
print('Recognition performance for the MSP-Podcast Test set: ')
print(emo_attr+' (CCC): '+str(recog_rsl))

