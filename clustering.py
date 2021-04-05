#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:16:19 2019

@author: winston
"""
from scipy.io import loadmat
import numpy as np
import torch.utils.data as data
import faiss
import time


def pil_loader(path, feat_norm_mean, feat_norm_std):
    """Loads an 6373-func image.
    Args:
        path (str): path to image file
        feat_norm_mean (arry): z-norm mean parameters
        feat_norm_std (arry): z-norm std parameters
    Returns:
        z-normalized 6373-func img
    """    
    img = loadmat(path.replace('.wav','.mat'))['Audio_data']
    # Z-normalization 
    img = (img-feat_norm_mean)/feat_norm_std
    img = img.reshape(-1)
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    img[np.isnan(img)]=0
    img[img>3]=3
    img[img<-3]=-3     
    return img

class ReassignedDataset(data.Dataset):
    """A dataset where the new labels are given in argument.
    Args:
        image_indexes (list): list of 6373-func-img indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of paths/labels (tuples) to audio files
    """

    def __init__(self, image_indexes, pseudolabels, dataset):
        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']         
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path, label = dataset[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, label, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel, emo_label) where pseudolabel is the cluster of index datapoint
        """
        path, label, pseudolabel = self.imgs[index]
        img = pil_loader(path, self.Feat_mean, self.Feat_std)
        return img, pseudolabel, label

    def __len__(self):
        return len(self.imgs)

class ReassignedDataset_unlabel(data.Dataset):
    """A dataset where the new labels are given in argument.
    Args:
        image_indexes (list): list of 6373-func-img indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of paths/labels (tuples) to audio files
    """

    def __init__(self, image_indexes, pseudolabels, dataset):
        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']          
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = pil_loader(path, self.Feat_mean, self.Feat_std)
        return img, pseudolabel
    
    def __len__(self):
        return len(self.imgs)

def cluster_assign(images_lists, dataset, dataset_type):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    if dataset_type == 'supervised':
        return ReassignedDataset(image_indexes, pseudolabels, dataset)
    elif dataset_type == 'non-supervised':
        return ReassignedDataset_unlabel(image_indexes, pseudolabels, dataset)

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]

class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()
        xb = data
        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss

