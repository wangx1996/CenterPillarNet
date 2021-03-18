"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import os
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("src"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataset import KittiDataset
from data_process.transformation import OneOf, Random_Rotation, Random_Scaling
from collections import defaultdict

def merge_batch(batchlist):
    batchdata_merged = defaultdict(list)
    batchtarget_merged = defaultdict(list)

    for batchdata in batchlist:
        metadatas, targets = batchdata

        for k, v in metadatas.items():
            batchdata_merged[k].append(v)

        for k, v in targets.items():
            #print(type(v))
            batchtarget_merged[k].append(v)

    ret = {}
    for key, elems in batchdata_merged.items():
        if key in ['voxels', 'num_points']:
            ret[key] = np.concatenate(elems, axis=0)

        elif key == 'coors':
            coors = []
            for i, coor in enumerate(elems):
                #print('coor {}'.format(coor.shape))
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
                #print('coor_pad {}'.format(coor_pad.shape))
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = elems
    tar = {}

    for key, elems in batchtarget_merged.items():
            tar[key] = torch.from_numpy(np.array(elems))
            #print(tar[key].shape)

    return ret, tar

def merge_batch_test(batchlist):
    batchdata_merged = defaultdict(list)
    batchtarget_merged = defaultdict(list)

    for batchdata in batchlist:
        metadatas = batchdata

        for k, v in metadatas.items():
            batchdata_merged[k].append(v)


    ret = {}
    for key, elems in batchdata_merged.items():
        if key in ['voxels', 'num_points']:
            ret[key] = np.concatenate(elems, axis=0)

        elif key == 'coors':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = elems

    return ret

def create_train_dataloader(configs, voxel_generator):
    """Create dataloader for training"""
    train_lidar_aug = OneOf([
        Random_Rotation(limit_angle=np.pi / 4, p=1.0),
        Random_Scaling(scaling_range=(0.95, 1.05), p=1.0),
    ], p=0.66)
    train_dataset = KittiDataset(configs, voxel_generator, mode='train', lidar_aug=train_lidar_aug, hflip_prob=configs.hflip_prob,
                                 num_samples=configs.num_samples)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler, collate_fn=merge_batch)

    return train_dataloader, train_sampler


def create_val_dataloader(configs, voxel_generator):
    """Create dataloader for validation"""
    val_sampler = None
    val_dataset = KittiDataset(configs, voxel_generator, mode='val', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler, collate_fn=merge_batch)

    return val_dataloader


def create_test_dataloader(configs, voxel_generator):
    """Create dataloader for testing phase"""
    print('xxxx')
    test_dataset = KittiDataset(configs, voxel_generator, mode='test', lidar_aug=None, hflip_prob=0., num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler, collate_fn=merge_batch_test)

    return test_dataloader