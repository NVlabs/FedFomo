"""
Code adapted from: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/collections.py

Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property 
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an 
express license agreement from NVIDIA CORPORATION is strictly prohibited.

# Source License 
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
# Configuration file for parameters, in conjunction with args.py

import os
import sys
import re
import torch

from federated.args import args
from federated.attr_dict import AttrDict

__C = AttrDict()
cfg_fl = __C
cfg_fl.TASK = 'classification'  # 'semantic_segmentation'

cfg_fl.SEED = 0
cfg_fl.TORCH_SEED = 0
cfg_fl.CUDA = True
cfg_fl.NO_DEBUG = False
cfg_fl.LOG_EPOCH = 5
cfg_fl.REPLICATE = 42

# Change these to where you want save results or models
cfg_fl.RESULTS_DIR = './results'
cfg_fl.MODEL_DIR = './models'

cfg_fl.DATASET = AttrDict()
cfg_fl.DATASET.DATASET_NAME = 'cifar10'
cfg_fl.DATASET.PRECOMPUTED_DIR = './precomputed'

cfg_fl.FEDERATION = AttrDict()
cfg_fl.FEDERATION.FED_AVERAGING = False
cfg_fl.FEDERATION.METHOD = 'fomo'  # How to federate, either 'embeddings' or 'fomo'
cfg_fl.FEDERATION.NUM_DISTRIBUTIONS = 5
cfg_fl.FEDERATION.CLIENTS_PER_DIST = 2   # required if NUM_CLIENTS not specified
cfg_fl.FEDERATION.NUM_CLIENTS = None  # required if CLIENTS_PER_DIST not specified
cfg_fl.FEDERATION.CLUSTERING_METHOD = 'KMeans'  # 'KMeans' or 'AgglomerativeClustering'
cfg_fl.FEDERATION.RANDOM_DISTS = False  # If True, partition datasets randomly across clients
cfg_fl.FEDERATION.CLIENT_RATIO = 1.  # Ratio of clients to participate each round
cfg_fl.FEDERATION.AVG_CLIENT_SIZE = 1  # Number of local clients to average together
cfg_fl.FEDERATION.MODEL_SIZE = cfg_fl.FEDERATION.AVG_CLIENT_SIZE
cfg_fl.FEDERATION.EPOCH = 5  # Number of local rounds between federations
cfg_fl.FEDERATION.LOSS_TEMPERATURE = 10  # Temperature parameter in taking softmax of differences

cfg_fl.FEDERATION.LOCAL_TRAIN_VAL_SIZE = None  # If specified, limit local training size to this

cfg_fl.CLIENT = AttrDict()
cfg_fl.CLIENT.TRAIN_SPLIT = 0.6  # Ratio of local dataset for training local model
cfg_fl.CLIENT.VAL_SPLIT = 0.2    # Ratio of local dataset used for comparing federated models
cfg_fl.CLIENT.TEST_SPLIT = 0.2   # Ratio of local dataset held out for final evaluation
cfg_fl.CLIENT.MANUAL = False

cfg_fl.CLIENT_WEIGHT = AttrDict()
cfg_fl.CLIENT_WEIGHT.WEIGHT_DELTA = 1.  # Delta for approximating step direction
cfg_fl.CLIENT_WEIGHT.NUM_UPDATE_CLIENTS = 8  # Integer
cfg_fl.CLIENT_WEIGHT.EPSILON = 0.3
cfg_fl.CLIENT_WEIGHT.EPSILON_DECAY = 0.05  # Every epoch, decrease by this amount, previously 1e-3
cfg_fl.CLIENT_WEIGHT.METHOD = 'e_greedy'  # 'e_greedy' or 'sub_federations'
cfg_fl.CLIENT_WEIGHT.BASELINE = 'first_model'  # 'first_model', 'last_model', 'model_avg'
cfg_fl.CLIENT_WEIGHT.LEAVE_ONE_OUT = True  # If true, infer weight delta to leave a model out, else use cfg_fl.CLIENT_WEIGHT.WEIGHT_DELTA

cfg_fl.MODEL_WEIGHT = AttrDict()
cfg_fl.MODEL_WEIGHT.UPDATE_POSITIVE_ONLY = True  # Only update with positive loss deltas

# Logging
cfg_fl.LOGX_STDOUT = True

# Following applies to setting up experiments with adversarial clients
# Use this configuration if args.manual_client_setup = True
cfg_fl.CLIENT.POPULATION = [
    {'client_id':  0, 'dist_id': 0, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id':  1, 'dist_id': 0, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id':  2, 'dist_id': 0, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
    {'client_id':  3, 'dist_id': 0, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
    {'client_id':  4, 'dist_id': 1, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id':  5, 'dist_id': 1, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id':  6, 'dist_id': 1, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
    {'client_id':  7, 'dist_id': 1, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
    {'client_id':  8, 'dist_id': 2, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id':  9, 'dist_id': 2, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id': 10, 'dist_id': 2, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
    {'client_id': 11, 'dist_id': 2, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
    {'client_id': 12, 'dist_id': 3, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id': 13, 'dist_id': 3, 'shared_val': True, 'lvr': 0.1, 'adversarial': False},
    {'client_id': 14, 'dist_id': 3, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
    {'client_id': 15, 'dist_id': 3, 'shared_val': False, 'lvr': 1.0, 'adversarial': False},
]


def assert_and_infer_cfg_fl(cfg_fl, args, make_immutable=True, train_mode=True):
    """
    Calls /semantic-segmentation/config.assert_and_infer_cfg and adds additional assertions
    """
    if args.manual_client_setup:
        cfg_fl.CLIENT.MANUAL = args.manual_client_setup

    if cfg_fl.CLIENT.MANUAL:
        print('-------------------------')
        print('> Clients manual settings')
        print('-------------------------')
        for i in cfg_fl.CLIENT.POPULATION:
            print(i)

    if args.replicate:
        cfg_fl.REPLICATE = args.replicate

    if args.seed:
        cfg_fl.SEED = args.seed
        cfg_fl.TORCH_SEED = args.seed

    if args.task:
        cfg_fl.TASK = args.task

    if args.dataset:
        cfg_fl.DATASET.DATASET_NAME = args.dataset

    if args.clients_per_dist:
        cfg_fl.FEDERATION.CLIENTS_PER_DIST = args.clients_per_dist

    if cfg_fl.FEDERATION.CLIENTS_PER_DIST is not None and cfg_fl.FEDERATION.NUM_CLIENTS is None:
        cfg_fl.FEDERATION.NUM_CLIENTS = cfg_fl.FEDERATION.CLIENTS_PER_DIST * cfg_fl.FEDERATION.NUM_DISTRIBUTIONS

    if args.num_clients:
        cfg_fl.FEDERATION.NUM_CLIENTS = args.num_clients

    if args.print_logx:
        cfg_fl.LOGX_STDOUT = True

    if args.num_distributions:
        cfg_fl.FEDERATION.NUM_DISTRIBUTIONS = args.num_distributions

    assertion_num_clients = "Either 'clients_per_dist' or 'num_clients' needs to be specified"
    assert cfg_fl.FEDERATION.CLIENTS_PER_DIST or cfg_fl.FEDERATION.NUM_CLIENTS, assertion_num_clients

#     if args.dist_type:
#         cfg.FEDERATION.DIST_TYPE = args.dist_type

    if args.clustering_method:
        cfg.FEDERATION.CLUSTERING_METHOD = args.clustering_method

    if args.federation_method:
        assert args.federation_method in ['fomo', 'embeddings', 'local', 'fedavg']
        cfg_fl.FEDERATION.METHOD = args.federation_method
        if args.federation_method == 'fedavg':
            cfg_fl.FEDERATION.FED_AVERAGING = True

    if args.random_distributions:
        cfg_fl.FEDERATION.RANDOM_DISTS = args.random_distributions  # True

    if args.federated_averaging:
        cfg_fl.FEDERATION.FED_AVERAGING = True
        cfg_fl.FEDERATION.METHOD = 'fedavg'

    if args.local_train_val_size:
        cfg_fl.FEDERATION.LOCAL_TRAIN_VAL_SIZE = args.local_train_val_size

    if args.federation_epoch:
        cfg_fl.FEDERATION.EPOCH = args.federation_epoch

    if args.num_update_clients:
        cfg_fl.CLIENT_WEIGHT.NUM_UPDATE_CLIENTS = args.num_update_clients

    if args.model_weight_delta:
        cfg_fl.CLIENT_WEIGHT.WEIGHT_DELTA = args.model_weight_delta

    if args.explicit_weight_delta:
        cfg_fl.CLIENT_WEIGHT.WEIGHT_DELTA = args.explicit_weight_delta
        cfg_fl.CLIENT_WEIGHT.LEAVE_ONE_OUT = False

    if args.client_weight_epsilon:
        cfg_fl.CLIENT_WEIGHT.EPSILON = args.client_weight_epsilon

    if args.client_weight_epsilon_decay:
        cfg_fl.CLIENT_WEIGHT.EPSILON_DECAY = args.client_weight_epsilon_decay

    if args.client_weight_method:
        cfg_fl.CLIENT_WEIGHT.METHOD = args.client_weight_method

    if args.update_positive_delta_only:
        cfg_fl.MODEL_WEIGHT.UPDATE_POSITIVE_ONLY = args.update_positive_delta_only

    if args.leave_one_out:
        cfg_fl.CLIENT_WEIGHT.LEAVE_ONE_OUT = args.leave_one_out

    if args.baseline_model:
        cfg_fl.CLIENT_WEIGHT.BASELINE = args.baseline_model

    if args.train_split:
        cfg_fl.CLIENT.TRAIN_SPLIT = args.train_split
        cfg_fl.CLIENT.VAL_SPLIT = 1 - args.train_split
        
    if args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'mnist':
        args.num_classes = 10

    return cfg_fl


# From ADLR semantic-segmentation/config.py
def update_dataset_cfg(cfg_fl, num_classes, ignore_label):
    cfg_fl.immutable(False)
    cfg_fl.DATASET.NUM_CLASSES = num_classes
    cfg_fl.DATASET.IGNORE_LABEL = ignore_label
    # logx.msg('num_classes = {}'.format(num_classes))
    cfg_fl.immutable(True)


# From ADLR semantic-segmentation/config.py
def update_dataset_inst(cfg_fl, dataset_inst):
    cfg_fl.immutable(False)
    cfg_fl.DATASET_INST = dataset_inst
    cfg_fl.immutable(True)
