# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""
Functions for setting up optimizers
"""

import torch
from torch import optim


def get_local_optimizer(args, net):
    """
    Returns optimizer (Adam or SGD)
    - args: federated.args
    - net (torch.nn): model to optimizer
    """
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=args.amsgrad)
    else:
        raise ValueError('Not a valid optimizer')

    if args.learning_rate_decay > 0:
        # Can implement more complicated LR scheduler here with net.parameters() if desired
        scheduler = None
    else:
        scheduler = None
    return optimizer, scheduler
