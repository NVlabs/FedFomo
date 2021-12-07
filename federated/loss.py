"""
Local and global loss functions
"""

import torch.nn as nn

from federated.configs import cfg_fl as cfg


def get_local_loss():
    """
    Returns training and validation criterion based on task specified in federated/configs.py
    """
    if cfg.TASK == 'classification':
        return nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    elif cfg.TASK == 'semantic_segmentation':
        raise NotImplementedError
    else:
        raise ValueError('Suitable loss not specified')
