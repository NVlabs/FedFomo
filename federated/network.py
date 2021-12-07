"""
Functions and classes for local and global model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from federated.configs import cfg_fl as cfg
from federated.utils import print_debug


def get_net(args, criterion=None):
    """
    Returns network architecture given argparse args
    """
    if cfg.TASK == 'classification' or args.task == 'classification':
        if args.arch == 'base_cnn':
            model = BaseConvNet()
        elif args.arch == 'base_cnn_224':
            model = BigBaseConvNet()
        elif args.arch == 'tf_cnn':
            model = TFConvNet()
        else:
            torchvision_models = importlib.import_module('torchvision.models')
            model = getattr(torchvision_models, args.arch)(pretrained=False)
        adjust_model_layers(model, args)
        return model
    else:
        raise NotImplementedError


def adjust_model_layers(model, args):
    """
    Perform network layer adjustments depending on dataset and architecture
    """
    if args.arch == 'base_cnn':
        if 'mnist' in cfg.DATASET.DATASET_NAME:
            model.conv1 = nn.Conv2d(1, 6, 5)
        model.fc3 = torch.nn.Linear(84, args.num_classes, bias=True)
    elif args.arch == 'base_cnn_224':
        if 'mnist' in cfg.DATASET.DATASET_NAME:
            model.conv1 = nn.Conv2d(1, 64, 5)
        model.fc3 = torch.nn.Linear(192, args.num_classes, bias=True)
    elif args.arch == 'tf_cnn':
        if 'mnist' in cfg.DATASET.DATASET_NAME:
            model.conv1 = nn.Conv2d(1, 32, 3)
            # model.fc1 = nn.Linear(64 * 3 * 3, 64)
        model.fc2 = torch.nn.Linear(64, args.num_classes, bias=True)
    else:
        raise NotImplementedError


class BaseConvNet(nn.Module):
    """
    Network architecture in the PyTorch image classification tutorial
    """
    def __init__(self):
        super(BaseConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def embed(self, x, layer=2):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        if layer == 1:
            return self.fc1(x)
        x = F.relu(self.fc1(x))
        if layer == 2:
            return self.fc2(x)
        x = F.relu(self.fc2(x))
        if layer == 3:
            return self.fc3(x)


class BigBaseConvNet(nn.Module):
    def __init__(self):
        super(BigBaseConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 53 * 53, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def embed(self, x, layer=3):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,  64 * 53 * 53)
        if layer == 1:
            return self.fc1(x)
        x = F.relu(self.fc1(x))
        if layer == 2:
            return self.fc2(x)
        x = F.relu(self.fc2(x))
        if layer == 3:
            return self.fc3(x)


class TFConvNet(nn.Module):
    """
    Network architecture in the Tensorflow image classification tutorial
    """
    def __init__(self):
        super(TFConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def embed(self, x, layer=3):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        if layer == 1:
            return self.fc1(x)
        x = F.relu(self.fc1(x))
        if layer == 2:
            return self.fc2(x)
