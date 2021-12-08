# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""
Utility functions of various function and utility
"""
import sys
import os
import numpy as np
import torch

from collections import Counter

from federated.configs import cfg_fl as cfg


def print_header(stdout, style=None):
    if style is None:
        print('-' * len(stdout))
        print(stdout)
        print('-' * len(stdout))
    elif style == 'bottom':
        print(stdout)
        print('-' * len(stdout))
    elif style == 'top':
        print('-' * len(stdout))
        print(stdout)


def print_debug(stdout, prefix=''):
    if cfg.NO_DEBUG:
        return
    print(f'DEBUG - {prefix}: {stdout}')


# Logging
class Logger(object):
    def __init__(self, args):
        """
        Logging class to save print output to save path specified below
        - For credit see https://stackoverflow.com/a/14906787

        Setup in main as follows:
        from federated.utils import Logger
        sys.stdout = Logger()
        """
        save_path = f'./logs/replicate-{args.replicate}'
        try:
            os.mkdir(save_path)
        except FileExistsError:
            pass

        self.terminal = sys.stdout  # Actually set up logging
        log_path = os.path.join(save_path, f'{args.experiment_name}.log')
        print(f'> Logging results to {log_path}')
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass


def summarize_args(args, as_script=False):
    """
    Display all experiment arguments
    If as_script == True, will print text for a runnable script to 
    reproduce experiment based on current configurations
    """
    arg_list = []
    for arg in vars(args):
        arg_list.append(f'--{arg} {getattr(args, arg)}')
    print('Experiment arguments')
    if as_script:
        script_text = ' '.join(arg_list)
        print(f'python main.py {script_text}')
    else:
        for arg_text in arg_list:
            print(arg_text)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_emd(targets_1, targets_2):
    """Calculates Earth Mover's Distance between two array-like objects (dataset labels)"""
    total_targets = []
    total_targets.extend(list(np.unique(targets_1)))
    total_targets.extend(list(np.unique(targets_2)))

    emd = 0

    counts_1 = Counter(targets_1)
    counts_2 = Counter(targets_2)

    size_1 = len(targets_1)
    size_2 = len(targets_2)

    for t in counts_1:
        count_2 = counts_2[t] if t in counts_2 else 0
        emd += np.abs((counts_1[t] / size_1) - (count_2 / size_2))

    for t in counts_2:
        count_1 = counts_1[t] if t in counts_1 else 0
        emd += np.abs((counts_2[t] / size_2) - (count_1 / size_1))

    return emd


def compute_parameter_difference(model_a, model_b, norm='l2'):
    """
    Compute difference in two model parameters
    """
    if norm == 'l1':
        total_diff = 0.
        total_diff_l2 = 0.
        # Compute L1-norm, i.e. ||w_a - w_b||_1
        for w_a, w_b in zip(model_a.parameters(), model_b.parameters()):
            total_diff += (w_a - w_b).norm(1).item()
            total_diff_l2 += torch.pow((w_a - w_b).norm(2), 2).item()

        return total_diff

    elif norm == 'l2_root':
        total_diff = 0.
        for w_a, w_b in zip(model_a.parameters(), model_b.parameters()):
            total_diff += (w_a - w_b).norm(2).item()
        return total_diff

    total_diff = 0.
    model_a_params = []
    for p in model_a.parameters():
        model_a_params.append(p.detach().cpu().numpy().astype(np.float64))

    for ix, p in enumerate(model_b.parameters()):
        p_np = p.detach().cpu().numpy().astype(np.float64)
        diff = model_a_params[ix] - p_np
        scalar_diff = np.sum(diff ** 2)
        total_diff += scalar_diff
    # Can be vectorized as
    # np.sum(np.power(model_a.parameters().detach().cpu().numpy() - 
    #                 model_a.parameters().detach().cpu().numpy(), 2))
    return total_diff  # Returns distance^2 between two model parameters
        
    
