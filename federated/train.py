# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Functions for training and evaluating models
"""
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from federated_datasets import load_federated_dataset
from federated.args import args
from federated.configs import cfg_fl as cfg

if args.enable_dp:
    from opacus import PrivacyEngine


def train(dataset, net, optim, epoch, local=True, criterion=None, 
          device=None, client=None, sampler=None, args=args):
    """
    General purpose local training function. Details for specific tasks implemented in methods below
    
    For FedFomo, we only consider the classification task (although there is boilerplate to support semantic segmentation).
    
    Args:
    - dataset (torch.utils.data.Dataset): Dataset
    - net (torch.nn.Module): Neural net architecture
    - optim (torch.optim.optimizer): Training ptimizer
    - epoch (int): Communication round
    - local (bool): Whether training local or global models. Should be True by default.
    - criterion (torch.nn.criterion): Criterion / loss function
    - device: Which device to train on
    - client: Which client's model to train
    - sampler: Alternative data sampler
    - args: Additional experiment arguments. Passed through federated.args by default.
    """
    net.train()
    if device is not None:
        net.to(device)

    if args.enable_dp:
        try:
            if args.federation_method == 'local' and epoch > 0:
                pass
            else:
                args.clipping = {'clip_per_layer': False, 'enable_state': True}

                bs = args.bs_trn_v if args.virtual_batch_rate else args.bs_trn
                privacy_engine = PrivacyEngine(net, batch_size=bs * args.n_accumulation_steps, sample_size=len(dataset),
                                               alphas=[1 + x/10. for x in range(1, 100)] + list(range(12, 64)),  # not sure about this
                                               noise_multiplier=args.sigma,
                                               max_grad_norm=args.max_per_sample_grad_norm,
                                               secure_rng=args.secure_rng,
                                               **args.clipping)
                privacy_engine.attach(optim)
        except Exception as e:
            # For future work with Opacus, tie in the privacy engine with the optimizer while reducing GPU memory limits
            pass
    if local and cfg.TASK == 'classification':
        dataloader = load_federated_dataset(dataset=dataset, train=True)
        return class_train(dataloader, net, optim, epoch, criterion, device, client)
    elif local and cfg.TASK == 'semantic_segmentation':
        dataloader = load_federated_dataset(dataset=dataset, train=True)
        return seg_train(dataloader, net, optim, epoch, device=device, client=client)
    else:
        raise NotImplementedError


def eval(dataset, net, epoch, local, criterion, device, client=None, 
         sampler=None, client_text='', optim=None, ensemble=False, args=args):
    """
    Evaluate a model on a given dataset
    """
    net.eval()
    if device is not None:
        net.to(device)

    if args.enable_dp:
        try:
            if args.federation_method == 'local' and epoch > 0:
                pass
            else:
                args.clipping = {'clip_per_layer': False, 'enable_state': True}
                bs = args.bs_val_v if args.virtual_batch_rate else args.bs_val
                alphas = ([1 + x/10. for x in range(1, 100)] + 
                          list(range(12, 64)))
                privacy_engine = PrivacyEngine(net, batch_size=bs * args.n_accumulation_steps, 
                                               sample_size=len(dataset),
                                               alphas=alphas,
                                               noise_multiplier=args.sigma,
                                               max_grad_norm=args.max_per_sample_grad_norm,
                                               secure_rng=args.secure_rng,
                                               **args.clipping)
                privacy_engine.attach(optim)
        except Exception as e:
            # For future work with Opacus, tie in the privacy engine 
            # with the optimizer while reducing GPU memory limits
            pass
    if local and cfg.TASK == 'classification':
        dataloader = load_federated_dataset(dataset=dataset, train=False)
        return class_eval(dataloader, net, epoch, criterion, device, client, 
                          client_text, ensemble=ensemble, args=args)
    elif local and cfg.TASK == 'semantic_segmentation':
        dataloader = load_federated_dataset(dataset=dataset, train=False)
        val_loss, iou_acc = seg_eval(dataloader, net, criterion, optim, epoch, device=device)
        return val_loss, iou_acc
    else:
        raise NotImplementedError


def class_train(dataloader, net, optim, epoch, criterion,
                device, client=None, return_stdout=True):
    """
    Training function for classification
    """
    total_loss = 0
    total_correct = 0
    total_train = 0
    device = device if device is not None else torch.device('cpu')

    start_time = time.time()
    if args.debugging:
        num_images = 0
        dict_unique = {}

    all_batch_loss = []
    for batch_ix, (inputs, targets) in enumerate(tqdm(dataloader, leave=False, desc='Training...')):
        inputs = inputs.to(device)
        try:
            if client.adversarial:  # For now just set the target to 0
                targets = torch.zeros(targets.shape, dtype=torch.long)
        except AttributeError:
            pass
        targets = targets.to(device)

        net.zero_grad() 
        outputs = net(inputs)

        # FedProx - factor in the mu's too
        if args.fedprox and client is not None:  
            prox_term = 0.
            # Compute ||w - w_t||^2
            client.first_model.to(client.device)
            for w, w_t in zip(net.parameters(),
                              client.first_model.parameters()):
                prox_term += torch.pow((w - w_t).norm(2), 2)

            loss = (criterion(outputs, targets) + 
                    (args.fedprox_mu / 2.) * prox_term)
        else:
            loss = criterion(outputs, targets)

        loss.backward()

        if args.enable_dp and args.virtual_batch_rate is not None:
            if (((batch_ix + 1) % args.virtual_batch_rate == 0) or 
                ((batch_ix + 1) == len(dataloader))):
                optim.step()
                optim.zero_grad()
            else:
                optim.virtual_step()
        else:
            optim.step()

        total_loss += loss.item()
        all_batch_loss.append(loss.item())

        predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        targets = targets.detach().cpu().numpy()
        total_train += len(targets)  # .size(0)
        total_correct += (predictions == targets).sum().item()

        # Clear up memory
        outputs = outputs.detach().cpu().numpy()
        del inputs; del targets; del outputs;
        torch.cuda.empty_cache()

    end_time = time.time()
    epoch_loss = sum(all_batch_loss)/len(all_batch_loss)  
    epoch_text = '' if epoch is None else f'Epoch: {epoch:<2} |'
    client_text = '' if client is None else f'Client {client.id:<2} | Dist {client.dist_id:<2} |'
    train_text = f'{epoch_text} {client_text} Train accuracy: {total_correct / total_train:<.4f} | Train loss: {epoch_loss:<.4f} | Batch train time: {(end_time - start_time) / len(dataloader) * 1000:<.2f} ms'

    # Clear up GPU memory
    for name, p in net.named_parameters():
        p.detach()
    net.to('cpu')

    if return_stdout:
        return epoch_loss, total_correct / total_train, train_text
    else:
        return epoch_loss, total_correct / total_train, None


def class_eval(dataloader, net, epoch, criterion, device, client, client_text='', 
               ensemble=False, args=args, return_stdout=True):
    """
    Evaluation function for classification
    """
    total_loss = 0
    total_correct = 0
    total_eval = 0
    device = device if device is not None else torch.device('cpu')

    disable_tqdm = False
    start_time = time.time()

    net.to(device)  # Align GPUs

    all_probs = []
    all_targets = []

    all_batch_loss = []

    with torch.no_grad():
        for batch_ix, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device) 
            targets = targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            all_batch_loss.append(loss.item())

            try:
                predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            except Exception as e:
                print(np.argmax(outputs.detach().cpu().numpy(), axis=1).shape)
                raise e
            targets = targets.detach().cpu().numpy()
            total_eval += len(targets)  # .size(0)
            total_correct += (predictions == targets).sum().item()

            if ensemble:
                log_probs = F.log_softmax(outputs, dim=1)
                all_probs.append(log_probs)
                all_targets.extend(list(targets.flatten()))

            # Clear up memory
            outputs = outputs.detach().cpu().numpy()
            del inputs; del targets; del outputs;
            torch.cuda.empty_cache()

    end_time = time.time()

    for name, p in net.named_parameters():
        p.detach()
    net.to('cpu')  # Clear up GPU memory

    epoch_loss = sum(all_batch_loss)/len(all_batch_loss)

    if client is None and ensemble is False:
        return epoch_loss, total_correct / total_eval, None

    epoch_text = '' if epoch is None else f'Epoch: {epoch:<2} |'
    if client_text == '':
        client_text = '' if client is None else f'Client {client.id:<2} | Dist {client.dist_id:<2} |'
    eval_text = f'{epoch_text} {client_text} Eval  accuracy: {total_correct / total_eval:<.4f} | Eval  loss: {epoch_loss:4<.4f} | Batch eval  time: {(end_time - start_time) / len(dataloader) * 1000:<.2f} ms'

    if ensemble:
        return epoch_loss, total_correct / total_eval, eval_text, torch.cat(all_probs), all_targets

    if return_stdout:
        return epoch_loss, total_correct / total_eval, eval_text
    else:
        return epoch_loss, total_correct / total_eval, None


def compute_loss_delta_by_class(dataset, baseline_net, comparison_net, epoch, criterion, 
                                device, args=args, return_stdout=True):
    """
    Compute alternative loss difference between baseline 
    and comparison models, factoring in the individual classes
    in the target validation dataset
    """
    # total_loss = 0
    total_loss_delta = 0.
    total_positives = 0

    total_correct_b = 0
    total_correct_c = 0
    # Loss in total
    total_reduced_loss_b = 0 
    total_reduced_loss_c = 0

    total_eval = 0
    device = device if device is not None else torch.device('cpu')

    criterion.reduce = False
    criterion.reduction = 'none'

    disable_tqdm = False

    dataloader = load_federated_dataset(dataset=dataset, train=False)

    start_time = time.time()
    with torch.no_grad():
        for batch_ix, (inputs, targets) in enumerate(tqdm(dataloader, disable=disable_tqdm, leave=False)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs_b = baseline_net(inputs)
            outputs_c = comparison_net(inputs)

            loss_b = criterion(outputs_b, targets)
            loss_c = criterion(outputs_c, targets)

            total_reduced_loss_b += loss_b.sum().item()
            total_reduced_loss_c += loss_c.sum().item()

            total_eval += targets.size(0)

            _, predictions_b = torch.max(outputs_b.data, 1)
            _, predictions_c = torch.max(outputs_c.data, 1)

            correct_mask_b = (predictions_b == targets)
            correct_mask_c = (predictions_c == targets)

            # Only consider where comparison model got correct
            masked_loss_b = correct_mask_c * loss_b  # correct_mask_b
            masked_loss_c = correct_mask_c * loss_c

            # Will be positive if comparison loss is lower than baseline loss
            loss_delta = masked_loss_b - masked_loss_c
            # loss_delta = loss_b - loss_c  # batch_size
            loss_delta[loss_delta < 0] = 0  # Mask out negatives

            total_loss_delta += torch.sum(loss_delta).item() # Add loss delta, i.e. how much better comparison model did
            # Number of datapoints where comparison model had lower loss than baseline model
            total_positives += (loss_delta != 0).sum().item()

            total_correct_b += (predictions_b == targets).sum().item()
            total_correct_c += (predictions_c == targets).sum().item()

    end_time = time.time()

    loss_delta = (total_loss_delta / total_eval) * (total_positives / total_eval)

    total_reduced_loss_b = total_reduced_loss_b / total_eval
    total_reduced_loss_c = total_reduced_loss_c / total_eval

    reduced_loss_delta = total_reduced_loss_b - total_reduced_loss_c

    text = f'Epoch {epoch} | Baseline acc: {total_correct_b / total_eval:<.4f} | Comparison acc: {total_correct_c / total_eval:<.4f} | Class loss delta: {loss_delta:<.4f} | Baseline loss: {total_reduced_loss_b:<.4f} | Comparison loss: {total_reduced_loss_c:<.4f} | Fraction: {(total_positives / total_eval):<.4f} | Batch eval time: {(end_time - start_time) / len(dataloader):<.4f}'
    if return_stdout:
        return loss_delta, reduced_loss_delta, text
    else:
        return loss_delta, reduced_loss_delta
