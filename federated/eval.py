# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Functions for evaluating federated models
"""
import os
import pickle
import torch

from scipy import stats
from tqdm import tqdm

from torch.utils.data import ConcatDataset, random_split

from federated.args import args
from federated.configs import cfg_fl as cfg
from federated.utils import *

import federated.train as federated_train
import federated.loss as federated_loss
import federated.optimizer as federated_optimizer


def setup_eval_datasets(population, limit_train_size=False):
    print_header('> Setting up evaluation datasets')
    # If population dataset was already setup, don't do anything
    if population.train_datasets:
        train_datasets = population.train_datasets
        test_datasets = population.test_datasets
        global_test_dataset = population.global_test_dataset
        return train_datasets, test_datasets, global_test_dataset
    # Otherwise set them up
    train_datasets = [[] for _ in range(len(population.distributions))]
    test_datasets = [[] for _ in range(len(population.distributions))]
    global_test_dataset = []
    for client in population.clients:
        train_datasets[client.dist_id].append(client.datasets[0])
        test_datasets[client.dist_id].append(client.datasets[2])
        global_test_dataset.append(client.datasets[2])

    population.train_datasets = [ConcatDataset(datasets) for datasets in train_datasets]
    population.test_datasets = [ConcatDataset(datasets) for datasets in test_datasets]
    population.global_test_dataset = ConcatDataset(global_test_dataset)

    if args.eval_local_finetune_size is not None and limit_train_size:
        for ix in range(len(population.train_datasets)):
            total_train_size = int(sum([len(d) for d in population.train_datasets[ix].datasets]))
            finetune_size = min([args.eval_local_finetune_size, total_train_size])
            dataset_lengths = [finetune_size, len(population.train_datasets[ix]) -  finetune_size]
            torch.manual_seed(cfg.TORCH_SEED)
            population.train_datasets[ix] = random_split(population.train_datasets[ix], dataset_lengths)[0]
            if args.debugging:
                print_debug(len(population.train_datasets[ix]), f'Train dist {ix} size')

    population.global_train_dataset = ConcatDataset(population.train_datasets)

    # Add these in for finetuning / meta-learning too
    population.train_datasets.append(population.global_train_dataset)
    population.test_datasets.append(population.global_test_dataset)

    return population.train_datasets, population.test_datasets, population.global_test_dataset


def setup_federated_models(server):
    """
    Initialize federated models with multiple possible hierarchies
    - May want to save these models, because then we can just load these during evaluation
    """
    if args.evaluate:
        saved_models_result = load_models_for_eval(server)
        if args.debugging:
            print_debug(saved_models_result, 'Saved / loading model result')
    else:
        # For now, if models were saved before, just load them
        saved_models_result = 1

    if saved_models_result == 0:
        pass
    else:
        if cfg.FEDERATION.METHOD != 'local':
            # Update local models
            server.initialize_clients(epoch=-1, selected_clients=server.last_active_clients)

        # Save updated client local models
        for client in server.last_active_clients:
            model_name = f'local_model_{client.id}-{args.experiment_name}.pth'
            model_path = os.path.join(args.model_path, model_name)
            torch.save({
                'model_state_dict': client.model.state_dict(),
                'optimizer_state_dict': client.optim.state_dict()
            }, model_path)
        print(f'> {len(server.last_active_clients)} client models saved to {args.model_path}!')

        # For local setup, just set the actual client models as the models to evaluate
        if cfg.FEDERATION.METHOD == 'local':
            return
        else:
            global_model = server.global_model
            server.federation_models = [m['model'] for m in server.federated_model_dicts if m['type'] != 'global']

        # Save subfederation averaged models / sub-global models
        for ix, federation_model in enumerate(server.federation_models):
            model_name = f'subfed_model_{ix}-{args.experiment_name}.pth'
            model_path = os.path.join(args.model_path, model_name)
            torch.save(federation_model, model_path)

        # Save federation client ids:
        with open(server.federations_path, 'wb') as f:
            server.clients_by_federation = []
            for ix, federation in enumerate(server.sub_federations):
                server.clients_by_federation.append([])
                for client in federation.clients:
                    server.clients_by_federation[ix].append(client.id)

            pickle.dump(server.clients_by_federation, f)

        print(f'> {len(server.federation_models)} sub-federation models saved to {args.model_path}!')

        # Save global model
        model_name = f'global_model-{args.experiment_name}.pth'
        model_path = os.path.join(args.model_path, model_name)
        torch.save(global_model, model_path)

        print(f'> Global model saved to {args.model_path}!')


def load_models_for_eval(server):
    """
    Function to load models for evaluation if previously initialized
    - Checkpoint to prevent having to train federations from beginning again

    Returns:
    - 0 if all models are successfully loaded from before
    - 1 if client models were not successfully loaded
    - 2 if subfed models were not successfully loaded
    - 3 if global model was not successfully loaded
    """
    try:
        client_model_paths = [f for f in os.listdir(args.model_path) if 'local_model_' in f and args.experiment_name in f]
        assert len(client_model_paths) > 0
        server.last_active_clients = []
        for model_path in client_model_paths:
            client_id = int(model_path.split('local_model_')[-1].split('-')[0])
            checkpoint = torch.load(os.path.join(args.model_path, model_path))
            server.population.clients[client_id].model.load_state_dict(checkpoint['model_state_dict'])
            server.population.clients[client_id].optim.load_state_dict(checkpoint['optimizer_state_dict'])
            server.last_active_clients.append(server.population.clients[client_id])
    except AssertionError:
        print('Error: Local models not loaded successfully')
        return 1

    if cfg.FEDERATION.METHOD == 'local':
        return 0

    try:
        federation_model_paths = [f for f in os.listdir(args.model_path) if 'subfed_model_' in f and args.experiment_name in f]
        assert len(federation_model_paths) > 0
        server.federation_models = []
        for federation_model_path in federation_model_paths:
            server.federation_models.append(torch.load(os.path.join(args.model_path, federation_model_path)))

        with open(server.federations_path, 'rb') as f:
            server.clients_by_federation = pickle.load(f)

    except AssertionError:
        if len(federation_model_paths) == 0 and cfg.FEDERATION.METHOD == 'fedavg':
            server.federation_models = []
            pass
        else:
            print('Error: Sub-global models not loaded successfully')
            return 2

    try:
        global_model_paths = [f for f in os.listdir(args.model_path) if 'global_model-' in f and args.experiment_name in f]
        assert len(global_model_paths) > 0
        server.global_model = torch.load(os.path.join(args.model_path, global_model_paths[0]))

    except AssertionError:
        print('Error: Global model not loaded successfully')
        return 3

    return 0


def run_federated_evaluations(server, dataset, dataset_dist, epoch):
    """
    Method to call for evaluating all federated models
    - Will fail if called before self.setup_federated_models()
    """
    # Eval on the held-out dataset-wide test set
    # - First evaluate using the client's personalized models?
    for ix, client in enumerate(tqdm(server.last_active_clients)):
        avg_loss, avg_acc, _ = client.eval(epoch=epoch, model=client.model, client=client,
                                           val_dataset=dataset, test=True, dataset_dist=dataset_dist)
        server.update_eval_metrics(avg_acc, avg_loss, epoch=epoch, 
                                   dataset_distribution=dataset_dist, 
                                   model_name=f'local-{client.id}')

    if cfg.FEDERATION.METHOD == 'local':
        return

    criterion, criterion_val = federated_loss.get_local_loss()
    avg_loss, avg_acc, _ = federated_train.eval(dataset=dataset, net=server.global_model, 
                                                epoch=epoch, local=True, criterion=criterion, 
                                                device=torch.device('cuda:0'))
    server.update_eval_metrics(avg_acc, avg_loss, epoch=epoch, 
                               dataset_distribution=dataset_dist, 
                               model_name='global')

    for ix, federation_model in enumerate(server.federation_models):
        avg_loss, avg_acc, _ = federated_train.eval(dataset=dataset, 
                                                    net=federation_model,
                                                    epoch=epoch,
                                                    local=True,
                                                    criterion=criterion,
                                                    device=torch.device('cuda:0'))  # By default we use cuda:0 as the "server device"
        if args.debugging:
            print_debug(len(server.clients_by_federation), 'Number subfederations')
            print_debug(len(server.federation_models), 'Number federation models')
            print_debug(server.clients_by_federation[ix], 'Clients by federation')
        model_name = 'subfed-{}'.format(sorted(server.clients_by_federation[ix]))
        server.update_eval_metrics(avg_acc, avg_loss, epoch=epoch, 
                                   dataset_distribution=dataset_dist, 
                                   model_name=model_name)


def eval_ensemble_models(server, level, dataset, dataset_dist=-1, epoch=-1):
    """
    Compute evaluation on a dataset using an ensemble of available models
    - Taken from evaluation method in LG-FedAvg
    - Most common use-cases:
    (1) Evaluate global test set using all local models
    (2) Evaluate global test set using learned sub-global models
    Inputs:
    - level (str): use 'local' or 'sub_global' models. 
    """
    print_header(f'> Evaluating {level} model ensembles')
    probs_all = []
    preds_all = []
    loss_all = []

    if level == 'local': 
        models = [c.model for c in server.last_active_clients]
    elif level == 'sub_global':
        models = server.federation_models
    else:
        raise KeyError("Choose between 'local' or 'sub_global' for level input")

    criterion, criterion_val = federated_loss.get_local_loss()

    for ix, model in enumerate(tqdm(models)):
        model.eval()
        loss, acc, eval_text, probs, targets = federated_train.eval(
            dataset=dataset, net=model, epoch=epoch, local=True,
            criterion=criterion, client=None, device=torch.device('cuda:0'),
            ensemble=True) 
        probs_all.append(probs.detach())
        preds = probs.detach().max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        preds_all.append(preds)
        loss_all.append(loss)

    labels = np.array(targets)  # Only save the last one
    preds_probs = torch.mean(torch.stack(probs_all), dim=0)

    # Average ensemble metrics
    preds_avg = preds_probs.data.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
    loss_test = criterion(preds_probs, torch.tensor(labels).to(torch.device('cuda:0'))).item()
    avg_acc = (preds_avg == labels).mean() * 100

    # Majority ensemble metrics
    preds_all = np.array(preds_all).T
    preds_maj = stats.mode(preds_all, axis=1)[0].reshape(-1)
    maj_acc = (preds_maj == labels).mean() * 100

    model_name = f'ensemble-{level}-avg'
    server.update_eval_metrics(avg_acc, loss_test, epoch=epoch, 
                               dataset_distribution=dataset_dist, 
                               model_name=model_name)

    model_name = f'ensemble-{level}-maj'
    server.update_eval_metrics(maj_acc, np.mean(loss_all), epoch=epoch, 
                               dataset_distribution=dataset_dist, 
                               model_name=model_name)

    print(f'Average {level} ensemble acc: {avg_acc:<.3f} | Majority {level} ensemble acc: {maj_acc:<.3f}')
