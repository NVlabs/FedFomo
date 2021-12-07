"""
Client classes     
- Each client has an associated local dataset and local model
- The local dataset is further divided into a train, val, and test set (where test is used for final evaluation)
"""

import copy
import importlib
import numpy as np

from os.path import join
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard

from federated.configs import cfg_fl as cfg
from federated.args import args
from federated.utils import *

import federated.loss as federated_loss
import federated.network as federated_network
import federated.optimizer as federated_optimizer
import federated.train as federated_train

from federation import init_federated_model

if args.enable_dp:  # Train with differential privacy
    from opacus.utils import stats
    from opacus import PrivacyEngine
    from opacus.utils.module_modification import convert_batchnorm_modules


class Client(object):
    def __init__(self, dataset=None, client_id=None, dist_id=None, population=None,
                 dist_client_ids=[], private=False, device=None, adversarial=False):
        """
        Client class for federated learning
        Inputs:
        - dataset (Dataset): local dataset
        - client_id (int): id for the client
        - dist_id (int): id for the distribution
        - population (Population): client population
        - dist_client_ids (int[]): list of client ids belonging to the same distribution
        - private (bool): for now not implemented, some degree of privacy
        - device (str): cuda device
        """
        self.id = client_id
        self.dist_id = dist_id
        self.target_dist_id = dist_id
        self.population = population
        self.dist_client_ids = dist_client_ids
        self.adversarial = adversarial

        self.federation_weights = {}
        self.model_deltas = {}
        self.model_deltas_sorted = None

        self.EMD = None  # dataset distribution comparison

        self.models = []
        self.model_ids = []
        self.model_deltas = []
        self.model_weights = []

        self.federation = None
        self.local_val_ratio = None
        self.shared_val = None

        self.args = args

        self.num_update_clients = cfg.CLIENT_WEIGHT.NUM_UPDATE_CLIENTS

        self.client_weight_method = cfg.CLIENT_WEIGHT.METHOD
        self.client_weights = None

        self.participated = False

        # Logging performance
        self.metrics = {'train_acc': [], 'train_loss': [],
                        'val_acc': [], 'val_loss': [],
                        'epoch': [], 'client': [],
                        'distribution': [], 'federation': [],
                        'federation_round': [], 'federating_client_ids': [],
                        'client_weights': [], 'model_deltas': [], 'EMD': [],
                        'target_distribution': [], 'unique_classes': []}

        self.test_metrics = {'test_loss': [], 'test_acc': [],
                             'epoch': [], 'client': [],
                             'distribution': [], 'federation': [],
                             'dataset_distribution': [],
                             'federation_round': [], 'federating_client_ids': [],
                             'client_weights': [], 'model_deltas': [], 'EMD': [],
                             'target_distribution': [], 'unique_classes': []}

        if args.enable_dp:
            self.metrics['epsilon'] = []
            self.metrics['delta'] = []
            self.metrics['best_alpha'] = []
            self.test_metrics['epsilon'] = []
            self.test_metrics['delta'] = []
            self.test_metrics['best_alpha'] = []

        if args.dataset == 'imagenet':
            self.metrics['train_acc1'] = []
            self.metrics['train_acc5'] = []
            self.metrics['val_acc1'] = []
            self.metrics['val_acc5'] = []
            self.test_metrics['test_acc1'] = []
            self.test_metrics['test_acc5'] = []

        self.model_weights_over_time = []
        self.client_weights_over_time = []

        self.train_stdout = None
        self.eval_stdout = None

        self.federating_client_ids = [None]

        self.last_active_round = 0

    def initialize(self, dataset, distribution, test_dataset=None):
        """
        Call this for each client to initialize training setup
        - dataset (torch.utils.data.Dataset child): The local dataset
        - distribution (dictionary): Data distribution object
        """
        if distribution is not None:
            self.dist_id = distribution['id']
            self.dist_client_ids = [c.id for c in distribution['clients']]
        else:
            self.dist_id = 0
            self.dist_client_ids = [0]

        self.init_data(dataset, test_dataset)
        self.init_client_weights()

    def init_data(self, dataset, test_dataset=None):
        """
        Initialize local datasets (train, val, test)
        Input:
        - dataset (torch.utils.data.Dataset): The local dataset
        - test_dataset (torch.utils.data.Dataset): If test dataset is precomputed and 
          referenced here (e.g. with pre-organized CIFAR-10 test split), client will 
          use this as their test set. Otherwise we split the dataset into train, val, and test splits.
        Output:
        - Initializes self.datasets, a list of new train, val, and test datasets
        """

        if test_dataset is not None:
            self.dataset_size = len(dataset)
            len_train_split = int(np.round(cfg.CLIENT.TRAIN_SPLIT * self.dataset_size))
            len_val_split = self.dataset_size - len_train_split

            # Limit train and val sizes if specified
            if cfg.FEDERATION.LOCAL_TRAIN_VAL_SIZE:
                train_split = cfg.CLIENT.TRAIN_SPLIT
                val_split = 1. - cfg.CLIENT.TRAIN_SPLIT

                len_train_split = min([len_train_split,
                                       int(np.round(cfg.FEDERATION.LOCAL_TRAIN_VAL_SIZE * train_split))])
                len_val_split = min([len_val_split,
                                     int(np.round(cfg.FEDERATION.LOCAL_TRAIN_VAL_SIZE * val_split))])

            split_lens = [len_train_split, len_val_split]
            len_extra_split = int(np.round(self.dataset_size - np.sum(split_lens)))

            split_lens.append(len_extra_split)

            try:
                self.datasets = torch.utils.data.random_split(dataset, split_lens,
                                                              generator=torch.Generator().manual_seed(args.data_seed))
            except Exception as e:
                torch.manual_seed(args.data_seed)
                self.datasets = torch.utils.data.random_split(dataset, split_lens)

            self.datasets.append(None)  # add in test_split, len(self.datasets) == 4
            self.datasets[-1] = self.datasets[2]  # add test_split to third index
            self.datasets[2] = test_dataset

            self.train_size = len(self.datasets[0])
            return

        else:
            self.dataset_size = len(dataset)

            len_train_split = int(np.round(cfg.CLIENT.TRAIN_SPLIT * self.dataset_size))
            len_test_split = int(np.round(cfg.CLIENT.TEST_SPLIT * self.dataset_size))
            len_val_split = self.dataset_size - (len_train_split + len_test_split)

            # Limit train and val sizes if specified
            if cfg.FEDERATION.LOCAL_TRAIN_VAL_SIZE:
                train_split = cfg.CLIENT.TRAIN_SPLIT
                val_split = 1. - cfg.CLIENT.TRAIN_SPLIT

                len_train_split = min([len_train_split,
                                       int(np.round(cfg.FEDERATION.LOCAL_TRAIN_VAL_SIZE * train_split))])
                len_val_split = min([len_val_split,
                                     int(np.round(cfg.FEDERATION.LOCAL_TRAIN_VAL_SIZE * val_split))])

            split_lens = [len_train_split,
                          len_val_split,
                          len_test_split]  # Keep this constant regardless

            # Allocate remaining to an extra split to keep torch.random_split happy
            len_extra_split = int(self.dataset_size - np.sum(split_lens))
            split_lens.append(len_extra_split)

            torch.manual_seed(cfg.TORCH_SEED)
            self.datasets = torch.utils.data.random_split(dataset, split_lens)

            self.train_size = len(self.datasets[0])

    def init_model(self, model=None, criterion=None, criterion_val=None):
        """
        Initialize local model for training or inference
        """
        if args.dataset == 'imagenet':
            args.num_classes = 1000
            self.device = None
        else:
            try:  # If more than 1 GPU, allocate clients among the GPUs
                self.device = torch.device(f'cuda:{self.id % args.ngpu}')
            except:
                if args.device is not None:
                    self.device = torch.device(f'cuda:{args.device}')
                else:
                    self.device = torch.device('cuda:0')

        if criterion is None:
            self.criterion, self.criterion_val = federated_loss.get_local_loss()
        else:
            self.criterion = criterion
            self.criterion_val = criterion_val

        if args.dataset == 'imagenet':
            self.model = (model.cuda() if model is not None else
                          federated_network.get_net(args, self.criterion).cuda())
            self.optim = torch.optim.SGD(self.model.parameters(), args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        else:
            args.num_classes = self.population.num_classes
            if args.enable_dp:
                self.model = (model if model is not None else
                              federated_network.get_net(args, self.criterion))
                self.model = convert_batchnorm_modules(self.model)
            else:
                self.model = (model.to(self.device) if model is not None else
                              federated_network.get_net(args, self.criterion).to(self.device))
            self.optim, self.scheduler = federated_optimizer.get_local_optimizer(args, self.model)      
        if args.enable_dp:
            # From the Opacus library - save additional differential privacy stats
            stats.set_global_summary_writer(
                tensorboard.SummaryWriter(join('/local/miczhang/workspaces/federated_learning/federated_fomo/results/dp_stats'), f'{args.experiment_name}-client_{self.id}')
            )
            stats.add(
                # stats about gradient norms aggregated for all layers
                stats.Stat(stats.StatType.GRAD, "AllLayers", frequency=0.1),
                # stats about gradient norms per layer
                stats.Stat(stats.StatType.GRAD, "PerLayer", frequency=0.1),
                # stats about clipping
                stats.Stat(stats.StatType.GRAD, "ClippingStats", frequency=0.1),
                # stats on training accuracy
                stats.Stat(stats.StatType.TRAIN, "accuracy", frequency=0.01),
                # stats on validation accuracy
                stats.Stat(stats.StatType.TEST, "accuracy"),
            )
            args.clipping = {'clip_per_layer': False, 'enbale_state': True}

        self.save_first_model()

    def init_client_weights(self):
        """
        Initialize client weights, used to determine which models
        are downloaded to which clients during federating rounds
        """
        # Client weights should express a propensity to federate with others.
        other_weight = 0.
        self.client_weights = np.ones(self.population.num_clients) * other_weight 
        # Set client's weighting for its own model
        self.client_weights[self.id] = args.client_initial_self_weight

        # When we compute the affinity matrix, we subtract everything from 1 first

    def update_model(self, federated_model_state):
        """
        After obtaining federated model parameters, update the local model
        Input:
        - federated_model_state (nn.Model.state_dict): Updated model parameters
        """
        self.model = init_federated_model(federated_model_state,
                                          criterion=self.criterion,
                                          device=None)
        if args.enable_dp:
            self.model = convert_batchnorm_modules(self.model)

        # Reset optimizer and schedular too to go along with the new model
        self.optim, self.scheduler = federated_optimizer.get_local_optimizer(args, self.model)

    def get_federating_clients(self, epoch, method=None,
                               num_clients=None,
                               epsilon=cfg.CLIENT_WEIGHT.EPSILON,
                               epsilon_decay=cfg.CLIENT_WEIGHT.EPSILON_DECAY,
                               possible_clients=None):
        """
        Obtain list of federating clients for the round
        Inputs:
        - method (str): How to select clients
        - num_clients (int): How many clients to select
        - epsilon (float): For exploration-based methods, probability of randomly selecting a client from the population
        - possible_clients (Clients []): List of available local clients to request
        """
        clients = [self]  # if self.participated else []
        num_clients = self.num_update_clients if num_clients is None else num_clients

        method = self.client_weight_method if method is None else method

        if method == 'sub_federations':
            [clients.append(client) for client in self.federation.clients if client not in clients]
        else:
            if method == 'e_greedy':
                # Precompute the top clients and random clients
                if possible_clients is not None:
                    possible_clients = [c for c in possible_clients if c.id != self.id]
                    possible_ids = [c.id for c in possible_clients]
                    client_weights = [self.client_weights[c_id] for c_id in possible_ids]
                else:
                    possible_clients = [c for c in self.population.clients if c.id != self.id]
                    client_weights = self.client_weights

                # argsort but with random tie-breaking
                random_vals = np.random.random(len(client_weights))
                top_clients_ix = list(np.lexsort((random_vals, client_weights))[::-1])

                # Essentially just shuffle here
                rand_clients = list(np.random.choice(possible_clients, size=len(possible_clients), replace=False))

                # E-greedy sampling: Select from remaining clients 
                # with the highest value, with epsilon chance to sample randomly
                for ix in range(num_clients):
                    explore = np.random.uniform(0, 1)
                    # Loop through until we get a suitable client 
                    client_chosen = False 
                    while not client_chosen:
                        # If exploring, take the first random client and remove
                        if explore < epsilon - (epoch * epsilon_decay):
                            possible_client = rand_clients.pop(0)

                        else:  # Otherwise take the first top client and remove
                            possible_client = possible_clients[top_clients_ix.pop(0)]
                        if possible_client not in clients:
                            clients.append(possible_client)
                            client_chosen = True
        self.federating_client_ids = [client.id for client in clients]
        return clients

    def save_last_model(self, to_disk=False):
        """
        For federating with weights, save the current model after training as the last model
        - Used for comparisons with other models during federating
        Args:
        - to_disk (bool): If True, save the model state_dict to disk. (By default we keep in memory)
        """
        # Parallelized / torch.multiprocessing
        if args.parallelize:
            self.last_model = federated_network.get_net(args, self.criterion)
            self.last_model_weights = copy.deepcopy(self.model.state_dict())
            new_state_dict = OrderedDict()
            for k, v in self.last_model_weights.items():
                name = k[7:]  # remove 'module'
                new_state_dict[name] = v

            if to_disk:
                last_model_path = os.path.join(self.args.model_path, f'm-{self.args.experiment_name}-c{self.id}.p')
                torch.save(new_state_dict, last_model_path)

            else:
                self.last_model.load_state_dict(new_state_dict)
                self.last_model.share_memory()  # Use this to access parameters from before?

            return
        else:
            self.last_model = federated_network.get_net(args, self.criterion)
            last_model_weights = copy.deepcopy(self.model.state_dict())

            self.last_model.load_state_dict(last_model_weights)

    def save_first_model(self, to_disk=False):
        """
        For federating with weights, save the current model after training as the first model
        - Used for comparisons with other models during federating
        Args:
        - to_disk (bool): If True, save the model state_dict to disk. (By default we keep in memory)
        """
        # Parallelized / torch.multiprocessing
        if args.parallelize:
            self.first_model = federated_network.get_net(args, self.criterion)
            self.first_model_weights = copy.deepcopy(self.model.state_dict())
            new_state_dict = OrderedDict()
            for k, v in self.first_model_weights.items():
                name = k[7:]  # remove 'module'
                new_state_dict[name] = v

            if to_disk:
                last_model_path = os.path.join(self.args.model_path, f'm-{self.args.experiment_name}-c{self.id}.p')
                torch.save(new_state_dict, last_model_path)

            else:
                self.first_model.load_state_dict(new_state_dict)
                # Use this to access parameters from before
                self.first_model.share_memory()  

            return
        else:
            self.first_model = federated_network.get_net(args, self.criterion)
            first_model_weights = copy.deepcopy(self.model.state_dict())

            self.first_model.load_state_dict(first_model_weights)

    def reset(self):
        self.models = []
        self.model_ids = []
        self.model_deltas = []
        self.model_weights = []

    def train(self, epoch, dataset=None, args=args, return_stdout=False):
        """
        Trains local model with federated.train and saves results in self.metrics
        Args:
        - epoch (int): The current local training epoch
        - dataset (torch.nn.data.Dataset): Training dataset 
          (by default we use the local training set initially allocated to the client)
        - args (argparse): Experiment arguments, by default from federated.args
        - return_stdout (bool): Whether to return training results for display
        """
        train_set = self.datasets[0] if dataset is None else dataset
        self.train_size = len(train_set)

        if args.enable_dp:
            self.model = convert_batchnorm_modules(self.model)
        else:
            self.optim, self.scheduler = federated_optimizer.get_local_optimizer(args, self.model)

        if args.parallelize:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        else:
            train_sampler = None

        loss, acc, train_text = federated_train.train(train_set, self.model, self.optim, epoch, 
                                                      local=True, criterion=self.criterion, 
                                                      device=self.device, client=self, sampler=train_sampler,
                                                      args=args)
        self.train_stdout = train_text

        if args.dataset == 'imagenet':
            acc1, acc5 = acc
            acc = acc1
            self.metrics['train_acc1'].append(acc1)
            self.metrics['train_acc5'].append(acc5)

        if args.enable_dp:
            stats.update(stats.StatType.TRAIN, acc1=acc)

        self.metrics['train_acc'].append(acc)
        self.metrics['train_loss'].append(loss)
        self.metrics['epoch'].append(epoch)
        self.metrics['client'].append(self.id)
        self.metrics['distribution'].append(self.dist_id)
        federation_id = self.federation.id if self.federation is not None else None
        self.metrics['federation'].append(federation_id)
        # self.metrics['federation_round'].append(epoch % (cfg.FEDERATION.EPOCH - 1) == 0 and epoch != 0)
        self.metrics['federation_round'].append(args.federation_round)
        self.metrics['federating_client_ids'].append(self.federating_client_ids)
        self.metrics['client_weights'].append(self.client_weights)
        self.metrics['model_deltas'].append(self.model_deltas_sorted)
        self.metrics['EMD'].append(self.EMD)
        self.metrics['target_distribution'].append(self.target_dist_id)
        self.metrics['unique_classes'].append(None)

        if args.enable_dp:
            epsilon, best_alpha = self.optim.privacy_engine.get_privacy_spent(args.delta)
            self.metrics['epsilon'].append(epsilon)
            self.metrics['delta'].append(args.delta)
            self.metrics['best_alpha'].append(best_alpha)

        if return_stdout:
            self.train_stdout = train_text
            return train_text

    def eval(self, epoch, model=None, client=None, val_dataset=None, 
             log=False, test=False, dataset_dist=None, args=args, parallelize=False, 
             return_stdout=False, output_metrics=False):
        """
        Evaluate client's model on test data

        Args:
        - epoch (int): Local training epoch
        - model (torch.nn.Module): Local model being evaluated. If None, sets to self.model
        - client (Client): Client whose local model is being evaluated. Defaults to self
        - val_dataset (torch.nn.data.Dataset): Client's local validation dataset by default
        - log (bool), test (bool): See note below
        - dataset_dist (int): If known, the distribution of the dataset being evaluated
        - parallelize (bool): Whether to parallelize (e.g. for ImageNet training). Still working on this
        - return_stdout (bool): If True, returns outputs for display
        - output_metrics (bool): If True, returns outputs to save (set True in instance (2) below) 

        Several instances when this is called:
        (1) During training, evaluate on test split to track acc. after each local training epoch
        (2) During training, evaluate on test split to track acc. after each federated update
        (3) After training, evaluate on test split
        (4) When computing federated updates, evaluate downloaded model on client's dataset

        For (1): Set log = True,  test = False
        For (2): Set log = False, test = False
        For (3): Set log = False, test = True
        For (4): Set log = False, test = False
        """

        val_set = self.datasets[1] if val_dataset is None else val_dataset
        if model is None:
            model = self.model

        if parallelize and args.parallelize:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
        else:
            val_sampler = None  # Just to be sure
            try:
                self.device = torch.device(f'cuda:{self.id % args.ngpu}')
                args.gpu = self.device
            except:
                args.gpu = self.device

        try:
            outputs = federated_train.eval(val_set, model, epoch, True, self.criterion, self.device, 
                                           client, sampler=val_sampler, optim=self.optim, args=args)
            self.eval_stdout = outputs[-1]
        except Exception as e:
            print_debug(len(val_set), 'len(val_set)')
            print_debug(len(self.datasets[1]), 'len(self.datasets[1])')
            print_debug(val_dataset, 'val_dataset')
            print_debug(len(val_set.indices), 'Number of indices')
            print_debug(np.max(val_set.indices), 'Largest index')
            print_debug(np.min(val_set.indices), 'Smallest index')
            print_debug(len(val_set.samples), 'DatasetFolder len(samples)')
            print_debug(self.id, 'Client id')
            raise e

        if args.dataset == 'imagenet':
            acc1, acc5 = outputs[1]

        if output_metrics:
            return outputs  # epoch_loss, total_correct / total_eval, None

        if log is True:
            self.metrics['val_loss'].append(outputs[0])

            if args.enable_dp:
                stats.update(stats.StatType.TEST, acc1=outputs[1])

            if args.dataset == 'imagenet':
                self.metrics['val_acc'].append(acc1)
                self.metrics['val_acc1'].append(acc1)
                self.metrics['val_acc5'].append(acc5)
            else:
                self.metrics['val_acc'].append(outputs[1])

        if test is True:
            if args.dataset == 'imagenet':
                self.test_metrics['test_acc'].append(acc1)
                self.test_metrics['test_acc1'].append(acc1)
                self.test_metrics['test_acc5'].append(acc5)
            else:
                self.test_metrics['test_acc'].append(outputs[1])

            self.test_metrics['test_loss'].append(outputs[0])
            self.test_metrics['epoch'].append(epoch)
            self.test_metrics['client'].append(self.id)
            self.test_metrics['distribution'].append(self.dist_id)
            self.test_metrics['dataset_distribution'].append(dataset_dist)
            self.test_metrics['target_distribution'].append(self.target_dist_id)
            federation_id = self.federation.id if self.federation is not None else None
            self.test_metrics['federation'].append(federation_id)
            self.test_metrics['federation_round'].append(args.federation_round)
            self.test_metrics['federating_client_ids'].append(self.federating_client_ids)
            self.test_metrics['client_weights'].append(self.client_weights)
            self.test_metrics['EMD'].append(self.EMD)
            self.test_metrics['unique_classes'].append(None)

            if args.enable_dp:
                epsilon, best_alpha = self.optim.privacy_engine.get_privacy_spent(args.delta)
                self.test_metrics['epsilon'].append(epsilon)
                self.test_metrics['delta'].append(args.delta)
                self.test_metrics['best_alpha'].append(best_alpha)
            return outputs

        if return_stdout:
            self.eval_stdout = outputs[-1]
            return outputs[-1]

        if cfg.TASK == 'classification':
            model_loss, model_accuracy, text = outputs
            return model_loss
        elif cfg.TASK == 'semantic_segmentation':
            model_loss, model_accuracy, text = outputs
            return model_loss
        return outputs

    def log_training_curve(self, epoch, acc, loss, val=False):
        """
        Save training results to client metrics for later analysis
        """
        if val:
            self.metrics['val_loss'].append(loss)
            self.metrics['val_acc'].append(acc)
        else:
            self.metrics['train_acc'].append(acc)
            self.metrics['train_loss'].append(loss)
            self.metrics['epoch'].append(epoch)
            self.metrics['client'].append(self.id)
            self.metrics['distribution'].append(self.dist_id)
            self.metrics['target_distribution'].append(self.target_dist_id)
            federation_id = self.federation.id if self.federation is not None else None
            self.metrics['federation'].append(federation_id)
            self.metrics['federation_round'].append(args.federation_round)
            self.metrics['federating_client_ids'].append(self.federating_client_ids)
            self.metrics['model_deltas'].append(self.model_deltas_sorted)
