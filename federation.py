"""
Federation class
"""

import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch

import federated.network as federated_network

from federated.args import args
from federated.configs import cfg_fl as cfg
from federated.utils import *


def init_federated_model(model_params, criterion, device=None):
    """
    Code to initialize a model given weights, criterion, and a cuda device
    """
    model = federated_network.get_net(args, criterion)
    try:
        model.load_state_dict(copy.deepcopy(model_params))
    except Exception as e:
        print(e)
        new_state_dict = OrderedDict()
        for k, v in copy.deepcopy(model_params).items():
            name = k[7:]  # remove 'module'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    return model


def average_model_params(clients=None, fed_models=None, fed_weights=None,
                         total_train_size=None, dataset_size_weight=True,
                         weight_delta=None, federating=False):
    """
    Compute a federated model, with a couple ways supported
    """
    # Collect weights and model state_dicts
    if weight_delta is not None:
        assert fed_models is not None
        weights = [1. - weight_delta, weight_delta]
        models = [copy.deepcopy(model.to(torch.device('cpu'))).state_dict()
                  for model in fed_models]
    elif clients is not None:
        weights = []; models = []
        for client in clients:
            if dataset_size_weight:
                assert total_train_size is not None
                weights.append((client.train_size / total_train_size))
            else:
                weights.append(1. / len(clients))  # Model average not considering dataset sizes
            models.append(copy.deepcopy(
                client.model.to(torch.device('cpu'))).state_dict())
    elif fed_models is not None:
        weights = []; models = []
        for ix, model in enumerate(fed_models):
            weight = (1. / len(fed_models) if fed_weights is None
                      else fed_weights[ix])
            weights.append(weight)
            models.append(copy.deepcopy(
                model.to(torch.device('cpu'))).state_dict())
    else:
        raise NotImplementedError

    # Check if weights are reasonable, i.e. sum close to 1
    try:
        assert np.sum(weights) == 1
    except:
        print_debug(np.sum(weights), f'Sum of model weights for client')
        adjustment = 1. - np.sum(weights)
        if np.abs(adjustment) < 1e-7:
            print_debug(adjustment, f'Close enough? Adjustment factor')
            pass
        else:
            if federating:
                pass
            else:
                raise AssertionError('Model weights do not sum to 1')

    # Compute the weighted model average
    model_avg = models[0]
    for k in model_avg.keys():
        model_avg[k] = model_avg[k] * weights[0]
    # Add additional models
    for k in model_avg.keys():
        for i in range(1, len(models)):
            model_avg[k] += models[i][k] * weights[i] 
    return model_avg


def average_model_params_momentum(clients, last_momentum_weights,
                                  last_global_model, nesterov=False):
    """
    Perform model averaging with momentum over clients' local models
    """
    total_train_size = np.sum([c.train_size for c in clients])

    assert last_momentum_weights is not None
    assert last_global_model is not None
    weights = []; models = [];
    for client in clients:
        weights.append((client.train_size / total_train_size))
        models.append(copy.deepcopy(
            client.model.to(torch.device('cpu'))).state_dict())

    # Update momentum weights
    last_model = copy.deepcopy(
        last_global_model.to(torch.device('cpu'))).state_dict()
    momentum_weights = copy.deepcopy(last_momentum_weights)

    if nesterov:
        look_ahead_model = copy.deepcopy(last_model)
        for k in look_ahead_model.keys():
            look_ahead_model[k] -= args.fed_momentum_gamma * last_momentum_weights[k]

        for k in momentum_weights.keys():
            for i in range(0, len(models)):
                momentum_weights[k] = (args.fed_momentum_gamma * last_momentum_weights[k]
                                       + (look_ahead_model[k] - models[i][k]) * weights[i])

        model_params = copy.deepcopy(last_model)
        for k in model_params.keys():
            model_params[k] -= momentum_weights[k]

        return model_params, momentum_weights

    for k in momentum_weights.keys():
        for i in range(0, len(models)):
            # Actual momentum term calculation
            momentum_weights[k] = (args.fed_momentum_gamma * last_momentum_weights[k]
                                   + (last_model[k] - models[i][k]) * weights[i])

    model_params = copy.deepcopy(last_model)
    for k in model_params.keys():
        for i in range(0, len(models)):
            # Actual momentum update
            model_params[k] -= momentum_weights[k]
    return model_params, momentum_weights


def compute_fomo_weights(client, baseline_model, comparison_models, 
                         parameter_distance=False, epoch=None):
    """
    Compute first order model optimization update weights
    - Calculated by comparing loss of baseline model against comparison models on client's val datasets
    Returns:
    - fomo weights (np.ndarray): List of weights for computing a FOMO federated update
    """
    loss_deltas = []
    weight_deltas = np.ones(len(comparison_models))

    baseline_model.eval()
    baseline_loss = client.eval(epoch=epoch, model=baseline_model, args=client.args)
    for comparison_model in comparison_models:
        comparison_model.eval()
        comparison_loss = client.eval(epoch=epoch, model=comparison_model, args=client.args)
        if args.baseline_model == 'model_avg':
            # Comparison loss is higher if evaluated model is good
            # L(all models avg w/o eval model) - L(all models avg) > 0
            loss_deltas.append(comparison_loss - baseline_loss)
        else:
            # Comparison loss is lower if evaluated model is good
            # L(own model) - L(eval model) > 0
            loss_deltas.append(baseline_loss - comparison_loss)

    if parameter_distance:
        for ix, comparison_model in enumerate(comparison_models):
            weight_delta = compute_parameter_difference(comparison_model,
                                                        baseline_model,
                                                        norm=args.model_delta_norm)
            weight_deltas[ix] = weight_delta
    # Should broadcast correctly
    return np.array(loss_deltas) / weight_deltas


class FederatedModel():
    def __init__(self, model, client_ids, model_type):
        """
        Federated model wrapper class that lets a client interact with a 
        sub-global or global model as if it were a client
        Args:
        - model (torch.nn): Federated model (not the model state_dict)
        - client_ids (np.ndarray): Numpy array of client ids associated with the model
        """
        self.model = model
        self.id = str(client_ids)[1:-1]  # array([0, 1, 2]) -> '0 1 2' 
        self.model_type = model_type  # 'global' or 'sub_global'


class Federation():
    def __init__(self, clients, federation_id, dataset_size_weight=True, 
                 fed_client_ratio=cfg.FEDERATION.CLIENT_RATIO, cfg=cfg, epoch=0):
        """
        Federation class. Compute this federation based on the active clients per federated round
        Args:
        - clients (Clients[]): List of clients assigned to this federation
        - federation_id (int): Federation index used for bookmarking
        - dataset_size_weight (bool): If true, compute average as in FedAvg, else use simple avg across client models
        """
        self.clients = clients
        self.id = federation_id
        self.fed_client_ratio = fed_client_ratio

        self.model_deltas = {}  # Store differences in model performance
        for client in self.clients:
            client.federation = self
            self.model_deltas[client.id] = {}

        self.model_weight_delta = cfg.CLIENT_WEIGHT.WEIGHT_DELTA
        self.baseline = cfg.CLIENT_WEIGHT.BASELINE  # either 'last_model' or 'model_avg' or 'current'
        self.leave_one_out = cfg.CLIENT_WEIGHT.LEAVE_ONE_OUT

        self.cfg = cfg
        self.epoch = epoch

        self.starting_epsilon = self.cfg.CLIENT_WEIGHT.EPSILON
        self.epsilon_decay = self.cfg.CLIENT_WEIGHT.EPSILON_DECAY
        self.current_epsilon = max([self.starting_epsilon - epoch * self.epsilon_decay, 0])

        # Compute federated model
        ## Only compute model among the most recently active clients
        last_active_round = max([c.last_active_round for c in self.clients])
        federating_clients = [c for c in self.clients if c.last_active_round == last_active_round]
        print(f'Federation {self.id} federating clients:', [c.id for c in federating_clients])
        self.total_train_size = np.sum([c.train_size for c in federating_clients])
        try:
            self.avg_model_params = average_model_params(federating_clients, total_train_size=self.total_train_size, 
                                                         dataset_size_weight=dataset_size_weight)

            self.model = init_federated_model(self.avg_model_params, criterion=self.clients[0].criterion, device=None)
            self.all_federated_models = {}  # Store other federated models (models from other Federations)
        except Exception as e:
            print(len(self.clients))
            if len(self.clients) == 0:
                pass
            raise e

    def get_loss_delta(self, client_id_a, client_id_b,
                       model_a=None, model_b=None):
        """
        Method to retrieve loss delta between two clients
        """
        try:
            delta = self.model_deltas[client_id_a][client_id_b]
        except KeyError:
            if client_id_a not in self.model_deltas:
                raise ValueError(f'Client {client_id_a} has not evaluated models yet')
            if client_id_b not in self.model_deltas[client_id_a]:
                raise ValueError(f'Client {client_id_b} has not been evaluated for client {client_id_a}')
        return delta  # if args.eval_by_class, this will be a tuple

    def compute_client_weights(self, epoch, baseline=None,
                               federated_models=None, available_clients=None):
        """
        1. Query clients for their desired clients they want to federate with
        2. Compute the model average based on the weight delta specified
        3. Collect information on the model losses based on the payout
        """
        np.random.seed(cfg.SEED)

        # First query client models and compute model loss deltas
        for ix, client in enumerate(tqdm(self.clients)):
            # Save model deltas to self.model_deltas
            self.eval_client_requested_models(client, epoch, baseline, federated_models, available_clients)

        # After all of model deltas are calculated, can actually compute the deltas for updating weights for the clients
        for client in self.clients:
            for client_ix, requested_client_id in enumerate(client.federating_client_ids):
                # Should be good given that federating_clients order is preserved
                client.model_deltas[client_ix] = self.get_loss_delta(client.id, requested_client_id)

            # Actually update model weights and client-to-client weights
            self.update_client_weights(client, epoch, baseline=baseline)

    def eval_client_requested_models(self, client, epoch, baseline=None, 
                                     federated_models=None, available_clients=None):
        """
        For a client, send copies of requested models or current federated models to a client,
        collecting reported evalluation losses for comparison
        """
        # Get baseline model
        if self.baseline == 'model_avg':
            baseline_model = self.model  # Use Federation's average as the baseline

        elif self.baseline == 'last_model':  # 1855
            try:
                baseline_model = client.last_model
            except AttributeError:
                client.save_last_model()
                baseline_model = client.last_model

        elif self.baseline == 'first_model':
            baseline_model = client.first_model

        elif self.baseline == 'current':
            baseline_model = client.model

        # client.federating_client_ids also initialized here:
        client_federating_clients = client.get_federating_clients(epoch=epoch,
                                                                  epsilon=self.cfg.CLIENT_WEIGHT.EPSILON,
                                                                  epsilon_decay=self.cfg.CLIENT_WEIGHT.EPSILON_DECAY,
                                                                  possible_clients=available_clients)
        # Incorporate the existing subglobal and global models
        if self.baseline != 'model_avg':
            client_federating_clients.extend(federated_models)
            client.federating_client_ids.extend([model.id for model in federated_models])

        # Number of federated average models <- incorporated already in line above
        # num_fed_avg_models = 0  # len(self.all_fed_models) - 1
        num_total_models = len(client_federating_clients)
        client.models = [None] * num_total_models
        client.model_ids = [None] * num_total_models
        client.model_deltas = [None] * num_total_models

        comparison_models = []

        # For each index based on client_federating_clients, 
        for client_ix, fed_client in enumerate(tqdm(client_federating_clients, leave=False)):
            if self.baseline == 'model_avg' and self.leave_one_out:
                # Compare baseline model against models where 1 client's contribution is left out
                # L(baseline) - L(baseline w/o model i) < 0 if model i is good (taking it out lead to higher loss)
                if len(client_federating_clients) > 1:
                    model_clients = [client for client in client_federating_clients if client.id != fed_client.id]
                    print(f'> Leave-client {fed_client.id}-out average:')
                    for c in model_clients:
                        print(f'Client {c.id}')
                    print('-------------------------')
                else:
                    model_clients = client_federating_clients
                weight_delta = 1. / num_total_models
                client.num_total_models = num_total_models
                model_params = average_model_params(model_clients, dataset_size_weight=0)
                comparison_model = init_federated_model(model_params,
                                                        criterion=client.criterion, 
                                                        device=client.device)
            elif self.baseline in ['last_model', 'first_model']:
                if self.model_weight_delta == 1:
                    comparison_model = init_federated_model(fed_client.model.state_dict(),
                                                            criterion=client.criterion,
                                                            device=client.device)
                else:
                    model_clients = [client.last_model, fed_client.model]
                    weight_delta = self.model_weight_delta
                    model_params = self.average_model_params(models=[client.last_model,
                                                                     fed_client.model],
                                                             weight_delta=weight_delta)
                    comparison_model = init_federated_model(model_params, 
                                                            criterion=client.criterion, 
                                                            device=client.device)
            elif self.baseline == 'current':
                comparison_model = init_federated_model(fed_client.model.state_dict(),
                                                        criterion=client.criterion,
                                                        device=client.device)
            comparison_models.append(comparison_model)
            client.models[client_ix] = fed_client.model
            client.model_ids[client_ix] = fed_client.id

        param_distance = False if args.no_model_delta_norm else True

        model_weights = compute_fomo_weights(client, baseline_model,
                                             comparison_models,
                                             parameter_distance=param_distance,
                                             epoch=epoch)

        for client_ix, fed_client in enumerate(client_federating_clients):
            self.model_deltas[client.id][fed_client.id] = model_weights[client_ix]

    def update_client_weights(self, client, epoch, baseline=None):
        """
        Actual code to update the client model weights and client-to-client weights
        - Right now epoch isn't used
        """

        # Update client-to-client weights first
        if args.eval_by_class:
            normalization_factor = np.abs(np.sum([md[-1] for md in client.model_deltas]))
            if normalization_factor < 1e-9:
                print_debug('Normalization factor is really small')
                normalization_factor += 1e-9  # Try to not have any 0's
                # But also need to deal with possibility that the deltas are super big
            all_deltas_normed = list(np.array([md[-1] for md in client.model_deltas]) / normalization_factor)
            num_reduced_positives = len([md[-1] for md in client.model_deltas if md[-1] > 0])
            client.model_deltas = [md[0] for md in client.model_deltas]
        else:
            try:
                normalization_factor = np.abs(np.sum(client.model_deltas))
            except Exception as e:
                print(client.model_deltas)
                raise e
            if normalization_factor < 1e-9:
                print_debug('Normalization factor is really small')
                normalization_factor += 1e-9 
            all_deltas_normed = list(np.array(client.model_deltas) / normalization_factor)  

        delta_multiplier = 0
        if args.baseline_model == 'model_avg':
            delta_multiplier = client.num_total_models
        for ix, delta in enumerate(all_deltas_normed):
            model_id = client.model_ids[ix]
            if type(model_id) == str:  # checking 'global' is not needed
                local_model_ids = [int(x) for x in model_id.split(' ') if x != '']
                model_type = 'Global' if len(local_model_ids) > 10 else 'Sub-global'
                for local_model_id in local_model_ids:
                    local_delta = delta / len(local_model_ids) * delta_multiplier
                    client.client_weights[local_model_id] += local_delta
            else:
                client.client_weights[client.model_ids[ix]] += delta * 1

        # Only consider positive models
        if cfg.MODEL_WEIGHT.UPDATE_POSITIVE_ONLY:
            num_local_positive = 0
            positive_deltas_all = []
            positive_deltas = []; positive_client_ids = []; positive_models = [];
            for ix, delta in enumerate(client.model_deltas):
                if delta > 0:
                    positive_deltas.append(delta)
                    positive_deltas_all.append(delta)
                    positive_client_ids.append(client.model_ids[ix])
                    positive_models.append(client.models[ix])
                    
                    if type(client.model_ids[ix]) != str:
                        num_local_positive += 1
                else:
                    positive_deltas_all.append(0.)

            if np.sum(positive_deltas_all) > 0:
                positive_deltas_all = np.array(positive_deltas_all) / np.sum(positive_deltas)

            client.models = positive_models
            client.federating_model_ids = positive_client_ids

            multiplier = 1
            client.model_weights = np.array(positive_deltas) / np.sum(positive_deltas) * multiplier

            if len(client.models) == 0:
                print_debug(f'No models performed higher than {self.baseline} for client {client.id}')
                if self.baseline == 'last_model':
                    client.models = [client.last_model]
                elif self.baseline == 'first_model':
                    client.models = [client.first_model]
                elif self.baseline == 'model_avg':
                    client.models = [self.model]
                elif self.baseline == 'current':
                    client.models = [client.model]
                else:
                    raise NotImplementedError
                client.model_weights = [1.]
                client.federating_model_ids = [client.id]

            deltas_sorted = [0] * client.population.num_clients
            for ix, delta in enumerate(all_deltas_normed):
                try:
                    deltas_sorted[client.model_ids[ix]] = delta
                except TypeError:
                    deltas_sorted.append((client.model_ids[ix], delta))

            model_deltas_sorted = [0] * client.population.num_clients
            for ix, delta in enumerate(client.model_weights):
                try:
                    model_deltas_sorted[client.federating_model_ids[ix]] = delta
                except TypeError:
                    model_deltas_sorted.append((client.model_ids[ix], delta))
            
            client.model_deltas_sorted = model_deltas_sorted

            client.model_weights_over_time.append(model_deltas_sorted)
            client.client_weights_over_time.append(copy.copy(client.client_weights))
            
            # Print extra information
            if args.debugging:
                print('-' * 20)
                print_debug(positive_client_ids, f'Federating clients for Client {client.id}')
                print_debug(client.model_ids, f'All chosen clients for Client {client.id}')

                print_debug(client.model_deltas, f'All client deltas for Client {client.id}')
                print_debug(all_deltas_normed, f'All normed client deltas for Client {client.id}')
                
                print_debug(deltas_sorted, f'All client deltas sorted for Client {client.id}')
                print_debug(model_deltas_sorted, f'All model weights sorted for Client {client.id}')
                print_debug(list(client.client_weights), f'All client weights for Client {client.id}')
            print('')

    
    def federate(self, last_global_model=None):
        """
        Compute actual federated models for each model assigned to the federation
        - Update the client models that participated this round
        - Also update the existing federated models?
        """
        total_train_size = np.sum([c.train_size for c in self.clients])

        if cfg.FEDERATION.FED_AVERAGING:
            if args.fed_momentum:
                assert last_global_model is not None
                for ix, client in enumerate(self.clients):
                    for name, p in client.model.named_parameters():
                        p.detach()
                    client.model.to('cpu')
                    del client.model; del client.optim; del client.scheduler;
                    client.update_model(copy.deepcopy(last_global_model))
                return  # Can ignore the rest but leaving for safekeeping
              
            assert last_global_model is not None
            for ix, client in enumerate(self.clients):
                for name, p in client.model.named_parameters():
                    p.detach()
                client.model.to('cpu')
                del client.model; del client.optim; del client.scheduler;
                client.update_model(copy.deepcopy(last_global_model))  # should be updated already
            return None

        elif cfg.FEDERATION.METHOD == 'fomo':
            # Each client gets a personalized model update
            for client in self.clients:  
                # Do model update step as before:
                weights = []; models = []
                for ix, model in enumerate(client.models):
                    weights.append(client.model_weights[ix])
                    models.append(copy.deepcopy(model.to(torch.device('cpu'))).state_dict())

                if args.baseline_model == 'first_model':
                    client_model = copy.deepcopy(client.first_model.to(torch.device('cpu'))).state_dict()
                elif args.baseline_model == 'last_model':
                    client_model = copy.deepcopy(client.last_model.to(torch.device('cpu'))).state_dict()
                elif args.baseline_model == 'current':
                    client_model = copy.deepcopy(client.model.to(torch.device('cpu'))).state_dict()
                elif args.baseline_model == 'model_avg':
                    client_model = copy.deepcopy(self.model.to(torch.device('cpu'))).state_dict()

                model_params = copy.deepcopy(client_model)
                
                for k in model_params.keys():
                    for i in range(0, len(models)):  # Main change is this is an explicit addition of updates
                        model_params[k] += (models[i][k] - client_model[k]) * weights[i]
                
                client.federated_update_params = model_params
                # Save the update to not interfere with other clients first, then actually update model
            for client in self.clients:
                for name, p in client.model.named_parameters():
                    p.detach()
                client.model.to('cpu')
                del client.model; del client.optim; del client.scheduler;
                client.update_model(client.federated_update_params)
            return None
