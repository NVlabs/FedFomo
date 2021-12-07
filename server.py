"""
Server class for setting up and maintaining federations + global models
"""
import os
import pickle
import copy
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset, random_split

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_distances

import federated

from federated.utils import *
from federated.args import args
from federated.configs import cfg_fl as cfg
from federation import Federation, FederatedModel
from federation import init_federated_model, average_model_params, average_model_params_momentum, compute_fomo_weights


class GlobalServer():
    """
    Server class. By default is used to maintain uploading and downloading of individual local models
    - Also maintains a model average baseline over all client local models

    Extended features for hierarchical FedFomo to:
    - Calculate "sub-federations". Compute clusters over the clients based on client-client weights
      as smaller federations with aligned target objectives 
    - Compute and maintain "sub-federated" models. Given clustered sub-federations, for each 
      compute a model average over all clients' local models in that sub-federation as a sub-federated model

    By default for clustering we require knowledge of number of clusters / distributions to discover. 
    - Potential extension is switching the clustering method for one that does not

    Args:
    - population (federated_datasets.dataset object): All clients
    - num_federations (int): Number of total sub-federations to setup
    - args (argparse): Experiment arguments
    """
    def __init__(self, population, num_federations=None, args=args):
        self.device = torch.device('cuda:0')

        self.args = args
        self.population = population
        self.num_federations = (num_federations if num_federations is not None 
                                else population.num_distributions)

        if args.federation_method == 'fomo' and args.shards_per_user is not None:
            self.num_federations = args.num_distributions
            print(f'> Number of clustering federations: {self.num_federations}')

        # Save evaluation metrics after federating
        self.eval_metrics = {'val_acc': [], 'val_loss': [],
                             'epoch': [], 'model': [],
                             'dataset_distribution': []}

        # Save evaluation metrics during learning
        self.client_eval_metrics = {'val_acc': [], 'val_loss': [],
                                    'val_acc_global': [], 'val_loss_global': [],
                                    'epoch': [], 'model': [],
                                    'dataset_distribution': [], 
                                    'target_distribution': [], 'EMD': []}

        if args.enable_dp:
            self.client_eval_metrics['epsilon'] = []
            self.client_eval_metrics['delta'] = []
            self.client_eval_metrics['best_alpha'] = []

        self.federations_path = os.path.join(args.model_path, f'federations-{args.experiment_name}.p')

        self.train_datasets = None
        self.test_datasets = None

        # Initialize first global model
        args.num_classes = self.population.num_classes
        self.criterion, self.criterion_val = federated.loss.get_local_loss()
        self.global_model = federated.network.get_net(args, self.criterion)
        # Initialize each client's local model
        for client in self.population.clients:
            self.criterion, self.criterion_val = federated.loss.get_local_loss()
            client_model = init_federated_model(self.global_model.state_dict(), self.criterion, device=None)
            client.init_model(client_model, self.criterion, self.criterion_val)
            print(f'Initializing client {client.id:3d} with global model parameters')

        # Initialize / keep track of federated models and uploaded clients
        self.federated_model_dicts = [{'model': self.global_model, 'round': 0, 
                                       'client_id': 'global', 'type': 'global'}]

        self.federated_model_dicts = []
        self.uploaded_clients = []  # Uploaded client models at t-1 can be downloaded by clients at t

        self.round = 0

        # Use for federated momentum
        last_momentum_weights = copy.deepcopy(
            self.global_model.to(torch.device('cpu'))).state_dict()
        for k in last_momentum_weights.keys():
            last_momentum_weights[k] *= 0  # Initialize to 0
        self.last_momentum_weights = last_momentum_weights
        self.momentum_global_model = copy.deepcopy(self.global_model)

    def initialize_clients(self, epoch, selected_clients=None):
        """
        At the start of each round, initialize active or selected clients
        Args:
        - epoch (int): The federation or communication round
        - selected_clients (Client[]): List of active clients
        """
        print_header(f'> Initalizing clients for round {self.round}...')

        # Initial round, there's nothing to do after initialization
        if self.round == 0:
            return selected_clients  # Everyone locally trains first

        # For FedAvg and round > 0, active clients first download
        # the existing global model from the server
        elif cfg.FEDERATION.FED_AVERAGING:
            for client in selected_clients:
                client.update_model(self.global_model.state_dict())
            return selected_clients

        # Otherwise continue with FedFomo mechanisms
        selected_clients = (self.population.clients if selected_clients is None
                            else selected_clients)

        # Initialize Federation object with all clients
        federation = Federation(clients=selected_clients, federation_id=0,
                                dataset_size_weight=True, cfg=self.cfg, epoch=epoch)

        # Display exploration and decay parameters
        print_header('Epsilon decay parameters:', style='top')
        print_header(f'Starting epsilon: {federation.starting_epsilon:<.4f} | Epsilon decay: {federation.epsilon_decay:<.4f} | Current epsilon: {federation.current_epsilon:<.4f}', style='bottom')

        # Prepare available federated models for download
        federated_models = [FederatedModel(model=m['model'],
                                           client_ids=m['client_id'],
                                           model_type=m['type'])
                            for m in self.federated_model_dicts]
        federation.compute_client_weights(epoch, baseline=None,
                                          federated_models=federated_models,
                                          available_clients=self.uploaded_clients)

        # Initialize new models for active clients
        if args.fed_momentum:
            federation.federate(last_global_model=self.global_model)
        else:
            federation.federate(last_global_model=self.global_model)
        return selected_clients

    def local_train(self, selected_clients, start_epoch, num_epochs=None):
        """
        Locally train the selected clients
        Returns:
        - ending epoch (int): the next global epoch
        """
        print_header(f'> Local training for federation round {self.round}...')
        print(f'>> learning rate: {self.args.lr}')
        if args.federation_method == 'local':
            pass
        else:
            for client in selected_clients:
                client.save_first_model()
        num_epochs = (num_epochs if num_epochs is not None
                      else args.federation_epoch)
        for e in tqdm(range(num_epochs), desc='Local epoch progress'):
            train_stdout = []
            eval_stdout = []
            global_epoch = start_epoch + e
            for client in tqdm(selected_clients, leave=False, desc='Client progress'):
                client.train(epoch=global_epoch, args=self.args)
                val_dataset = client.datasets[2]
                client.eval(epoch=global_epoch, client=client,
                            val_dataset=val_dataset, log=True, args=self.args)
                # Logging
                train_stdout.append(client.train_stdout)
                eval_stdout.append(client.eval_stdout)
                if args.federation_method == 'fomo':
                    if e == 0:
                        client.save_last_model()
                    elif e == args.federation_epoch - 2:
                        client.save_last_model()

            for ix in range(len(train_stdout)):
                if ix == 0:  # Fancier formatting with arguable utility
                    print_header(train_stdout[ix], style='top')
                else:
                    print(train_stdout[ix])
                print_header(eval_stdout[ix], style='bottom')
        print_header(args.experiment_name)
        return start_epoch + args.federation_epoch

    def local_eval(self, selected_clients, epoch):
        """
        Evaluate models of selected clients
        - Most naturally call after performing a federated updated
        - Evaluate on client's test set
        - For additional reference also evaluate on the combined test set
          over all clients (population test set)
        """
        losses = []
        accuracies = []
        for client in tqdm(selected_clients, leave=False):
            client_loss, client_acc, text = client.eval(epoch=epoch,
                                                        client=client,
                                                        val_dataset=client.datasets[2],
                                                        log=False, test=False,
                                                        args=self.args,
                                                        output_metrics=True)
            losses.append(client_loss)
            accuracies.append(client_acc)
            self.client_eval_metrics['val_acc'].append(client_acc)
            self.client_eval_metrics['val_loss'].append(client_loss)
            self.client_eval_metrics['epoch'].append(epoch)
            self.client_eval_metrics['model'].append(f'local_{client.id}')
            self.client_eval_metrics['dataset_distribution'].append(client.dist_id)
            self.client_eval_metrics['target_distribution'].append(client.target_dist_id)
            self.client_eval_metrics['EMD'].append(client.EMD)

            if args.enable_dp:
                epsilon, best_alpha = client.optim.privacy_engine.get_privacy_spent(args.delta)
                self.client_eval_metrics['epsilon'].append(epsilon)
                self.client_eval_metrics['delta'].append(args.delta)
                self.client_eval_metrics['best_alpha'].append(best_alpha)

        global_losses = []
        global_accuracies = []
        dataset = self.population.test_data
        for client in tqdm(selected_clients, leave=False):
            client_loss, client_acc, text = client.eval(epoch=epoch,
                                                        client=client,
                                                        val_dataset=dataset,
                                                        log=False, test=False,
                                                        args=self.args,
                                                        output_metrics=True)
            global_losses.append(client_loss)
            global_accuracies.append(client_acc)

            self.client_eval_metrics['val_acc_global'].append(client_acc)
            self.client_eval_metrics['val_loss_global'].append(client_loss)

        print(f'Experiment name:', args.experiment_name)
        print_header(f'Epoch {epoch} | Local  acc avg: {np.mean(accuracies):<3f} | Local  loss avg: {np.mean(losses):<3f}', style='top')
        print_header(f'Epoch {epoch} | Global acc avg: {np.mean(global_accuracies):<3f} | Global loss avg: {np.mean(global_losses):<3f}', style='bottom')

    def update_server_models(self, selected_clients, all_activated_clients):
        """
        Alternative to setup_federations. Given locally trained models and updated adjacency matrix: 
        (1) Upload trained local models to the server: update self.uploaded_clients
        (2) Based on clients' client_weights, update client-client weight adjacency matrix

        Additionally for hierarchical FedFomo:
        (3) Compute new subfederations / subglobal models from updated matrix
        (4) Based on updated subglobal models, compute new global model
        """
        print_header(f'> Updating server models for round {self.round}...')
        # self.uploaded_clients = copy.deepcopy(selected_clients)
        self.uploaded_clients = selected_clients  # Deepcopy does not seem necessary here
        # Update sub-global models
        if cfg.FEDERATION.METHOD == 'fomo':
            fed_type = f'fomo-{cfg.CLIENT_WEIGHT.METHOD} '
        else:
            fed_type = f'{cfg.FEDERATION.METHOD} '
        print_header(f'>>> Setting up {fed_type}federations for FL round {args.federation_round}...')

        if cfg.FEDERATION.METHOD == 'fedavg':
            # Assign everyone the same federation
            # federation_labels = [0] * len(selected_clients)
            federation_labels = [0] * len(self.population.clients)
            self.clients_per_federation = [[]]
        elif cfg.FEDERATION.METHOD == 'fomo':
            # Additional hierarchical FedFomo
            num_federations = self.num_federations
            client_matrix = self.get_adjacency_matrix(self.population.clients,
                                                      symmetric=True)
            clustering = SpectralClustering(n_clusters=num_federations,
                                            affinity='precomputed',
                                            n_init=10, random_state=args.seed,
                                            assign_labels='discretize')
            federation_labels = clustering.fit_predict(client_matrix)
            self.client_matrix = client_matrix
            self.clients_per_federation = [[] for _ in range(num_federations)]
        else:
            raise NotImplementedError("Federation method must be 'fedavg' or 'fomo'")

        for ix, label in enumerate(federation_labels):
            self.clients_per_federation[label].append(
                self.population.clients[ix])

        self.sub_federations = [Federation(client_list, federation_id=ix) for
                                (ix, client_list) in enumerate(self.clients_per_federation)]

        # Check new federations
        print('-' * (36 + 3 * len(self.population.distributions[0]['clients'])))
        for ix, clients in enumerate(self.clients_per_federation):
            print(f'Clients in federation {ix}:', [c.id for c in clients])
        print('-' * (36 + 3 * len(self.population.distributions[0]['clients'])))
        for ix, dist in enumerate(self.population.distributions):
            print(f'Clients in distribution {ix:<2}:', [c.id for c in dist['clients']])
        print('-' * (36 + 3 * len(self.population.distributions[0]['clients'])))

        if cfg.FEDERATION.FED_AVERAGING:
            if args.fed_momentum:
                momentum_outputs = average_model_params_momentum(selected_clients,
                                                                 self.last_momentum_weights, 
                                                                 self.global_model,
                                                                 nesterov=args.fed_momentum_nesterov)
                global_model_params, momentum_weights = momentum_outputs
                self.last_momentum_weights = momentum_weights
            else:
                total_train_size = np.sum([c.train_size for c in selected_clients])
                global_model_params = average_model_params(clients=selected_clients,
                                                           total_train_size=total_train_size)
            self.global_model = init_federated_model(global_model_params,
                                                     criterion=self.criterion,
                                                     device=None)
            self.federated_model_dicts = [{'model': self.global_model, 'round': self.round,
                                           'client_id': np.array([c.id for c in selected_clients]),
                                           'type': 'global'}]
            return

        # Update server's federated models
        sub_federated_dicts = []
        for subfed in self.sub_federations:
            sub_federated_dicts.append({'model': subfed.model, 'round': self.round, 
                                        'client_id': np.array([c.id for c in subfed.clients]),
                                        'type': 'sub_global'})
        self.federated_model_dicts = sub_federated_dicts

        # Compute global model from sub_federated models
        global_model_params = average_model_params(fed_models=[sf.model for sf in self.sub_federations])
        self.global_model = init_federated_model(global_model_params,
                                                 criterion=self.criterion, device=None)
        self.federated_model_dicts.append({'model': self.global_model, 'round': self.round, 
                                           'client_id': np.array([c.id for c in selected_clients]),
                                           'type': 'global'})

    def get_distance_matrix(self, clients, metric='cosine'):
        """
        Compute pairwise distance matrix of client embeddings
        based on specified metric
        """
        self.embeddings = [client.embedding for client in clients]
        if metric == 'cosine':
            return cosine_distances(self.embeddings, self.embeddings)
        else:
            raise NotImplementedError

    def get_adjacency_matrix(self, clients, symmetric=True, shift_positive=False, 
                             normalize=False, rbf_kernel=True, rbf_delta=1., 
                             softmax_client_weights=args.softmax_client_weights):
        """
        If learning federations through updated client weight preferences,
        first compute adjacency matrix given client-to-client weights
        Args:
        - clients (Clients[]): list of clients <-- should be the population.clients array
        - symmetric (bool): return a symmetric matrix
        - shift_positive (bool): shift all values to be >= 0 (add (0 - lowest value) to everything)
        - normalize (bool): after other transformations, normalize everything to range [0, 1]
        - rbf_kernel (bool): apply Gaussian (RBF) kernel to the matrix
        """
        if softmax_client_weights:
            matrix = []
            for client in clients:
                matrix.append(np.exp(client.client_weights) / 
                              np.sum(np.exp(client.client_weights)))
            matrix = np.array(matrix)
        else:
            matrix = np.array([client.client_weights for client in clients])
            matrix = 1. - matrix  # Affinity matrix is reversed -> lower value = better
            if rbf_kernel:
                matrix = np.exp(-1. * matrix ** 2 / (2. * rbf_delta ** 2))
        if symmetric:
            matrix = (matrix + matrix.T) * 0.5
        if shift_positive:
            if np.min(matrix) < 0:
                matrix = matrix + (0 - np.min(matrix))
        if normalize:
            matrix = matrix / (np.max(matrix) - np.min(matrix))
        return matrix

    def update_eval_metrics(self, accuracy, loss, epoch,
                            dataset_distribution, model_name):
        """
        Call this to update eval metrics
        """
        self.eval_metrics['val_acc'].append(accuracy)
        self.eval_metrics['val_loss'].append(loss)
        self.eval_metrics['epoch'].append(epoch)
        self.eval_metrics['model'].append(model_name)
        self.eval_metrics['dataset_distribution'].append(dataset_distribution)
