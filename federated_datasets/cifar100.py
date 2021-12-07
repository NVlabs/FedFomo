"""
Federated CIFAR-100 dataset and client population class
"""

import pickle
import copy
import numpy as np

from os.path import join, exists
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, random_split

import torchvision
import torchvision.transforms as transforms

from federated.configs import cfg_fl as cfg
from federated.args import args
from federated.utils import *
from federated_datasets import shuffle_client_targets
from federated_datasets.latent_dist import get_embeddings

from client import Client


class Population():
    def __init__(self,
                 cluster_method=cfg.FEDERATION.CLUSTERING_METHOD,
                 num_distributions=cfg.FEDERATION.NUM_DISTRIBUTIONS,
                 num_clients=cfg.FEDERATION.NUM_CLIENTS,
                 num_clients_per_dist=cfg.FEDERATION.CLIENTS_PER_DIST,
                 precomputed=True,
                 embedding_fname='embeddings-d=cifar100-e=23-st=5-lr=0.001-mo=0.9-wd=0.0001-fc2-train.npy',
                 test_embedding_fname='embeddings-d=cifar100-e=23-st=5-lr=0.001-mo=0.9-wd=0.0001-fc2-test.npy',
                 shuffle=True):

        self.dataset_root = './data/cifar100'

        self.distributions = []
        self.cluster_method = cluster_method
        self.num_distributions = num_distributions
        self.num_clients_per_dist = num_clients_per_dist

        self.train_datasets = None

        self.num_classes = 100
        args.num_classes = 100

        self.num_clients = (self.num_distributions * self.num_clients_per_dist
                            if args.num_clients is None
                            else num_clients)

        self.num_clients_per_dist = (self.num_clients_per_dist if args.clients_per_dist is not None else 
                                     int(self.num_clients / self.num_distributions))

        print(f'number distributions: {self.num_distributions}')
        print(f'num clients per dist: {self.num_clients_per_dist}')

        print(f'num clients: {self.num_clients}')
        self.clients = [None] * self.num_clients

        self.precomputed = precomputed
        self.precomputed_root = cfg.DATASET.PRECOMPUTED_DIR

        # Pretrained embeddings
        self.embedding_fname = embedding_fname
        self.train_embedding_fname = embedding_fname
        self.test_embedding_fname = test_embedding_fname

        kmeans_labels_prefix = f'kmeans-nd={args.num_distributions}-s={args.seed}-ds={args.data_seed}'
        self.kmeans_train_fname = f'{kmeans_labels_prefix}-{self.embedding_fname}'
        self.kmeans_test_fname = f'{kmeans_labels_prefix}-{self.test_embedding_fname}'

        self.random_dists = cfg.FEDERATION.RANDOM_DISTS

        self.load_dataset()

        if args.shards_per_user is not None:  # If shard_per_user is specified, separate at client-level
            self.init_clients_by_shard()
        else:
            self.init_distributions(shuffle=shuffle)
            self.init_clients()
            if args.shuffle_targets:
                # shuffle client targets
                print(f'> Shuffling client target distributions...')
                shuffle_client_targets(self.clients)
                for c in self.clients[:10]:
                    print(f'Client {c.id} dist id {c.dist_id} -> target id {c.target_dist_id}')

    def load_dataset(self):
        """
        Load CIFAR-100 dataset from torchvision.datasets.CIFAR10
        """
        # ImageNet normalization constants
        imagenet_normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))
        
        self.setup_transform = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   imagenet_normalize])

        self.setup_test_transform = transforms.Compose([transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        imagenet_normalize])

        if args.arch in ['tf_cnn', 'base_cnn']:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            imagenet_normalize])
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 imagenet_normalize])
            self.transform = transform
            self.test_transform = test_transform
        else:
            self.transform = self.setup_transform
            self.test_transform = self.setup_test_transform

        self.train_data = torchvision.datasets.CIFAR100(root=self.dataset_root, train=True,
                                                        download=True, transform=self.transform)

        self.test_data = torchvision.datasets.CIFAR100(root=self.dataset_root, train=False,
                                                       download=True, transform=self.test_transform)

    def init_clients_by_shard(self):
        dict_users_train, rand_set_all = client_noniid(self.train_data, args.num_clients, args.shards_per_user,
                                                       seed=args.data_seed)
        dict_users_test, rand_set_all = client_noniid(self.test_data, args.num_clients, args.shards_per_user,
                                                      rand_set_all=rand_set_all, seed=args.data_seed)

        self.distributions.append({'images': None,  # placeholder
                                   'labels': None,
                                   'clients': [],
                                   'id': -1,
                                   'indices': None,
                                   'test_labels': None,
                                   'test_indices': None})

        for ix in dict_users_train:
            client = Client(client_id=ix)
            client.local_val_ratio = args.local_val_ratio
            client.shared_val = True

            client_dataset = torch.utils.data.Subset(self.train_data, indices=dict_users_train[ix])
            client_test_dataset = torch.utils.data.Subset(self.test_data, indices=dict_users_test[ix])

            client.population = self

            client.initialize(client_dataset, self.distributions[0], client_test_dataset)
            client.targets = [self.train_data.targets[x] for x in dict_users_train[ix]]
            client.unique_classes = np.unique(client.targets)

            client.dataset_train_indices = dict_users_train[ix]
            train_split_size = int(np.round(len(dict_users_train[ix]) * args.train_split))
            # Set up client indices specifically for local training set
            client.local_train_indices = np.random.choice(client.dataset_train_indices, 
                                                          size=train_split_size, replace=False)
            client.dataset_test_indices = dict_users_test[ix]

            self.clients[client.id] = client
            self.distributions[0]['clients'].append(client)
            print(f'Loaded client {self.clients[client.id].id} data!')

        average_emd = np.mean([compute_emd(client.targets, self.train_data.targets) for client in self.clients])
        print_header(f'> Global mean EMD: {average_emd}')
        for client in self.clients:
            client.EMD = average_emd

    def init_distributions(self, shuffle):
        """
        Initialize client data distributions with through latent non-IID method
        - Groups datapoints into D groupings based on clustering their
          hidden-layer representations from a pre-trained model
        """
        dict_data = {'inputs': np.array(self.train_data.data),
                     'targets': np.array(self.train_data.targets),
                     'test_inputs': np.array(self.test_data.data),
                     'test_targets': np.array(self.test_data.targets)}

        try:  # First try to load pre-computed data
            path = join(self.precomputed_root, self.train_embedding_fname)
            print(f'> Loading training embeddings from {path}...')
            with open(path, 'rb') as f:
                dict_data['train_embeddings'] = np.load(f)

            path = join(self.precomputed_root, self.test_embedding_fname)
            print(f'> Loading test embeddings from {path}...')
            with open(path, 'rb') as f:
                dict_data['test_embeddings'] = np.load(f)
        except FileNotFoundError:
            print(f'>> Embedding path not found. Calculating new embeddings...')
            setup_train_data = torchvision.datasets.CIFAR100(root=self.dataset_root, 
                                                            train=True,
                                                            download=False,
                                                            transform=self.setup_transform)
            setup_test_data = torchvision.datasets.CIFAR100(root=self.dataset_root, 
                                                           train=False,
                                                           download=False,
                                                           transform=self.setup_test_transform)
            all_embeddings = get_embeddings(setup_train_data,
                                            setup_test_data,
                                            num_epochs=100,  # 10,
                                            args=args,
                                            stopping_threshold=5)
            
            train_embeddings, test_embeddings = all_embeddings
            dict_data['train_embeddings'] = train_embeddings
            dict_data['test_embeddings'] = test_embeddings

        if args.pathological_non_iid:
            """
            Randomly allocate classes to n distributions as equally as possible
            """
            dist_labels = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                dist_labels[i] = i % self.num_distributions
            np.random.seed(args.data_seed)  # Shuffle them 
            np.random.shuffle(dist_labels)
            kmeans_labels = np.array([dist_labels[t] for t in dict_data['targets']])
            kmeans_labels_test = np.array([dist_labels[t] for t in dict_data['test_targets']])
        else:
            if self.random_dists:
                print('> Random dataset distribution initialization...')
                kmeans_labels = np.random.randint(self.num_distributions, size=len(dict_data['inputs']))
                kmeans_labels_test = np.random.randint(self.num_distributions, size=len(dict_data['test_inputs']))

            else:
                try:  # First see if kmeans labels were already computed, and load those
                    path = join(self.precomputed_root, self.kmeans_train_fname)
                    with open(path, 'rb') as f:
                        kmeans_labels = np.load(path)
                    path = join(self.precomputed_root, self.kmeans_test_fname)
                    print(f'> Loaded clustered labels from {path}!')
                    with open(path, 'rb') as f:
                        kmeans_labels_test = np.load(path)
                    print(f'> Loaded clustered labels from {path}!')
                except FileNotFoundError:
                    # Compute PCA first on combined train and test embeddings
                    embeddings = np.concatenate([dict_data['train_embeddings'], 
                                                 dict_data['test_embeddings']])
                    np.random.seed(args.data_seed)
                    num_components = 256  # 32
                    pca = PCA(n_components=num_components)
                    dict_data['embeddings'] = pca.fit_transform(embeddings)
                    print_debug(pca.explained_variance_ratio_.cumsum()[-1], f'Ratio of embedding variance explained by {num_components} principal components')

                    # Compute clusters
                    np.random.seed(args.data_seed)
                    km = KMeans(n_clusters=self.num_distributions, init='k-means++', 
                                max_iter=100, n_init=5)
                    # For random distributions, specify km.labels_ randomly across the number of distributions
                    if self.random_dists:
                        km.labels_ = np.random.randint(self.num_distributions, size=len(embeddings))  
                        print_debug(len(dict_data['inputs']), 'Size of dataset')
                    else:
                        km.fit(dict_data['embeddings'])

                    kmeans_labels_train_test = km.labels_
                    # Partition into train and test
                    kmeans_labels = kmeans_labels_train_test[:len(self.train_data.targets)]
                    kmeans_labels_test = kmeans_labels_train_test[len(self.train_data.targets):]

                    assert len(kmeans_labels) == len(dict_data['inputs'])
                    print_debug(len(kmeans_labels_test), 'len(kmeans_labels_test)')
                    print_debug(len(dict_data['test_inputs']), "len(dict_data['test_inputs'])")
                    assert len(kmeans_labels_test) == len(dict_data['test_inputs'])

                    path = join(self.precomputed_root, self.kmeans_train_fname)
                    with open(path, 'wb') as f:
                        np.save(f, kmeans_labels)
                    print(f'> Saved clustered labels to {path}!')

                    path = join(self.precomputed_root, self.kmeans_test_fname)
                    with open(path, 'wb') as f:
                        np.save(f, kmeans_labels_test)
                    print(f'> Saved clustered labels to {path}!')

        loaded_images = dict_data['inputs']
        loaded_labels = dict_data['targets']

        loaded_images_test = dict_data['test_inputs']
        loaded_labels_test = dict_data['test_targets']

        for cluster_label in range(self.num_distributions):
            indices = np.where(kmeans_labels == cluster_label)[0]
            images_dist = loaded_images[indices]
            labels_dist = loaded_labels[indices]

            if shuffle:
                np.random.seed(args.data_seed)
                shuffle_ix = list(range(images_dist.shape[0]))
                np.random.shuffle(shuffle_ix)
                images_dist = images_dist[shuffle_ix]
                labels_dist = labels_dist[shuffle_ix]
                indices = indices[shuffle_ix]

            test_indices = np.where(kmeans_labels_test == cluster_label)[0]
            test_images_dist = loaded_images_test[test_indices]
            test_labels_dist = loaded_labels_test[test_indices]

            if shuffle:
                # np.random.seed(cfg.SEED)
                np.random.seed(args.data_seed)
                shuffle_ix = list(range(test_images_dist.shape[0]))
                np.random.shuffle(shuffle_ix)
                test_images_dist = test_images_dist[shuffle_ix]
                test_labels_dist = test_labels_dist[shuffle_ix]
                test_indices = test_indices[shuffle_ix]

            # Should be good if the embeddings were calculated in order
            self.distributions.append({'images': images_dist,
                                       'labels': labels_dist,
                                       'clients': [],
                                       'id': cluster_label,
                                       'indices': indices,
                                       'test_labels': test_labels_dist,
                                       'test_indices': test_indices})

        for d in range(len(self.distributions)):
            dist_dict = self.distributions[d]
            labels = dist_dict['labels']
            print_debug(labels.shape, f'Distribution {d} labels shape')
            print_debug(dist_dict['test_labels'].shape, f'Distribution {d} test labels shape')

    def init_clients(self):
        print_header('> Initializing clients...')

        print(f'>> Total clients: {self.num_clients}')
        print(f'>> Clients per distribution: {self.num_clients_per_dist}')

        ix = 0
        dist_ix = 0
        # Fill in clients
        if cfg.CLIENT.MANUAL:
            for client_params in cfg.CLIENT.POPULATION:
                client = Client(client_id=client_params['client_id'])
                client.local_val_ratio = client_params['lvr']
                client.shared_val = client_params['shared_val']
                self.distributions[client_params['dist_id']]['clients'].append(client)
        else:
            while ix < self.num_clients and dist_ix < len(self.distributions):
                client = Client(client_id=ix)
                client.local_val_ratio = args.local_val_ratio
                client.shared_val = True
                self.distributions[dist_ix]['clients'].append(client)
                if (ix + 1) % self.num_clients_per_dist == 0:
                    dist_ix += 1
                ix += 1
            for i in range(ix, self.num_clients):
                client = Client(client_id=i)
                client.local_val_ratio = args.local_val_ratio
                client.shared_val = True
                print_debug(i % self.num_distributions, 'client mod')
                self.distributions[i % self.num_distributions]['clients'].append(client)

        if args.num_adversaries > 0:  # Randomly allocate adversaries, 1 per distribution at first
            count = 0
            for i in range(self.num_clients_per_dist):
                for j in range(self.num_distributions):
                    if count < args.num_adversaries:
                        self.distributions[j]['clients'][i].adversarial = True
                        count += 1

        # Progress bar for initializing clients
        tqdm_clients = tqdm(total=self.num_clients)
        np.random.seed(args.data_seed)
        for ix, dist in enumerate(self.distributions):
            # Already shuffled during initialization, but can do again if desired
            np.random.seed(args.data_seed)
            shuffle_ix = list(range(len(self.distributions[ix]['images'])))
            np.random.shuffle(shuffle_ix)

            images = dist['images'][shuffle_ix]
            labels = dist['labels'][shuffle_ix]
            indices = dist['indices'][shuffle_ix]

            # Transpose images and numpy dims
            print_debug(dist['images'].shape, "dist['images'].shape")
            images = np.expand_dims(images, 1)  # Add dimension for single channel

            data_by_clients = np.array_split(images, len(dist['clients']))
            labels_by_clients = np.array_split(labels, len(dist['clients']))
            indices_by_clients = np.array_split(indices, len(dist['clients']))

            # Do the same for test set
            test_indices = dist['test_indices']
            test_indices_by_clients = np.array_split(test_indices, len(dist['clients']))

            # Initialize clients
            for cix, client in enumerate((dist['clients'])):
                client.population = self   # Give access to the population for each client

                # Setup data
                assert data_by_clients[cix].shape[0] == labels_by_clients[cix].shape[0]

                # Setup total local dataset for each client
                client_dataset = torch.utils.data.Subset(self.train_data, indices=indices_by_clients[cix])
                client_test_dataset = torch.utils.data.Subset(self.test_data, indices=test_indices_by_clients[cix])

                client.initialize(client_dataset, dist, client_test_dataset)
                client.targets = [self.train_data.targets[x] for x in indices_by_clients[cix]]
                client.unique_classes = np.unique(client.targets)

                self.clients[client.id] = client  # Give another reference to the client
                tqdm_clients.update(n=1)

                client.dataset_train_indices = indices_by_clients[cix]
                train_split_size = int(np.round(len(indices_by_clients[cix]) * args.train_split))
                # Set up client indices specifically for local training set
                client.local_train_indices = np.random.choice(client.dataset_train_indices, 
                                                              size=train_split_size, replace=False)
                client.dataset_test_indices = test_indices_by_clients[cix]

        tqdm_clients.close()

        if args.parallelize:
            pass
        else:
            # Compute EMD
            average_emd = np.mean([compute_emd(client.targets, self.train_data.targets) for client in self.clients])
            print_header(f'> Global mean EMD: {average_emd}')
            for client in self.clients:
                client.EMD = average_emd
            if args.num_distributions > 1:
                self.finalize_client_datasets()

        print_debug([f'Dist: {c.dist_id}, id: {c.id}, adversarial: {c.adversarial}' for c in self.clients])

    def finalize_client_datasets(self):
        """
        For each client, prepare datasets with additional distribution mixing
        - Based on args.local_train_dist_ratio and args.local_val_dist_ratio, can mix datasets accordingly
        """
        train_data_by_dists = [[] * self.num_distributions]
        val_data_by_dists = [[] * self.num_distributions]

        pooled_train_data = []
        pooled_val_data = []

        # Ratio of client's local train set and val set that should be shared
        train_mixing_split_ratio = ((1. - args.local_train_dist_ratio) / 
                                    (args.local_train_dist_ratio * (self.num_clients - 1) - self.num_clients_per_dist + 1))
        val_mixing_split_ratio = ((1. - args.local_val_dist_ratio) / 
                                  (args.local_val_dist_ratio * (self.num_clients - 1) - self.num_clients_per_dist + 1))

        # Make sure everyone has the same val size for even mixing and evaluation
        min_val_size = int(np.min([len(client.datasets[1].indices) 
                                   for client in self.clients]))

        min_train_size = int(np.min([len(client.datasets[0].indices) 
                                     for client in self.clients]))

        # Allocate portions of each client's train and val splits for mixing
        torch.manual_seed(args.data_seed)
        np.random.seed(args.data_seed)

        for client in self.clients:
            if args.num_local_val_pooled > 1:
                pooled_data_len = int(args.num_local_val_pooled)
            else:
                pooled_data_len = int(len(client.datasets[1]) * args.num_local_val_pooled)
            saved_data_len = len(client.datasets[1]) - pooled_data_len

            # Split client's original validation split into split to keep and split to share
            pooled_data_split = random_split(client.datasets[1], [saved_data_len, pooled_data_len])
            # Save the split to share
            pooled_val_data.append(pooled_data_split[1])
            # Among the split to keep, further organize into a ratio to hold onto
            # If lvr == 0., then we should have full mixing between distributions
            saved_data_len = int(len(pooled_data_split[0]) * client.local_val_ratio)

            datasets = random_split(pooled_data_split[0], 
                                    [saved_data_len, len(pooled_data_split[0]) - saved_data_len])
            client.datasets[1] = datasets[0]

        # Then add the pooled data back to each client
        for ix, client in enumerate(self.clients):
            if args.local_val_model_delta_ratio:
                client.datasets.append(ConcatDataset(pooled_val_data))
                
            else:
                client_val_set = [client.datasets[1]]

                all_data_counts = [len(x.indices) for x in pooled_val_data]

                if client.shared_val:
                    client_val_set.extend(pooled_val_data)

                client.datasets[1] = ConcatDataset(client_val_set)

        print_header('Client finalized dataset sizes')
        for ix, dist in enumerate(self.distributions):
            for cix, client in enumerate(dist['clients']):
                print(f'Dist {ix} client {cix} dataset size - train: {len(client.datasets[0])}, val: {len(client.datasets[1])}, test: {len(client.datasets[2])}')
            print('')
