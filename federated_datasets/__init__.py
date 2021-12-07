"""
Federated datset setup and loaders
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from federated.args import args


def load_federated_dataset(dataset, train=True):
    """
    Get the actual dataloader given a dataset
    - Based on ADLR semantic-segmentation/datasets.setup_loaders()
    Inputs:
    - dataset (torch.data.Dataset): A (client's) dataset
    - train (bool): Whether to load with training params or not
    Output:
    - The dataloader for the dataset
    """
    if args.apex:
        from datasets.sampler import DistributedSampler
        # pad and permutation are True for training set, False for validation
        sampler = DistributedSampler(dataset,
                                     pad=train,
                                     permutation=train,
                                     consecutive_sample=False)
        train_batch_size = args.bs_trn
    elif args.parallelize:  # imagenet - 8/18
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    train_batch_size = args.bs_trn

    batch_size = train_batch_size if train else args.bs_val
    shuffle = (sampler is None) if train else False
    drop_last = True if args.enable_dp else False
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers // 2,
                            shuffle=shuffle, drop_last=drop_last, sampler=sampler)
    return dataloader


def shuffle_client_targets(clients):
    """
    Call if args.shuffle_targets is specified.
    Given all clients in a population, shuffles their val and test splits randomly
    This also updates client.dataset_test_indices, but not client.local_train_indices
    Finally also changes the client.target_dist_id attribute
    """
    client_ids = [c.id for c in clients]
    np.random.seed(args.seed)
    np.random.shuffle(client_ids)
    # Now reassign the datasets and test indices
    for ix, client in enumerate(clients):
        client.target_dist_id = clients[client_ids[ix]].dist_id
        # Update local eval splits
        client.temp_val_set = clients[client_ids[ix]].datasets[1]
        client.temp_test_set = clients[client_ids[ix]].datasets[2]
        client.temp_dataset_test_indices = clients[client_ids[ix]].dataset_test_indices
    # Actually update once everyone assigned
    for client in clients:
        client.datasets[1] = client.temp_val_set
        client.datasets[2] = client.temp_test_set
        client.dataset_test_indices = client.temp_dataset_test_indices


def client_noniid(dataset, num_users, shard_per_user, rand_set_all=[], seed=args.data_seed):
    """
    Sample non-IID client data from dataset in pathological manner - from LG-FedAvg implementation
    :param dataset:
    :param num_users:
    :return: (dictionary, where keys = client_id / index, and values are dataset indices), rand_set_all (all classes)

    shard_per_user should be a factor of the dataset size
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    np.random.seed(seed)
    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        np.random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # Divide and assign
    np.random.seed(seed)
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all