"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property 
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an 
express license agreement from NVIDIA CORPORATION is strictly prohibited.
"""


import os
import sys
import time
import pickle
import copy
import importlib
import pandas as pd

import torch
import torch.multiprocessing as mp

import federated

from server import GlobalServer
from federated.args import args
from federated.utils import *
from federated.configs import cfg_fl as cfg, assert_and_infer_cfg_fl
from federated.eval import *


def save_client_metrics(clients, training_metrics_path):
    df_clients = []
    for client in clients:
        df_clients.append(pd.DataFrame(client.metrics))

    # Save client data
    pd.concat(df_clients, ignore_index=True).to_csv(training_metrics_path, index=False)
    del df_clients
    torch.cuda.empty_cache()
    print(f'Saved training metrics to {training_metrics_path}!')


# Enable CUDNN Benchmarking optimization
if args.deterministic:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    """
    Run federated learning
    """

    cfg = assert_and_infer_cfg_fl(cfg, args, make_immutable=False)
    args.ngpu = torch.cuda.device_count()
    
    args.device = torch.device(f'cuda:{args.device}') if args.device is not None and torch.cuda.is_available else torch.device('cpu')

    print(f'> Num clients per update (if e-greedy): {cfg.CLIENT_WEIGHT.NUM_UPDATE_CLIENTS}')
    print(f'> Training method: {cfg.FEDERATION.METHOD}')


    if cfg.FEDERATION.METHOD == 'fomo':
        if cfg.CLIENT_WEIGHT.METHOD == 'e_greedy':
            cm = 'eg'
        elif cfg.CLIENT_WEIGHT.METHOD == 'sub_federations':
            cm = 'sf'
        if cfg.CLIENT_WEIGHT.BASELINE == 'first_model':
            baseline_name = 'fm'
        elif cfg.CLIENT_WEIGHT.BASELINE == 'last_model':
            baseline_name = 'lm'
        elif cfg.CLIENT_WEIGHT.BASELINE == 'model_avg':
            baseline_name = 'ma'
        experiment_params = f'-fe={cfg.FEDERATION.EPOCH}-cm={cm}-b={baseline_name}-wd={cfg.CLIENT_WEIGHT.WEIGHT_DELTA}-nuc={cfg.CLIENT_WEIGHT.NUM_UPDATE_CLIENTS}-we={cfg.CLIENT_WEIGHT.EPSILON}-wed={cfg.CLIENT_WEIGHT.EPSILON_DECAY}-scw={args.softmax_client_weights}'

        if args.softmax_model_deltas:
            experiment_params += '-sd'

        if args.num_federations is not None:
            experiment_params += f'-nf={args.num_federations}'

        if args.fedavg_rounds is not None:
            experiment_params += f'-far={args.fedavg_rounds}'

        if args.local_rounds is not None:
            experiment_params += f'-lar={args.local_rounds}'

    elif cfg.FEDERATION.METHOD == 'fedavg':
        experiment_params = f'-fe={cfg.FEDERATION.EPOCH}'

    elif cfg.FEDERATION.METHOD == 'local':
        experiment_params = ''

    if cfg.CLIENT.MANUAL:
        lvr = 'man0'
    else:
        lvr = args.local_val_ratio

    train_curve = 'g' if args.global_training_curve else 'l'

    if cfg.TASK == 'semantic_segmentation':
        task = 'segm'
    elif cfg.TASK == 'classification':
        task = 'class'

    if args.num_adversaries > 0:
        experiment_params += f'-na={args.num_adversaries}'

    clients_arg = f'nc={args.num_clients}' if args.num_clients else f'cpd={cfg.FEDERATION.CLIENTS_PER_DIST}'
    distributions_arg = f'su={args.shards_per_user}' if args.shards_per_user else f'nd={cfg.FEDERATION.NUM_DISTRIBUTIONS}'

    ltvs = f'ltvs={args.local_train_val_size:}-' if args.local_train_val_size else ''
    num_local_pooled = ''
    if args.num_local_val_pooled > 0:
        num_local_pooled += f'nlvp={args.num_local_val_pooled}-'
    if args.num_local_train_pooled > 0:
        num_local_pooled += f'nltp={args.num_local_train_pooled}-'
    args.experiment_name = f'm={cfg.FEDERATION.METHOD}-d={cfg.DATASET.DATASET_NAME}-{distributions_arg}-{clients_arg}-rd={cfg.FEDERATION.RANDOM_DISTS}-ts={cfg.CLIENT.TRAIN_SPLIT}-{ltvs}me={args.max_epoch}-arch={args.arch}-lr={args.lr}-lrd={args.learning_rate_decay}-mo={args.momentum}-o={args.optimizer}-bst={args.bs_trn}-bsv={args.bs_val}{experiment_params}-ds={args.data_seed}-s={cfg.SEED}-r={args.replicate}'

    if args.eval_distribution_test:
        args.experiment_name = args.experiment_name + '-edt'

    if args.local_val_model_delta_ratio:
        args.experiment_name = args.experiment_name + f'-lvmdr={args.local_val_model_delta_ratio}'

    if args.eval_by_class:
        args.experiment_name = args.experiment_name + '-ebc'

    if args.local_upper_bound:
        args.experiment_name = args.experiment_name + '-lub'

    if args.pathological_non_iid:
        args.experiment_name = args.experiment_name + '-pniid'

    if args.shuffle_targets:
        args.experiment_name = args.experiment_name + '-st'

    if args.fedprox:
        if args.fedprox_mu is not None:
            args.experiment_name = args.experiment_name + f'-fp{args.fedprox_mu}'

    if args.fed_momentum:
        args.experiment_name = args.experiment_name + f'-fmg={args.fed_momentum_gamma}'
        if args.fed_momentum_nesterov:
            args.experiment_name += f'-nag'

    if args.enable_dp:
        args.experiment_name += f'-na={args.n_accumulation_steps}-sd={args.sigma}-C={args.max_per_sample_grad_norm}-d={args.delta}'
        if args.virtual_batch_rate is not None:
            args.experiment_name += f'-vbr={args.virtual_batch_rate}'
            args.bs_trn_v = args.bs_trn
            args.bs_val_v = args.bs_val
            args.bs_trn = int(args.bs_trn / args.virtual_batch_rate)
            args.bs_val = int(args.bs_val / args.virtual_batch_rate)
            print(f'Virtual train batch size: {args.bs_trn_v} | Virtual val batch size: {args.bs_val_v}')
            print(f'Train batch size: {args.bs_trn} | Val batch size: {args.bs_val}')

    # Save model paths
    args.model_path = os.path.join(cfg.MODEL_DIR, cfg.DATASET.DATASET_NAME)
    try:
        os.mkdir(args.model_path)
    except FileExistsError:
        pass

    args.model_path = os.path.join(args.model_path, f'replicate-{args.replicate}')
    try:
        os.mkdir(args.model_path)
    except FileExistsError:
        pass

    # Saving results
    print(f'> Save name: {args.experiment_name}')
    save_dir = os.path.join(cfg.RESULTS_DIR, f'replicate-{args.replicate}')

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass

    print(f'> Saving files to {save_dir}...')
    # Finally log them:
    sys.stdout = Logger(args=args)
    summarize_args(args=args, as_script=True)

    training_metrics_path = os.path.join(save_dir, f'client_train-{args.experiment_name}.csv')
    test_metrics_path = os.path.join(save_dir, f'client_test-{args.experiment_name}.csv')
    embeddings_path = os.path.join(save_dir, f'server_embeddings-{args.experiment_name}.csv')

    print_header(f'>>> Number of GPUs available: {args.ngpu}')

    # Modularity among datasets
    dataset_module = importlib.import_module('federated_datasets.{}'.format(cfg.DATASET.DATASET_NAME))

    if cfg.TASK == 'classification':
        population = getattr(dataset_module, 'Population')()
    else:
        raise NotImplementedError

    # Initialize global server
    if cfg.FEDERATION.FED_AVERAGING:
        server = GlobalServer(population=population, num_federations=1)
    elif args.federation_method == 'fomo' and args.num_federations is not None:
        server = GlobalServer(population=population, num_federations=args.num_federations)
    else:
        server = GlobalServer(population=population)

    server.cfg = cfg  # Pass asserted and inferred configs to server

    evals_setup = False
    # If True, show performance on the global in-distribution test set during training
    if args.eval_distribution_test:
        setup_eval_datasets(population, limit_train_size=False)
        evals_setup = True
        for ix, dist in enumerate(population.distributions):
            for cix, client in enumerate(dist['clients']):
                client.datasets[2] = population.test_datasets[ix]  # Assign the distribution test set to client's test set

    if args.local_upper_bound and args.federation_method == 'local':
        if evals_setup:
            pass
        else:
            setup_eval_datasets(population, limit_train_size=False)

        for ix, dist in enumerate(population.distributions):
            for cix, client in enumerate(dist['clients']):
                client.datasets[0] = population.train_datasets[ix]

    # Evaluation
    if args.evaluate:
        # Evaluate federated models
        if evals_setup:
            pass
        else:
            setup_eval_datasets(population, limit_train_size=False)
        args.federation_round = -1
        evaluate_federated_models(server)
        federated_metrics_path = os.path.join(save_dir, f'fed_test-{args.experiment_name}-elfs={args.eval_local_finetune_size}.csv')
        pd.DataFrame(server.eval_metrics).to_csv(federated_metrics_path, index=False)
        print(f'Saved federated test metrics to {federated_metrics_path}!')
        sys.exit()

    ###################
    # Actual Training #
    ###################

    # Save client-to-client weight matrices
    client_weight_matrices = []

    all_activated_clients = []

    np.random.seed(cfg.SEED)  # stan reproducibility
    for epoch in range(args.max_epoch):
        server.round = epoch
        args.federation_round = epoch

        # First round, everyone locally train
        if epoch == 0 and args.federating_ratio < 1 and args.federation_method == 'fomo':
            print('> First round initializing all client models...')
            server.local_eval([population.clients[0]], epoch * args.federation_epoch)  # Evaluate the models at the start
            server.local_train(population.clients, epoch * args.federation_epoch, num_epochs=1)  # Train and record fine-tuned number
            # Randomly select subset to upload to server
            m = max(int(args.federating_ratio * population.num_clients), 1)
            client_indices = np.random.choice(range(population.num_clients), m, replace=False)
            federating_clients = [population.clients[ix] for ix in client_indices]
            server.uploaded_clients = copy.deepcopy(federating_clients)
            continue
        else:
            print('> Selecting subset of active models...')
            np.random.seed(cfg.SEED)
            m = max(int(args.federating_ratio * population.num_clients), 1)
            client_indices = np.random.choice(range(population.num_clients), m, replace=False)

        if args.debugging:
            print_debug(client_indices, 'selected clients')

        if args.fedavg_rounds is not None and epoch < args.fedavg_rounds:
            args.federation_method = 'fedavg'
            cfg.FEDERATION.METHOD = 'fedavg'
            cfg.FEDERATION.FED_AVERAGING = True
        elif args.fedavg_rounds is not None and epoch >= args.fedavg_rounds:
            args.federation_method = 'fomo'
            cfg.FEDERATION.METHOD = 'fomo'
            cfg.FEDERATION.FED_AVERAGING = False

        if args.local_rounds is not None and epoch < args.local_rounds:
            args.federation_method = 'local'
            cfg.FEDERATION.METHOD = 'local'
            cfg.FEDERATION.FED_AVERAGING = False
        elif args.local_rounds is not None and epoch >= args.local_rounds:
            args.federation_method = 'fomo'
            cfg.FEDERATION.METHOD = 'fomo'
            cfg.FEDERATION.FED_AVERAGING = False

        if cfg.FEDERATION.METHOD == 'local' and args.debugging:
            client_indices = sorted(client_indices)

        federating_clients = [population.clients[ix] for ix in client_indices]

        print('Federating Clients:', [f.id for f in federating_clients])

        server.last_active_clients = federating_clients
        for client in federating_clients:
            client.last_active_round = epoch  # Update last active round
            if client not in all_activated_clients:  # <- 8/6, not sure how useful all activated clients should be
                all_activated_clients.append(client)

        if cfg.FEDERATION.METHOD == 'local':
            server.local_eval(federating_clients, epoch * args.federation_epoch)  # Evaluate the models at the start
            server.local_train(federating_clients, epoch * args.federation_epoch)
            save_client_metrics(population.clients, training_metrics_path)
            pd.DataFrame(server.client_eval_metrics).to_csv(test_metrics_path, index=False)

            if args.debugging:
                sys.exit()  # Early stop for debugging
            # Decay learning rate
            server.args.lr = np.max([args.min_learning_rate, server.args.lr * args.learning_rate_decay])

            if args.local_rounds is not None and epoch == args.local_rounds - 1:
                server.uploaded_clients = copy.deepcopy(federating_clients)  # Prepare for next round
            continue

        print_header('****~~Federating part~~****')
        federating_clients = server.initialize_clients(epoch, selected_clients=federating_clients)

        for client in federating_clients:
            client.save_first_model()

        server.local_eval(federating_clients, epoch * args.federation_epoch)  # Evaluate the models at the start
        server.local_train(federating_clients, epoch * args.federation_epoch)  # Train and record fine-tuned number

        # Make sure server.population.clients is updated to match federating_clients
        for federating_client_ix, population_client_ix in enumerate(client_indices):
            server.population.clients[population_client_ix] = federating_clients[federating_client_ix]

        for client in federating_clients:
            client.participated = True

            # Save models after local training - use to analyze the divergence
            torch.save(client.model.state_dict(), f'./models/m_c{client.id}_d{client.dist_id}_e{epoch}-{args.experiment_name}.pt')

        server.update_server_models(federating_clients, all_activated_clients)

        # Save data
        save_client_metrics(population.clients, training_metrics_path)
        pd.DataFrame(server.client_eval_metrics).to_csv(test_metrics_path, index=False)

        if cfg.FEDERATION.METHOD == 'fomo':
            client_weight_matrices.append(server.client_matrix)
            client_weights_over_time = {'client': [], 'weights': [], 'distribution': []}
            model_weights_over_time = {'client': [], 'weights': [], 'distribution': []}
            for client in population.clients:
                client_weights_over_time['client'].append(client.id)
                client_weights_over_time['distribution'].append(client.dist_id)
                client_weights_over_time['weights'].append(client.client_weights_over_time)

                model_weights_over_time['client'].append(client.id)
                model_weights_over_time['distribution'].append(client.dist_id)
                model_weights_over_time['weights'].append(client.model_weights_over_time)

            with open(os.path.join(save_dir, f'client_cwt-{args.experiment_name}.p'), 'wb') as f:
                pickle.dump(client_weights_over_time, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(save_dir, f'client_mwt-{args.experiment_name}.p'), 'wb') as f:
                pickle.dump(model_weights_over_time, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_dir, f'client_matrix-{args.experiment_name}.p'), 'wb') as f:
            pickle.dump(client_weight_matrices, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Decay learning rate
        server.args.lr = np.max([args.min_learning_rate, server.args.lr * args.learning_rate_decay])

    # One more round of federated update and evaluation
    if cfg.FEDERATION.METHOD != 'local':
        m = max(int(args.federating_ratio * population.num_clients), 1)
        client_indices = np.random.choice(range(population.num_clients), m, replace=False)  # Somethign about the randomness that makes things wack?

        if args.replicate in [1653, 1654]:
            client_indices = sorted(client_indices)

        federating_clients = [population.clients[ix] for ix in client_indices]
        federating_clients = server.initialize_clients(epoch + 1, selected_clients=federating_clients)
        server.local_eval(federating_clients, (epoch + 1) * args.federation_epoch)
    else:
        server.local_eval(federating_clients, (epoch + 1) * args.federation_epoch)

    # Save data
    save_client_metrics(population.clients, training_metrics_path)
    # Save evaluation data after federating
    pd.DataFrame(server.client_eval_metrics).to_csv(test_metrics_path, index=False)

