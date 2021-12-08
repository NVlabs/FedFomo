# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""
Experiment arguments
- Overwrites federated.config.py in case of overlap
"""

import argparse

parser = argparse.ArgumentParser(description='Federated Learning')

# General setup
parser.add_argument('--seed', default=None, type=int, help="Random seed (default 0)")
parser.add_argument('--seed_torch', default=None, type=int,
                    help="Random seed for torch and CUDA (default 0)")
parser.add_argument('--data_seed', default=0, type=int, help="Random seed for initializing data")
parser.add_argument('--deterministic', default=False, action='store_true', 
                    help="Make PyTorch deterministic")
parser.add_argument('--cuda', default=True, type=bool,
                    help="Whether to allow using CUDA")
parser.add_argument('--task', type=str, default=None,
                    help="Local client task: ('classification')")
parser.add_argument('--print_logx', default=False, action='store_true',
                    help="Whether to print runx.log.msg() to stdout")
parser.add_argument('-r', '--replicate', default=None, type=int,
                    help="Replicate number")

parser.add_argument('--debugging', default=False, action='store_true',
                    help="Call for debugging logic / conditionals")

parser.add_argument('--evaluate', default=False, action='store_true',
                    help="If true, load saved models and only run server evaluation")
parser.add_argument('--no_eval', default=False, action='store_true',
                    help="If true, don't evaluate federated models during same training call")

parser.add_argument('-edt', '--eval_distribution_test', action='store_true',
                    help="If true, set the entire population distribution's test set as the training curve test set")
parser.add_argument('-lub', '--local_upper_bound', action='store_true',
                    help="If true, set the entire population distribution's train set as the train set - use for local model upper bound evaluation")

parser.add_argument('--apex', default=False, action='store_true',
                    help="Apex for training")
parser.add_argument('--device', default=0, type=int,
                    help="If specified, enforce training on a specific GPU")

## Distributed training
parser.add_argument('--parallelize', default=False, action='store_true',
                    help="If true, assign clients to individual GPUs and parallelize training and evaluation of models across GPUs")
parser.add_argument('--rank', default=-1, type=int,
                    help='Node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:29500', type=str,
                    help='Url used to set up distributed training')
parser.add_argument('--world_size', default=-1, type=int,
                    help='Number of nodes for distributed training')


# Data
parser.add_argument('--dataset', type=str, default=None,
                    help="Dataset to use: ('cifar10', 'cifar10_shard', 'cifar100', 'mnist', 'emnist')")
parser.add_argument('--num_workers', type=int, default=2,
                    help="CPU worker threads per dataloader instance")

## ImageNet
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')

# Federated distribution setup
parser.add_argument('--max_epoch', default=100, type=int,
                    help="Number of federated learning communication rounds")
parser.add_argument('--manual_client_setup', default=False, action='store_true',
                    help="Whether to use manual config.py settings for initializing clients")
parser.add_argument('--num_distributions', type=int, default=None,
                    help="Number of distributions to setup federated experiments")
parser.add_argument('--clients_per_dist', type=int, default=None,
                    help="Number of clients per distribution, required if 'num_clients' is None (default = 5)")
parser.add_argument('--num_clients', type=int, default=None,
                    help="Total number of clients, required if 'clients_per_dist' not specified")
parser.add_argument('--clustering_method', type=str, default=None,
                    help="Clustering method to set up distributions ('KMeans' | 'AgglomerativeClustering')")
parser.add_argument('-ltvs', '--local_train_val_size', type=int, default=None,
                    help="If specified, number of data points available in the local train set")
parser.add_argument('-rd', '--random_distributions', default=False, action='store_true',
                    help="If distributions should be initialized randomly to keep IID local datasets")

## Dataset distribution mixing, e.g. sharing data to help with non-IID data
parser.add_argument('-ltdr', '--local_train_dist_ratio', default=1.0, type=float,
                    help="Fraction of the local train data that should be the client's originally assigned distribution")
parser.add_argument('-lvr', '--local_val_ratio', default=1.0, type=float,
                    help="Fraction of local validation data that should be the client's original val data")
parser.add_argument('-lvdr', '--local_val_dist_ratio', default=1.0, type=float,
                    help="Fraction of local validation data that should be the client's originally assigned distribution")
parser.add_argument('-nlvp', '--num_local_val_pooled', default=0., type=float,
                    help="Number of local val data points pooled / shared")
parser.add_argument('-nltp', '--num_local_train_pooled', default=0., type=float,
                    help="Number of local train data points pooled / shared")
parser.add_argument('--pool_train_data', default=False, action='store_true',
                    help="If true, clients should share the multi-distribution dataset points for training")
parser.add_argument('--pool_val_data', default=True, action='store_true',
                    help="If true, clients should share the multi-distribution dataset points for validation")
parser.add_argument('--num_adversaries', default=0, type=int,
                    help="Number of clients to carry non-useful or adversarial data")

## Pathological Non-IID setup
parser.add_argument('--pathological_non_iid', default=False, action='store_true',
                    help="If true, randomly allocate n classes per distribution")
parser.add_argument('--classes_per_dist', type=int, default=None,
                    help="Specify number of classes per distribution")
parser.add_argument('--shards_per_user', type=int, default=None,
                    help="If specified, sample pathological non-iid at the client level")
parser.add_argument('--shuffle_targets', default=False, action='store_true',
                    help="If specified, shuffle the val and test splits among the clients")

# Model training
parser.add_argument('--arch', type=str, default='tf_cnn', 
                    choices=['base_cnn', 'base_cnn_224', 'tf_cnn'],
                    help="Network architecture. Default is CNN used in Tensorflow tutorial")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Local model learning rate (shared amongst clients)")
parser.add_argument('--global_training_curve', default=False, action='store_true',
                    help="If true, save evaluation results on globally held-out test set")
parser.add_argument('--optimizer', type=str, default='sgd',
                    choices=['sgd', 'adam'], help="optimizer")
parser.add_argument('--amsgrad', action='store_true', help="amsgrad for adam")
parser.add_argument('--bs_trn', type=int, default=50,
                    help="Batch size for training per gpu")
parser.add_argument('--bs_val', type=int, default=50,
                    help="Batch size for Validation per gpu")
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('-lr_decay', '--learning_rate_decay', type=float, default=1.0)
parser.add_argument('-min_lr', '--min_learning_rate', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--nesterov', default=False, action='store_true',
                    help="Use Nesterov momentum")
parser.add_argument('--train_split', default=None, type=float,
                    help="Training split ratio for local training")
parser.add_argument('--fedavg_rounds', default=None, type=int,
                    help="If specified, run federated averaging for n rounds, then switch to fomo")
parser.add_argument('--local_rounds', default=None, type=int,
                    help="If specified, run local training for n rounds, then switch to fomo")

# Federations
parser.add_argument('-nf', '--num_federations', default=None, type=int,
                    help="Number of federations to create during training")
parser.add_argument('-fr', '--federating_ratio', default=1.0, type=float,
                    help="Fraction of clients to participate in each federated round")
parser.add_argument('-fe', '--federation_epoch', default=None, type=int,
                    help="Number of local training epochs before federating")
parser.add_argument('--federated_averaging', default=False, action='store_true')
parser.add_argument('--federation_method', default=None, type=str,
                    help="The federation method, either ('fomo', 'local', 'fedavg')")
parser.add_argument('--num_update_clients', default=None, type=int,
                    help="Number of clients to query for an update")

## Client weights / FOMO arguments
parser.add_argument('--model_weight_delta', default=None, type=float,
                    help="Model delta for computing average models")
parser.add_argument('--client_weight_epsilon', default=None, type=float,
                    help="Exploration parameter for finding clients to federate with")
parser.add_argument('--client_weight_epsilon_decay', default=None, type=float,
                    help="Every epoch, decrease by this amount (reduce exploration)")
parser.add_argument('--client_weight_method', default=None, type=str,
                    help="How to select weights, either ('sub_federations' | 'e_greedy')")
parser.add_argument('--client_initial_self_weight', default=0.1, type=float,
                    help="Initial value that client should weight its own model, ideally > 0.")

parser.add_argument('--update_positive_delta_only', default=False, action='store_true',
                    help="If true, when updating model weights only consider positive deltas")
parser.add_argument('--leave_one_out', default=False, action='store_true',
                    help="If true, infer model weight delta to leave a model out, o.w. use --model_weight_delta")
parser.add_argument('--explicit_weight_delta', default=None, type=float,
                    help="If specified, functions the same as --model_weight_delta")

parser.add_argument('--baseline_model', default=None, type=str,
                    help="Baseline comparison model: ('first_model' | 'last_model' | 'model_avg' | 'current')")
parser.add_argument('--model_delta_norm', default='l1', type=str,
                    help="Which norm to use in FedFomo update, from ('l1' | 'l2' | 'l2_root')")
parser.add_argument('--no_model_delta_norm', default=False, action='store_true',
                    help="If specified do not normalize loss deltas with model norm")


parser.add_argument('--softmax_client_weights', default=False, action='store_true',
                    help="If true, softmax the client weight vectors to preserve ordering into 0 and 1")
parser.add_argument('--softmax_model_deltas', default=False, action='store_true',
                    help="If true, softmax the model delta vectors for soft weighting of each model")

## Additional extra training choices - not used for current results
parser.add_argument('-lvmdr', '--local_val_model_delta_ratio', default=None, type=float,
                    help="Use with nlvp > 0, if specified evaluate models both on local and public val set, computing total performance as a weighted average")
parser.add_argument('-ebc', '--eval_by_class', default=False, action='store_true',
                    help="If true, for classification evaluation compare the losses over specific classes. If another model does better than a client's model over a specific subset of classes, ")
parser.add_argument('--infer_self_weight', default=False, action='store_true',
                    help="If true, calculate weighting without regard to self")

## Differential privacy
### Requires the Opacus library
parser.add_argument('-na', '--n_accumulation_steps', default=1, type=int,
                    help="Number of mini-batches to accumulate into an effective batch")
parser.add_argument('--sigma', default=1.0, type=float,
                    help="Noise multiplier (default=1.0)")
parser.add_argument('-c', '--max_per_sample_grad_norm', default=1.0, type=float,
                    help="clip per-sample gradients to this norm (default 1.0)")
parser.add_argument('-dp', '--enable_dp', default=False, action='store_true',
                    help="Enable differentially private training via DP-SGD")
parser.add_argument('--secure_rng', default=False, action='store_true',
                    help="Enable Secure RNG to have trustworthy privacy guarantees. Higher performance cost")
parser.add_argument('--delta', default=1e-5, type=float,
                    help="Target delta (default: 1e-5)")
parser.add_argument('-vbr', '--virtual_batch_rate', default=None, type=float,
                    help="If specified, divide bs_trn and bs_val by virtual_batch_rate to keep memory low")
### FedProx
parser.add_argument('--fedprox', default=False, action='store_true',
                    help="If true, locally train with proximal term")
parser.add_argument('--fedprox_mu', default=None, type=float,
                    help="Proximal term mu coefficient")
### FedAvgM
parser.add_argument('--fed_momentum', default=False, action='store_true',
                    help="If true, use momentum in the federated update")
parser.add_argument('-fmg', '--fed_momentum_gamma', default=0., type=float,
                    help="Gamma term in momentum for computing update")
parser.add_argument('-fmn', '--fed_momentum_nesterov', default=False, action='store_true',
                    help="If true, use Nesterov accumulated gradient for momentum update")
### Federated Evaluation
parser.add_argument('-elfs', '--eval_local_finetune_size', default=None, type=int,
                    help="Number of samples to make available during fine-tuning after federations are learned")
parser.add_argument('--eval_global_only', default=False, action='store_true')
### Clustered Federated Learning
parser.add_argument('-e1', '--epsilon_1', default=0.42, type=float)
parser.add_argument('-e2', '--epsilon_2', default=1.6, type=float)
### Personalization with Moreau Envelopes
parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg"])
parser.add_argument("--K", type=int, default=5, help="Computation steps")
parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
parser.add_argument("--times", type=int, default=5, help="running time")

args = parser.parse_args()
