"""
Train an initial model to compute non-IID client datasets based on the latent representations of samples.
"""
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.models import vgg11_bn

from tqdm import tqdm

from federated.configs import cfg_fl as cfg
from federated.network import BaseConvNet, adjust_model_layers


def get_embeddings(train_set, test_set, num_epochs, args,
                   stopping_threshold=5):
    if args.device is None:
        args.device = torch.device('cuda:0')
    net = vgg11_bn(pretrained=True, progress=True)
    net.classifier[6] = nn.Linear(4096, args.num_classes)
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    lr = 0.001; momentum = 0.9; weight_decay = 1e-4
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f'> Computing model embeddings to setup latent non-IID federated datasets...')
    train_losses = []
    test_losses = []; min_test_loss_avg = 1e10  # For early stopping
    early_stopping_counter = 0
    
    train_loader = DataLoader(train_set, batch_size=args.bs_trn,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.bs_val,
                             shuffle=False, num_workers=args.num_workers)
   
    model_name = f'./models/{args.dataset}/setup_model-e={num_epochs - 1}-st={stopping_threshold}-lr={lr}-mo={momentum}-wd={weight_decay}.pt'
    
    try:
        net.load_state_dict(torch.load(model_name))
        net.to(args.device)
        net.eval()
        print(f'Data setup model loaded from {model_name}!')
        epoch = num_epochs - 1
    except:
        print(f'Data setup model from {model_name} not found. Training.')
    
        for epoch in range(num_epochs):
            train_loss, train_acc = train(net, train_loader, optimizer,
                                          criterion, epoch, args)
            # Note because we're just computing embeddings for these samples using 
            # fixed hyperparameters to *set up* the dataset + evaluation setting,
            # there's no validation. We just report test metrics to make sure the
            # hidden-layer representations are not overfitting.
            test_loss, test_acc = evaluate(net, test_loader, 
                                           criterion, epoch, args)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            # Early stopping
            if np.mean(test_losses) < min_test_loss_avg:
                min_test_loss_avg = np.mean(test_losses)
            else:
                early_stopping_counter += 1

            if early_stopping_counter == stopping_threshold:
                # Save model, end
                model_name = f'./models/{args.dataset}/setup_model-e={epoch}-st={stopping_threshold}-lr={lr}-mo={momentum}-wd={weight_decay}.pt'
                torch.save(net.state_dict(), model_name)
                break
                
            # Save model at end of each epoch regardless
            model_name = f'./models/{args.dataset}/setup_model-e={epoch}-st={stopping_threshold}-lr={lr}-mo={momentum}-wd={weight_decay}.pt'
            torch.save(net.state_dict(), model_name)
            
        if early_stopping_counter < stopping_threshold:
            model_name = f'./models/{args.dataset}/setup_model-e={epoch}-st={stopping_threshold}-lr={lr}-mo={momentum}-wd={weight_decay}.pt'
            torch.save(net.state_dict(), model_name)
        
    # Compute and save training and test embeddings
    train_embeddings = compute_embeddings(net, train_set, args, split='train')
    _, train_embeddings, _ = train_embeddings  # Default, use last-hidden layer
    test_embeddings = compute_embeddings(net, test_set, args, split='test')
    _, test_embeddings, _ = test_embeddings  # Default, use last-hidden layer
    
    train_embedding_fname = f'embeddings-d={args.dataset}-e={epoch}-st={stopping_threshold}-lr={lr}-mo={momentum}-wd={weight_decay}-fc2-train.npy'
    test_embedding_fname  = f'embeddings-d={args.dataset}-e={epoch}-st={stopping_threshold}-lr={lr}-mo={momentum}-wd={weight_decay}-fc2-test.npy'
    
    with open(f'./precomputed/{train_embedding_fname}', 'wb') as f:
        np.save(f, train_embeddings)
    print(f'> Saved FL setup train embeddings to ./precomputed/{train_embedding_fname}!')
    with open(f'./precomputed/{test_embedding_fname}', 'wb') as f:
        np.save(f, test_embeddings)
    print(f'> Saved FL setup test embeddings to ./precomputed/{test_embedding_fname}!')
    return train_embeddings, test_embeddings


def train(net, dataloader, optimizer, criterion, epoch, args):
    net.train()
    net.to(args.device)
    
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(dataloader, desc='Training')):
        inputs, labels = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Epoch: {epoch} | Train accuracy: {100 * correct / total:<.2f}% | Train loss: {running_loss / len(dataloader):<.4f}')
    return running_loss, correct / total


def evaluate(net, dataloader, criterion, epoch, args):
    net.eval()
    net.to(args.device)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc='Evaluating')):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch: {epoch} | Eval accuracy: {100 * correct / total:<.2f}% | Eval loss: {running_loss / len(dataloader):<.4f}')
    return running_loss, correct / total


# Embedding helper functions
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []


def compute_embeddings(net, dataset, args, split):
    """
    Actually compute embeddings given a dataset
    """
    total_embeddings = []
    total = 0
    correct = 0
    
    dataloader = DataLoader(dataset, batch_size=args.bs_val,
                            shuffle=False, num_workers=args.num_workers)
    
    net.eval()
    net.to(args.device)
    
    save_output = SaveOutput()
    hook_handles = []
    for layer in net.modules():
#         if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
#             handle = layer.register_forward_hook(save_output)
#             hook_handles.append(handle)
        if isinstance(layer, torch.nn.Linear):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader,
                                      desc=f'Computing embeddings ({split})')):
            inputs, labels = data
            inputs = inputs.to(args.device)
            outputs = net(inputs)
            # embeddings = net.embed(inputs)
            # total_embeddings.append(embeddings.detach().cpu().numpy())
            
            # Compute classification accuracy of setup model
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted.cpu() == labels).sum().item()
            
        total_embeddings = [None] * len(save_output.outputs)
        for ix, output in enumerate(save_output.outputs):
            total_embeddings[ix] = output.detach().cpu().numpy().squeeze()
            
#         total_embeddings = [e.flatten() for e in total_embeddings]
        n_samples = len(dataset.targets)
        total_embeddings_fc1 = np.stack(total_embeddings[0::3]).reshape((n_samples, -1))
        total_embeddings_fc2 = np.stack(total_embeddings[1::3]).reshape((n_samples, -1))
        total_embeddings_fc3 = np.stack(total_embeddings[2::3]).reshape((n_samples, -1))
        
        num_samples = len(dataset.targets)
        total_embeddings_fc1
        print(total_embeddings_fc1.shape)
        print(total_embeddings_fc2.shape)
        print(total_embeddings_fc3.shape)
        
        print(f'Latent distribution setup model accuracy: {100 * correct / total:<.2f}%')
    
    # total_embeddings = np.concatenate(total_embeddings)
    return total_embeddings_fc1, total_embeddings_fc2, total_embeddings_fc3
            
    
