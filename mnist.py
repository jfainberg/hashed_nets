import argparse
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from layers import HashLinear
from utils import get_equivalent_compression


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch HashedNets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--nhLayers', type=int, default=1,
                        help='# hidden layers, excluding input/output layers')
    parser.add_argument('--nhu', type=int, default=1000,
                        help='Number of hidden units')
    parser.add_argument('--hashed', default=False, action='store_true',
                        help='Enable hashing')
    parser.add_argument('--compress', type=float, default=0.03125,
                        help='Compression rate')
    parser.add_argument('--hash-bias', default=False, action='store_true',
                        help='Hash bias terms')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate at t=0')
    parser.add_argument('--decay-factor', type=float, default=0.1,
                        help='Learning rate decay factor')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Mini-batch size (1 = pure stochastic')
    parser.add_argument('--validation-percent', type=float, default=0.1,
                        help='Percent of training data used for validation')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum (SGD only)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate')
    parser.add_argument('--l2reg', type=float, default=0.0,
                        help='l2 regularisation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum # of epochs')
    parser.add_argument('--patience', type=int, default=2,
                        help='Number of epochs to wait before scaling lr.')
    parser.add_argument('--no-xi', default=True, action='store_false',
                        help='Do not use auxiliary hash (sign factor)')
    parser.add_argument('--hash-seed', type=int, default=2,
                        help='Seed for hash functions')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    print(args)
    return args


def load_data(batch_size, kwargs):
    '''
    Load MNIST data. Largely from PyTorch MNIST example.
    '''
    train_dataset = datasets.MNIST('../data',
                                   train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    validation_percent = 0.1
    split = int(math.floor(validation_percent * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, sampler=train_sampler, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, sampler=valid_sampler, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, valid_loader, test_loader


def train(model, device, train_loader, optimizer, epoch, log_interval=5):
    '''
    One epoch of training.
    '''
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()), end='\r')
        train_loss += loss.item()

    return train_loss / len(train_loader)


def evaluate(model, device, loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(loader.sampler)
    accuracy = 100. * correct / len(loader.sampler)

    return loss, accuracy


class Net(nn.Module):
    '''
    Standard feedforward network with ReLU activations
    and optional interleaving dropout layers.
    '''
    def __init__(self, input_dim, output_dim, nhLayers=1, nhu=1000,
                 compress=1.0, dropout=0.25):
        super(Net, self).__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim

        c_nhu = round(nhu * compress)

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(input_dim, c_nhu)
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(self, 'linear' + str(layer), nn.Linear(c_nhu, c_nhu))
            setattr(self, 'dropout' + str(layer), nn.Dropout(dropout))

        self.linear_out = nn.Linear(c_nhu, output_dim)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, 'linear' + str(layer))(x))
            x = getattr(self, 'dropout' + str(layer))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


class HashedNet(nn.Module):
    '''
    Feedforward network with hashed linear layers,
    ReLU activations and optional interleaving dropout layers.
    '''
    def __init__(self, input_dim, output_dim, nhLayers=1, nhu=1000,
                 compress=1.0, dropout=0.25, hash_seed=2):
        super(HashedNet, self).__init__()
        self.nhLayers = nhLayers
        self.input_dim = input_dim

        self.dropout0 = nn.Dropout(dropout)
        self.linear1 = HashLinear(input_dim, nhu, compress)
        self.dropout1 = nn.Dropout(dropout)

        for layer in range(2, nhLayers + 1):
            setattr(self, 'linear' + str(layer), HashLinear(nhu, nhu, compress,
                                                            hash_seed))
            setattr(self, 'dropout' + str(layer), nn.Dropout(dropout))

        self.linear_out = HashLinear(nhu, output_dim, compress,
                                     hash_bias=False, hash_seed=hash_seed)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        x = self.dropout0(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        for layer in range(2, self.nhLayers + 1):
            x = F.relu(getattr(self, 'linear' + str(layer))(x))
            x = getattr(self, 'dropout' + str(layer))(x)

        x = self.linear_out(x)
        return F.log_softmax(x, dim=1)


def main():
    args = parse_arguments()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    random.seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    tr_loader, val_loader, test_loader = load_data(args.batch_size, kwargs)
    input_dim = 784
    output_dim = 10

    if args.hashed:
        model = HashedNet(input_dim, output_dim, args.nhLayers, args.nhu,
                          args.compress, args.dropout, args.hash_seed).to(device)
    else:
        eq_compress = get_equivalent_compression(input_dim, output_dim,
                                                 args.nhu, args.nhLayers, args.compress)
        model = Net(input_dim, output_dim, args.nhLayers, args.nhu,
                    eq_compress, args.dropout).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.l2reg)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.decay_factor,
                                                     patience=args.patience,
                                                     verbose=True)

    print('The number of parameters is: {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, device, tr_loader, optimizer, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader)
        scheduler.step(val_loss)
        print('Epoch {} Train loss: {:.3f} Val loss: {:.3f} Val acc: {:.2f}%'.format(
              epoch, tr_loss, val_loss, val_acc))

    test_loss, test_acc = evaluate(model, device, test_loader)
    print('Test loss: {:.3f} Test acc: {:.2f}%'.format(test_loss, test_acc))

    if (args.save_model):
        torch.save(model.state_dict(), "mnist.pt")


if __name__ == '__main__':
    main()
