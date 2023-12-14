import argparse
import numpy as np
import os.path as osp
import os
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from loader import QM9
from model_gcn import GCN


def train(model, optimizer, loader, device):
    model.train()
    train_loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        tmp_real_data = data.y[~torch.isinf(data.y)].view(-1, 1)
        tmp_pred_data = model(data.x, data.edge_index, data.batch)
        tmp_pred_data_mod = tmp_pred_data[~torch.isinf(data.y)].view(-1, 1)
        loss = F.mse_loss(tmp_pred_data_mod, tmp_real_data)
        loss.backward()
        optimizer.step()
        train_loss_all += loss.item() * data.num_graphs

    return np.sqrt(train_loss_all / len(loader.dataset))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN message passing layers.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--global_pooling', type=str, default="sum",
                        help='global pooling (sum, mean - default: sum)')
    parser.add_argument('--batch_norm', type=int, default=1, help='Batch Normalization off (0) or on (1) (default: 1)')
    parser.add_argument('--seed', type=int, default=321, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_geometric.seed_everything(args.seed)

    dataset = QM9('Train/')

    val_percent = 0
    test_percent = 0

    print('Operating on following hardware: ' + str(device), flush=True)

    # Shuffle training data
    dataset = dataset.shuffle()
    print('---- Target: Training of multitask model for 12 QM9 properties ----')
    # Print model parameters
    print("GCN model parameters: ", flush=True)
    print('Number of hidden units: ' + str(args.hidden_dim), flush=True)
    print('Number of GNN message passing layers: ' + str(args.num_layers), flush=True)
    print('Global pooling: ' + str(args.global_pooling), flush=True)
    print('Batch Normalization: ' + str(args.batch_norm), flush=True)

    # Calculate mean for target data points (multitask)
    norm_percent = test_percent
    norm_length = int(len(dataset) * norm_percent)
    mean_lst = []
    std_lst = []

    for target in range(0, len(dataset.data.y[norm_length])):
        tmp_target_data = dataset.data.y[norm_length:,target]
        mean_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).mean())
        std_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).std())

    # Normalize targets to mean = 0 and std = 1
    mean = torch.tensor(mean_lst, dtype=torch.float32)
    std = torch.tensor(std_lst, dtype=torch.float32)
    dataset.data.y = (dataset.data.y - mean) / std

    print('Target data mean: ' + str(mean.tolist()), flush=True)
    print('Target data standard deviation: ' + str(std.tolist()), flush=True)

    print('Training is based on ' + str(dataset.num_features) + ' atom features and ' +
          str(dataset.num_edge_features) + ' edge features for a molecule.', flush=True)

    # Split dataset into train, validation, test set
    test_length = int(len(dataset) * test_percent)
    val_length = int(len(dataset) * val_percent)

    val_dataset = dataset[test_length:test_length+val_length]
    train_dataset = dataset[test_length+val_length:]
    test_dataset = dataset[:test_length]

    print('Length of train dataset: ' + str(len(train_dataset)), flush=True)
    print('Length of validation dataset: ' + str(len(val_dataset)), flush=True)
    print('Length of test dataset: ' + str(len(test_dataset)), flush=True)

    num_tasks = dataset[0].y.shape[1]
    print("num tasks: ", num_tasks, flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, pin_memory_device='cuda')

    model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, batch_norm=args.batch_norm,
                num_layers=args.num_layers, pool=args.global_pooling, hidden_dim=args.hidden_dim,
                pred=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, optimizer, train_loader, device)
        print(f'Epoch: {epoch}, Training Loss: {train_loss}', flush=True)

    model_state = model.state_dict()

    # Save the model
    if not osp.exists('models_gcn'):
        os.makedirs('models_gcn')
    if not osp.exists('models_gcn/multitask'):
        os.makedirs('models_gcn/multitask')
    model_path = ('models_gcn/multitask/' + args.global_pooling + '_' + str(args.hidden_dim) +
                  '_' + str(args.num_layers) + '_' + str(args.batch_norm) + '_gcn.pt')

    torch.save(model_state, model_path)


if __name__ == "__main__":
    main()
