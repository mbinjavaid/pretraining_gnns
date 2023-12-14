import argparse
import os
import glob
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import GINConcat, GIN, GCN


def train(model, device, optimizer, loader):
    model.train()
    train_loss_all = 0
    count = 0
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
        count += list(tmp_real_data.size())[0]

    return train_loss_all / count


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio - dropout only implemented for GNN type gin and ginconcat. (default: 0.2)')
    parser.add_argument('--use_embeddings', type=int, default=0,
                        help='whether to use feature embeddings for nodes (1) or multi-hot encodings (0) - '
                             'if using strategy masking or contextpred, will search for pretrained model(s) with same '
                             'use_embeddings setting. (default: 0)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='hidden dimension for GNN - if using strategy masking or contextpred, '
                             'if using strategy masking or contextpred, will not use this setting (will instead '
                             'adopt the hidden_dim setting of the pretrained model being used). (default: 300)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers - '
                             'if using strategy masking or contextpred, will not use this setting (will instead '
                             'adopt the num_layers setting of the pretrained model being used). (default: 5)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='whether batch normalization is used (1) or not (0) - if using strategy masking or '
                             'contextpred, will not use this setting (will instead '
                             'adopt the batch_norm setting of the pretrained model being used). (default: 1)')
    parser.add_argument('--strategy', type=str, default='none',
                        help='which pretrained model to use as initialization: masking, contextpred, or none (default: none)')
    parser.add_argument('--global_pooling', type=str, default="sum",
                        help='global pooling (sum, mean). Default: sum)')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help="ginconcat, gin, gcn - "
                             "only matters if strategy is none - otherwise, will load the same GNN model "
                             "as the saved pretrained model being used (default: gin).")
    parser.add_argument('--seed', type=int, default=321, help="Random seed.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_geometric.seed_everything(args.seed)

    if args.use_embeddings:
        from loader_embeddings import QM9
    else:
        from loader_multihot import QM9
    dataset = QM9('Train/')

    val_percent = 0
    test_percent = 0

    print('Operating on following hardware: ' + str(device), flush=True)

    # Shuffle training data
    dataset = dataset.shuffle()
    print('Supervised pre-training on top of the following self-supervised strategy: ' + args.strategy, flush=True)

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

    # Create a list of all the models to be loaded, i.e. files with the .pt extension inside the folder:
    if args.strategy == 'masking':
        if args.use_embeddings:
            pretrained_model_list = glob.glob('models/embeddings_pretrain_masking/*.pt')
        else:
            pretrained_model_list = glob.glob('models/multihot_pretrain_masking/*.pt')
    elif args.strategy == 'contextpred':
        if args.use_embeddings:
            pretrained_model_list = glob.glob('models/embeddings_pretrain_contextpred/*.pt')
        else:
            pretrained_model_list = glob.glob('models/multihot_pretrain_contextpred/*.pt')
    elif args.strategy == 'none':
        pretrained_model_list = ['none']
    else:
        raise ValueError('Invalid strategy specified (must be masking, contextpred, or none).')

    if len(pretrained_model_list) == 0:
        raise FileNotFoundError('No pretrained models found for the specified strategy. Please check the strategy, '
                         'use_embeddings, hidden_dim, num_layers, and batch_norm settings.')

    if pretrained_model_list[0] != 'none':
        print('Found the following pretrained models: ' + str(pretrained_model_list), flush=True)
        for pretrained_model in pretrained_model_list:
            # the filename follows the pattern of, e.g. 300_5_0_ginconcat.pt
            pretrained_model_ = os.path.basename(pretrained_model)
            # Remove the .pt ending:
            pretrained_model_ = pretrained_model_[:-3]
            # Split the filename into its components:
            pretrained_model_ = pretrained_model_.split('_')
            hidden_dim = int(pretrained_model_[0])
            num_layers = int(pretrained_model_[1])
            batch_norm = int(pretrained_model_[2])
            gnn_type = pretrained_model_[3]

            torch_geometric.seed_everything(args.seed)
            # Load the appropriate architecture based on which pretrained model is being considered:
            if gnn_type == 'ginconcat':
                model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                                  drop_ratio=args.dropout_ratio, pred=True, use_embeddings=args.use_embeddings,
                                  hidden_dim=hidden_dim, num_layers=num_layers, pool=args.global_pooling,
                                  batch_norm=batch_norm).to(device)
                model_name = 'ginconcat'
            elif gnn_type == 'gin':
                model = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                            drop_ratio=args.dropout_ratio, pred=True, use_embeddings=args.use_embeddings,
                            hidden_dim=hidden_dim, num_layers=num_layers, pool=args.global_pooling,
                            batch_norm=batch_norm).to(device)
                model_name = 'gin'
            elif gnn_type == 'gcn':
                model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, batch_norm=batch_norm, pred=True,
                            use_embeddings=args.use_embeddings, hidden_dim=hidden_dim, pool=args.global_pooling,
                            num_layers=num_layers, drop_ratio=args.dropout_ratio).to(device)
                model_name = 'gcn'
            else:
                raise ValueError('Invalid GNN model specified.')

            print('---- Target: Training of ' + model_name + ' multitask model for all QM9 properties on top of '
                                                             'strategy: ' + args.strategy + ' ---- ', flush=True)
            print('Using the following pretrained model as initialization: ' + os.path.basename(pretrained_model), flush=True)
            # Load the pretrained model
            pretrained_model_state = torch.load(pretrained_model)

            # Load the pretrained model weights into the model:
            model_state_dict = model.state_dict()
            for name, param in pretrained_model_state.items():
                if ('graph_prediction' not in name) and ('mlp' not in name):
                    model_state_dict[name] = param
            # Update the model's state dictionary with the GCN layer parameters
            model.load_state_dict(model_state_dict)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      pin_memory_device='cuda', num_workers=args.num_workers)

            # Training loop
            for epoch in range(1, args.epochs + 1):
                train_loss = train(model, device, optimizer, train_loader)
                print(f'Epoch: {epoch}, Training Loss: {train_loss}', flush=True)

            model_state = model.state_dict()
            # Save the model
            # Extract only the filename from the path and save it in a variable:
            model_filename = args.global_pooling + '_' + os.path.basename(pretrained_model)
            if not os.path.exists("models"):
                os.makedirs("models")
            if args.use_embeddings:
                if not os.path.exists("models/embeddings_pretrain_supervised"):
                    os.makedirs("models/embeddings_pretrain_supervised")
                model_path = "models/embeddings_pretrain_supervised/" + model_filename
            else:
                if not os.path.exists("models/multihot_pretrain_supervised"):
                    os.makedirs("models/multihot_pretrain_supervised")
                model_path = "models/multihot_pretrain_supervised/" + model_filename

            torch.save(model_state, model_path)

    # Case where no pretrained model is used (i.e. only supervised pretraining):
    else:
        torch_geometric.seed_everything(args.seed)
        print('Only carrying out supervised pre-training ...', flush=True)
        if args.gnn_type == 'ginconcat':
            model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                              drop_ratio=args.dropout_ratio, pred=True, pool=args.global_pooling,
                              hidden_dim=args.hidden_dim, use_embeddings=args.use_embeddings,
                              num_layers=args.num_layers, batch_norm=args.batch_norm).to(device)
            model_name = 'ginconcat'
        elif args.gnn_type == 'gin':
            model = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                              drop_ratio=args.dropout_ratio, pred=True, pool=args.global_pooling,
                              hidden_dim=args.hidden_dim, use_embeddings=args.use_embeddings,
                              num_layers=args.num_layers, batch_norm=args.batch_norm).to(device)
            model_name = 'gin'
        elif args.gnn_type == 'gcn':
            model = GCN(num_tasks=num_tasks, num_features=dataset.num_features,
                        drop_ratio=args.dropout_ratio, pred=True, pool=args.global_pooling,
                        hidden_dim=args.hidden_dim, use_embeddings=args.use_embeddings,
                        num_layers=args.num_layers, batch_norm=args.batch_norm).to(device)
            model_name = 'gcn'
        else:
            raise ValueError('Invalid GNN model specified.')

        print('---- Target: Training of ' + model_name + ' multitask model for all QM9 properties on top of '
                                                         'strategy: ' + args.strategy + ' ---- ', flush=True)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  pin_memory_device='cuda', num_workers=args.num_workers)

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, device, optimizer, train_loader)
            print(f'Epoch: {epoch}, Training Loss: {train_loss}', flush=True)

        model_state = model.state_dict()
        # Save the model
        # 300_5_sum_0_ginconcat.pt
        model_filename = args.global_pooling + '_' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(args.batch_norm) + '_' + args.gnn_type + "_sup_only" + ".pt"
        if not os.path.exists("models"):
            os.makedirs("models")
        if args.use_embeddings:
            if not os.path.exists("models/embeddings_pretrain_supervised"):
                os.makedirs("models/embeddings_pretrain_supervised")
            model_path = "models/embeddings_pretrain_supervised/" + model_filename
        else:
            if not os.path.exists("models/multihot_pretrain_supervised"):
                os.makedirs("models/multihot_pretrain_supervised")
            model_path = "models/multihot_pretrain_supervised/" + model_filename
        torch.save(model_state, model_path)


if __name__ == "__main__":
    main()
