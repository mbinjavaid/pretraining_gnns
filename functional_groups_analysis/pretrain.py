import argparse
import pickle
import sys
import os
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import GIN, GINConcat, GCN


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
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio - dropout only implemented for GNN type gin and ginconcat. (default: 0.2)')
    parser.add_argument('--global_pooling', type=str, default="sum",
                        help='global pooling (sum, mean - default: sum)')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help="ginconcat, gin, or gcn - default: gin")
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='whether to use batch norm (default: 1).')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--features', type=str, default="chem",
                        help="chemical features (chem) or functional group encodings (fge) - default: chem")
    parser.add_argument('--pretrain_subset', type=str, default="10k",
                        help="10k, 10kExcluded, or full - default: 10k")
    parser.add_argument('--seed', type=int, default=321, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    args = parser.parse_args()

    if args.features == "chem":
        from loader_chem import QM9
    elif args.features == "fge":
        from loader_fge import QM9
    else:
        raise ValueError('Invalid feature type specified (must be chem or fge).')

    if args.pretrain_subset == "10k":
        try:
            with open('functional_groups/qm9_ait_10k_indices.pkl', 'rb') as f:
                qm9_ait_10k = pickle.load(f)
            with open('functional_groups/qm9_hf_10k_indices.pkl', 'rb') as f:
                qm9_hf_10k = pickle.load(f)
            with open('functional_groups/qm9_ld50_10k_indices.pkl', 'rb') as f:
                qm9_ld50_10k = pickle.load(f)
            with open('functional_groups/qm9_lmv_10k_indices.pkl', 'rb') as f:
                qm9_lmv_10k = pickle.load(f)
            with open('functional_groups/qm9_logp_10k_indices.pkl', 'rb') as f:
                qm9_logp_10k = pickle.load(f)
            with open('functional_groups/qm9_osha_twa_10k_indices.pkl', 'rb') as f:
                qm9_osha_10k = pickle.load(f)
            with open('functional_groups/qm9_tb_10k_indices.pkl', 'rb') as f:
                qm9_tb_10k = pickle.load(f)
            with open('functional_groups/qm9_tm_10k_indices.pkl', 'rb') as f:
                qm9_tm_10k = pickle.load(f)

            pretraining_indices = [qm9_ait_10k, qm9_hf_10k, qm9_ld50_10k,
                                        qm9_lmv_10k, qm9_logp_10k, qm9_osha_10k,
                                        qm9_tb_10k, qm9_tm_10k]
        except FileNotFoundError:
            print("Functional group pretraining indices not found. Please run qm9_fgs_split.py first.")
            sys.exit()

    elif args.pretrain_subset == "10kExcluded":
        try:
            with open('functional_groups/qm9_ait_10k_excluded_indices.pkl', 'rb') as f:
                qm9_ait_10k_excluded = pickle.load(f)
            with open('functional_groups/qm9_hf_10k_excluded_indices.pkl', 'rb') as f:
                qm9_hf_10k_excluded = pickle.load(f)
            with open('functional_groups/qm9_ld50_10k_excluded_indices.pkl', 'rb') as f:
                qm9_ld50_10k_excluded = pickle.load(f)
            with open('functional_groups/qm9_lmv_10k_excluded_indices.pkl', 'rb') as f:
                qm9_lmv_10k_excluded = pickle.load(f)
            with open('functional_groups/qm9_logp_10k_excluded_indices.pkl', 'rb') as f:
                qm9_logp_10k_excluded = pickle.load(f)
            with open('functional_groups/qm9_osha_twa_10k_excluded_indices.pkl', 'rb') as f:
                qm9_osha_10k_excluded = pickle.load(f)
            with open('functional_groups/qm9_tb_10k_excluded_indices.pkl', 'rb') as f:
                qm9_tb_10k_excluded = pickle.load(f)
            with open('functional_groups/qm9_tm_10k_excluded_indices.pkl', 'rb') as f:
                qm9_tm_10k_excluded = pickle.load(f)

            pretraining_indices = [qm9_ait_10k_excluded, qm9_hf_10k_excluded, qm9_ld50_10k_excluded,
                                     qm9_lmv_10k_excluded, qm9_logp_10k_excluded, qm9_osha_10k_excluded,
                                      qm9_tb_10k_excluded, qm9_tm_10k_excluded]
        except FileNotFoundError:
            print("Functional group pretraining indices not found. Please run qm9_fgs_split.py first.")
            sys.exit()
    elif args.pretrain_subset != "full":
        raise ValueError('Invalid pretraining subset specified (must be 10k, 10k_excluded, or full).')

    indices_names = ['ait', 'hf', 'ld50', 'lmv', 'logp', 'osha', 'tb', 'tm']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_geometric.seed_everything(args.seed)

    dataset = QM9('Train/')

    val_percent = 0
    test_percent = 0

    print('Operating on following hardware: ' + str(device), flush=True)
    print('On the following subset of QM9: ' + args.pretrain_subset, flush=True)

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

    if args.pretrain_subset != "full":
        for idx, indices in enumerate(pretraining_indices):
            torch_geometric.seed_everything(args.seed)
            # Split dataset into train, validation, test set
            test_length = int(len(dataset) * test_percent)
            val_length = int(len(dataset) * val_percent)

            val_dataset = dataset[test_length:test_length+val_length]
            test_dataset = dataset[:test_length]
            train_dataset = dataset.index_select(indices)

            print('Length of train dataset: ' + str(len(train_dataset)), flush=True)
            print('Length of validation dataset: ' + str(len(val_dataset)), flush=True)
            print('Length of test dataset: ' + str(len(test_dataset)), flush=True)

            num_tasks = dataset[0].y.shape[1]
            print("num tasks: ", num_tasks, flush=True)

            # Load the appropriate architecture based on which pretrained model is being considered:
            if args.gnn_type == 'ginconcat':
                model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                                  drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                                  hidden_dim=args.hidden_dim, num_layers=args.num_layers, pool=args.global_pooling,
                                  batch_norm=args.batch_norm).to(device)
                model_name = 'ginconcat'
            elif args.gnn_type == 'gin':
                model = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                            drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                            hidden_dim=args.hidden_dim, num_layers=args.num_layers, pool=args.global_pooling,
                            batch_norm=args.batch_norm).to(device)
                model_name = 'gin'
            elif args.gnn_type == 'gcn':
                model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, batch_norm=args.batch_norm, pred=True,
                            use_embeddings=False, hidden_dim=args.hidden_dim, pool=args.global_pooling,
                            num_layers=args.num_layers, drop_ratio=args.dropout_ratio).to(device)
                model_name = 'gcn'
            else:
                raise ValueError('Invalid GNN model specified.')

            print('---- Target: Training of ' + model_name + ' multitask model for all QM9 properties with split ' + args.pretrain_subset +
                  ' for functional groups in ' + indices_names[idx] + ' ---- ', flush=True)

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
            model_filename = (indices_names[idx] + '_' + args.features + '_' + args.pretrain_subset +
                                               '_' + args.global_pooling +
                                               '_' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' +
                                               str(args.batch_norm) + '_' + args.gnn_type + ".pt")
            if not os.path.exists("models"):
                os.makedirs("models")
            if not os.path.exists("models/fgs_pretrain"):
                os.makedirs("models/fgs_pretrain")
            model_path = 'models/fgs_pretrain/' + model_filename
            torch.save(model_state, model_path)

    elif args.pretrain_subset == "full":
        torch_geometric.seed_everything(args.seed)
        # Split dataset into train, validation, test set
        test_length = int(len(dataset) * test_percent)
        val_length = int(len(dataset) * val_percent)

        val_dataset = dataset[test_length:test_length + val_length]
        test_dataset = dataset[:test_length]
        # train_dataset = dataset

        print('Length of train dataset: ' + str(len(dataset)), flush=True)
        print('Length of validation dataset: ' + str(len(val_dataset)), flush=True)
        print('Length of test dataset: ' + str(len(test_dataset)), flush=True)

        num_tasks = dataset[0].y.shape[1]
        print("num tasks: ", num_tasks, flush=True)

        # Load the appropriate architecture based on which pretrained model is being considered:
        if args.gnn_type == 'ginconcat':
            model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                              drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                              hidden_dim=args.hidden_dim, num_layers=args.num_layers, pool=args.global_pooling,
                              batch_norm=args.batch_norm).to(device)
            model_name = 'ginconcat'
        elif args.gnn_type == 'gin':
            model = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                        drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                        hidden_dim=args.hidden_dim, num_layers=args.num_layers, pool=args.global_pooling,
                        batch_norm=args.batch_norm).to(device)
            model_name = 'gin'
        elif args.gnn_type == 'gcn':
            model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, batch_norm=args.batch_norm, pred=True,
                        use_embeddings=False, hidden_dim=args.hidden_dim, pool=args.global_pooling,
                        num_layers=args.num_layers, drop_ratio=args.dropout_ratio).to(device)
            model_name = 'gcn'
        else:
            raise ValueError('Invalid GNN model specified.')

        print('---- Target: Training of ' + model_name + ' multitask model for all QM9 properties on the complete '
                                                         'QM9 dataset ---- ', flush=True)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  pin_memory_device='cuda', num_workers=args.num_workers)

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, device, optimizer, train_loader)
            print(f'Epoch: {epoch}, Training Loss: {train_loss}', flush=True)

        model_state = model.state_dict()
        # Save the model
        model_filename = (args.features + '_' + args.pretrain_subset +
                                           '_' + args.global_pooling +
                                           '_' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' +
                                           str(args.batch_norm) + '_' + args.gnn_type + ".pt")
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("models/fgs_pretrain"):
            os.makedirs("models/fgs_pretrain")
        model_path = 'models/fgs_pretrain/' + model_filename
        torch.save(model_state, model_path)


if __name__ == "__main__":
    main()
