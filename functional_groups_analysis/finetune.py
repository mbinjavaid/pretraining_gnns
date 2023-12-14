import pickle
import sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import argparse
import glob
import pandas as pd
import os
import torch_geometric
from model import GIN, GINConcat, GCN
from loader_chem import DownstreamDataset as DownstreamDatasetChem
from loader_fge import DownstreamDataset as DownstreamDatasetFGE


def train(model, device, optimizer, loader):
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


def test(model, device, mean, std, loader):
    model.eval()
    val_loss_all = 0
    mse_all = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            tmp_real_data = data.y
            # Re-scale data back to original scale:
            tmp_real_data = tmp_real_data * std + mean
            tmp_real_data_mod = tmp_real_data[~torch.isinf(data.y)].view(-1, 1)
            tmp_pred_data = model(data.x, data.edge_index, data.batch)
            # Re-scale predictions to original scale:
            tmp_pred_data = tmp_pred_data * std + mean
            tmp_pred_data_mod = tmp_pred_data[~torch.isinf(data.y)].view(-1, 1)
            mse_all += F.mse_loss(tmp_pred_data_mod, tmp_real_data_mod, reduction='sum').item()
            val_loss_all += F.l1_loss(tmp_pred_data_mod, tmp_real_data_mod, reduction='sum').item()

    return val_loss_all / len(loader.dataset), np.sqrt(mse_all / len(loader.dataset))


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
                        help='embedding dimensions - only considered if use_pretrained is 0 (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio - dropout only implemented for GNN type gin and ginconcat. (default: 0.5)')
    parser.add_argument('--global_pooling', type=str, default="sum",
                        help='global pooling - only considered if use_pretrained is 0 (sum, mean - default: sum)')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help="ginconcat, gin, or gcn - only considered if use_pretrained is 0 - default: gin")
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='whether to use batch norm - only considered if use_pretrained is 0 (default: 1).')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers - only considered if use_pretrained is 0 (default: 5)')
    parser.add_argument('--features', type=str, default="chem",
                        help="chemical features (chem) or functional group encodings (fge) - only considered if use_pretrained is 0 - default: chem")
    parser.add_argument('--use_pretrained', type=int, default=1,
                        help="use pretrained models (1) or not (0) - default: 1")
    parser.add_argument('--split', type=str, default="fg",
                        help="random or fg split - default: fg")
    parser.add_argument('--seed', type=int, default=321, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    args = parser.parse_args()

    # Load the functional group split indices for each property:
    try:
        with open('functional_groups/ait_test_indices.pkl', 'rb') as f:
            ait_test_indices = pickle.load(f)
        with open('functional_groups/hf_test_indices.pkl', 'rb') as f:
            hf_test_indices = pickle.load(f)
        with open('functional_groups/ld50_test_indices.pkl', 'rb') as f:
            ld50_test_indices = pickle.load(f)
        with open('functional_groups/lmv_test_indices.pkl', 'rb') as f:
            lmv_test_indices = pickle.load(f)
        with open('functional_groups/logp_test_indices.pkl', 'rb') as f:
            logp_test_indices = pickle.load(f)
        with open('functional_groups/osha_twa_test_indices.pkl', 'rb') as f:
            osha_twa_test_indices = pickle.load(f)
        with open('functional_groups/tb_test_indices.pkl', 'rb') as f:
            tb_test_indices = pickle.load(f)
        with open('functional_groups/tm_test_indices.pkl', 'rb') as f:
            tm_test_indices = pickle.load(f)
    except FileNotFoundError:
        print("Test indices not found. Please run functional_group_split.py first.")
        sys.exit()

    target_properties_downstream = ['AiT', 'Hf', 'LD50', 'Lmv', 'LogP', 'OSHA_TWA', 'Tb', 'Tm']
    abbreviations = ['ait', 'hf', 'ld50', 'lmv', 'logp', 'osha', 'tb', 'tm']
    dataset_test_indices = [ait_test_indices, hf_test_indices, ld50_test_indices, lmv_test_indices, logp_test_indices,
                            osha_twa_test_indices, tb_test_indices, tm_test_indices]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_geometric.seed_everything(args.seed)

    # num_tasks = 1

    if args.use_pretrained:
        pretrained_model_list = glob.glob('models/fgs_pretrain/*.pt')

        if len(pretrained_model_list) == 0:
            print("No pretrained model found. Please pre-train a model first by running pretrain.py.")
            sys.exit()

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/functional_groups'):
        os.makedirs('results/functional_groups')

    for dataset_idx, target_property_downstream in enumerate(target_properties_downstream):
        if args.features == 'fge':
            dataset = DownstreamDatasetFGE('Train/', dataset_name=target_property_downstream.lower())
        elif args.features == 'chem':
            dataset = DownstreamDatasetChem('Train/', dataset_name=target_property_downstream.lower())

        num_tasks = dataset[0].y.shape[1]
        print("num tasks: ", num_tasks, flush=True)

        # Extract the mean and standard deviation of the target property from the dataset:
        norm_percent = 0
        norm_length = int(len(dataset) * norm_percent)
        mean_lst = []
        std_lst = []
        for target in range(0, len(dataset.data.y[norm_length])):
            tmp_target_data = dataset.data.y[norm_length:, target]
            mean_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).mean())
            std_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).std())

        # Normalize targets to mean = 0 and std = 1
        mean = torch.tensor(mean_lst, dtype=torch.float)
        std = torch.tensor(std_lst, dtype=torch.float)

        print('Target data mean: ' + str(mean.tolist()), flush=True)
        print('Target data standard deviation: ' + str(std.tolist()), flush=True)

        mean_gpu = mean.to(device)
        std_gpu = std.to(device)

        if args.split == 'fg':
            val_indices_all_fgs = dataset_test_indices[dataset_idx]
        elif args.split == 'random':
            torch_geometric.seed_everything(args.seed)
            np.random.seed(args.seed)
            val_indices_all_fgs = [[] for _ in range(0, len(dataset_test_indices[dataset_idx]))]
            for fold in range(0, len(dataset_test_indices[dataset_idx])):
                chosen_indices = []
                for _ in range(len(dataset_test_indices[dataset_idx][fold])):
                    chosen_index = np.random.randint(0, len(dataset))
                    # Make sure the same index is not chosen more than once:
                    while chosen_index in chosen_indices:
                        chosen_index = np.random.randint(0, len(dataset))
                    val_indices_all_fgs[fold].append(chosen_index)
                    chosen_indices.append(chosen_index)
        else:
            raise ValueError('Invalid split specified (must be random or fg).')

        if args.use_pretrained:
            # In pretrained_model_list, find the index of the pretrained model that was trained for the current property
            # (i.e. the one with the dataset name in its filename):
            pretrained_specific_dataset = [s for s in pretrained_model_list if abbreviations[dataset_idx] in s]
            pretrained_full = [s for s in pretrained_model_list if 'full' in s]
            # join the two lists:
            pretrained_full.extend(pretrained_specific_dataset)
            # print(pretrained_full)
            for pretrained_model in pretrained_full:
                df = pd.DataFrame(
                    columns=['pretraining', 'features', 'gnn_type', 'pooling', 'hidden_dim', 'num_layers', 'batch_norm',
                             'dataset', 'average_rmse', 'median_rmse', 'minimum_rmse',
                             'std_rmse', 'cv'])
                # Get the filename of the pretrained model:
                pretrained_model_name = os.path.basename(pretrained_model)
                # remove the .pt extension:
                pretrained_model_name = pretrained_model_name[:-3]
                # print(pretrained_model_name)
                # split based on _:
                pretrained_model_params = pretrained_model_name.split('_')
                if 'full' not in pretrained_model_name:
                    downstream_dataset = pretrained_model_params[0]
                    features = pretrained_model_params[1]
                    pretrain_subset = pretrained_model_params[2]
                    global_pooling = pretrained_model_params[3]
                    hidden_dim = int(pretrained_model_params[4])
                    num_layers = int(pretrained_model_params[5])
                    batch_norm = int(pretrained_model_params[6])
                    gnn_type = pretrained_model_params[7]
                else:
                    features = pretrained_model_params[0]
                    pretrain_subset = pretrained_model_params[1]
                    global_pooling = pretrained_model_params[2]
                    hidden_dim = int(pretrained_model_params[3])
                    num_layers = int(pretrained_model_params[4])
                    batch_norm = int(pretrained_model_params[5])
                    gnn_type = pretrained_model_params[6]

                if features == 'fge':
                    dataset = DownstreamDatasetFGE('Train/', dataset_name=target_property_downstream.lower())
                elif features == 'chem':
                    dataset = DownstreamDatasetChem('Train/', dataset_name=target_property_downstream.lower())
                else:
                    raise ValueError('Invalid features specified in pretrained model name (must be fge or chem).')

                dataset.data.y = (dataset.data.y - mean) / std

                state_dict_pretrained = torch.load(pretrained_model)

                print("Fine tuning on the dataset " + target_property_downstream + " with pretrained model: "
                      + pretrained_model_name + " and using " + args.split + " split.", flush=True)

                # Initialize lists to store RMSE values for each fold
                all_folds_val_rmses = []
                all_folds_val_maes = []

                for fold, val_indices in enumerate(val_indices_all_fgs):
                    torch_geometric.seed_everything(args.seed)
                    train_indices = [index for index in range(len(dataset)) if index not in val_indices]

                    # Load the appropriate architecture based on which pretrained model is being considered:
                    if gnn_type == 'ginconcat':
                        model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                                          drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                                          hidden_dim=hidden_dim, num_layers=num_layers,
                                          pool=global_pooling, batch_norm=batch_norm).to(device)
                        model_name = 'ginconcat'
                    elif gnn_type == 'gin':
                        model = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                                    drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                                    hidden_dim=hidden_dim, num_layers=num_layers, pool=global_pooling,
                                    batch_norm=batch_norm).to(device)
                        model_name = 'gin'
                    elif gnn_type == 'gcn':
                        model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, batch_norm=batch_norm,
                                    pred=True,
                                    use_embeddings=False, hidden_dim=hidden_dim, pool=global_pooling,
                                    num_layers=num_layers, drop_ratio=args.dropout_ratio).to(device)
                        model_name = 'gcn'
                    else:
                        raise ValueError('GNN type not recognized (must be ginconcat, gin, or gcn).')

                    # Create train and validation datasets for the current fold
                    train_dataset = dataset[list(train_indices)]
                    val_dataset = dataset[list(val_indices)]

                    # Re-initialize the parameters of the GNN layers using the pre-trained model, for each new fold
                    model_state_dict = model.state_dict()
                    for name, param in state_dict_pretrained.items():
                        if ('graph_prediction' not in name) and ('mlp' not in name):
                            model_state_dict[name] = param
                    # Update the model's state dictionary with the GCN layer parameters
                    model.load_state_dict(model_state_dict)

                    # Define optimizer
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

                    # Create train and validation loaders
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                              pin_memory_device='cuda', num_workers=args.num_workers)
                    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True,
                                            pin_memory_device='cuda')

                    print('Finetuning model ' + pretrained_model_name + ' for property ' + target_property_downstream, flush=True)
                    print('Length of train dataset: ' + str(len(train_dataset)), flush=True)
                    print('Length of validation dataset: ' + str(len(val_dataset)), flush=True)
                    print("Num features: " + str(dataset.num_features), flush=True)

                    # Training loop
                    for epoch in range(1, args.epochs + 1):
                        train_rmse = train(model, device, optimizer, train_loader)
                        print(f'Downstream Property: {target_property_downstream}, '
                              f'Fold: {fold+1}, Epoch: {epoch}, Training Loss: {train_rmse}', flush=True)

                    # After training finished, check test performance of the model on the current fold
                    val_mae, val_rmse = test(model, device, mean_gpu, std_gpu, val_loader)
                    print("Test MAE for fold " + str(fold+1) + " of property " + target_property_downstream +
                          ": " + str(val_mae), flush=True)
                    print("Test RMSE for fold " + str(fold+1) + " of property " + target_property_downstream +
                          ": " + str(val_rmse), flush=True)

                    all_folds_val_rmses.append(val_rmse)
                    all_folds_val_maes.append(val_mae)

                # Stats over all folds of the current run:
                std_rmse = np.std(all_folds_val_rmses)
                average_rmse = np.mean(all_folds_val_rmses)
                median_rmse = np.median(all_folds_val_rmses)
                minimum_rmse = np.min(all_folds_val_rmses)
                cv = (std_rmse / average_rmse) * 100
                cv_mae = (np.std(all_folds_val_maes) / np.mean(all_folds_val_maes)) * 100

                print("Test Stats for property " + target_property_downstream + " using model " + pretrained_model_name +
                      "  over all 10 folds:\n", flush=True)
                print("All RMSEs: " + str(all_folds_val_rmses), flush=True)
                print("All MAEs: " + str(all_folds_val_maes), flush=True)
                print('Standard deviation (RMSE): ' + str(std_rmse) + ' (MAE): ' + str(np.std(all_folds_val_maes)), flush=True)
                print('Average RMSE: ' + str(average_rmse) + ' MAE: ' + str(np.mean(all_folds_val_maes)), flush=True)
                print('Median RMSE: ' + str(median_rmse) + ' MAE: ' + str(np.median(all_folds_val_maes)), flush=True)
                print('Minimum RMSE: ' + str(minimum_rmse) + ' MAE: ' + str(np.min(all_folds_val_maes)), flush=True)
                print('Coefficient of variation (RMSE): ' + str(cv) + ' (MAE): ' + str(cv_mae), flush=True)
                print("\n", flush=True)

                df.loc[len(df)] = [pretrain_subset, features, model_name, global_pooling, hidden_dim, num_layers,
                                   batch_norm, target_property_downstream, average_rmse, median_rmse, minimum_rmse,
                                   std_rmse, cv]

                all_folds_val_rmses = []
                all_folds_val_maes = []

                # Write results
                # Save df to csv (append, and only include header if it's the first time)
                if os.path.exists('results/functional_groups/' + features + '_' + args.split + '_split.csv'):
                    df.to_csv('results/functional_groups/' + features + '_' + args.split + '_split.csv', mode='a', header=False,
                              index=False)
                else:
                    df.to_csv('results/functional_groups/' + features + '_' + args.split + '_split.csv', index=False)

        else:
            # Just fine-tune without using any pretrained models:
            df = pd.DataFrame(
                columns=['pretraining', 'features', 'gnn_type', 'pooling', 'hidden_dim', 'num_layers', 'batch_norm',
                         'dataset', 'average_rmse', 'median_rmse', 'minimum_rmse',
                         'std_rmse', 'cv'])

            if args.features == 'fge':
                dataset = DownstreamDatasetFGE('Train/', dataset_name=target_property_downstream.lower())
            elif args.features == 'chem':
                dataset = DownstreamDatasetChem('Train/', dataset_name=target_property_downstream.lower())
            else:
                raise ValueError('Invalid features specified in pretrained model name (must be fge or chem).')

            dataset.data.y = (dataset.data.y - mean) / std

            print("Fine tuning on the dataset " + target_property_downstream + " without pretraining, using " +
                  args.split + " split.", flush=True)

            # Initialize lists to store RMSE values for each fold
            all_folds_val_rmses = []
            all_folds_val_maes = []

            for fold, val_indices in enumerate(val_indices_all_fgs):
                torch_geometric.seed_everything(args.seed)
                train_indices = [index for index in range(len(dataset)) if index not in val_indices]

                # Load the appropriate architecture based on which pretrained model is being considered:
                if args.gnn_type == 'ginconcat':
                    model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                                      drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                                      hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                      pool=args.global_pooling, batch_norm=args.batch_norm).to(device)
                    model_name = 'ginconcat'
                elif args.gnn_type == 'gin':
                    model = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                                drop_ratio=args.dropout_ratio, pred=True, use_embeddings=False,
                                hidden_dim=args.hidden_dim, num_layers=args.num_layers, pool=args.global_pooling,
                                batch_norm=args.batch_norm).to(device)
                    model_name = 'gin'
                elif args.gnn_type == 'gcn':
                    model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, batch_norm=args.batch_norm,
                                pred=True,
                                use_embeddings=False, hidden_dim=args.hidden_dim, pool=args.global_pooling,
                                num_layers=args.num_layers, drop_ratio=args.dropout_ratio).to(device)
                    model_name = 'gcn'
                else:
                    raise ValueError('GNN type not recognized (must be ginconcat, gin, or gcn).')

                # Create train and validation datasets for the current fold
                train_dataset = dataset[list(train_indices)]
                val_dataset = dataset[list(val_indices)]

                # Define optimizer
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

                # Create train and validation loaders
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                          pin_memory_device='cuda', num_workers=args.num_workers)
                val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True,
                                        pin_memory_device='cuda')

                print('Fine-tuning for property ' + target_property_downstream + ' without pretraining', flush=True)
                print('Length of train dataset: ' + str(len(train_dataset)), flush=True)
                print('Length of validation dataset: ' + str(len(val_dataset)), flush=True)
                print("Num features: " + str(dataset.num_features), flush=True)

                # Training loop
                for epoch in range(1, args.epochs + 1):
                    train_rmse = train(model, device, optimizer, train_loader)

                    print(f'Downstream Property: {target_property_downstream}, '
                          f'Fold: {fold + 1}, Epoch: {epoch}, Training Loss: {train_rmse}', flush=True)

                # After training finished, check test performance of the model on the current fold
                val_mae, val_rmse = test(model, device, mean_gpu, std_gpu, val_loader)
                print("Test MAE for fold " + str(fold + 1) + " of property " + target_property_downstream +
                      ": " + str(val_mae), flush=True)
                print("Test RMSE for fold " + str(fold + 1) + " of property " + target_property_downstream +
                      ": " + str(val_rmse), flush=True)

                all_folds_val_rmses.append(val_rmse)
                all_folds_val_maes.append(val_mae)

            # Stats over all folds of the current run:
            std_rmse = np.std(all_folds_val_rmses)
            average_rmse = np.mean(all_folds_val_rmses)
            median_rmse = np.median(all_folds_val_rmses)
            minimum_rmse = np.min(all_folds_val_rmses)
            cv = (std_rmse / average_rmse) * 100
            cv_mae = (np.std(all_folds_val_maes) / np.mean(all_folds_val_maes)) * 100

            print("Test Stats for property " + target_property_downstream + " using model " + model_name +
                  " (NOT PRETRAINED) over all 10 folds:\n", flush=True)
            print("All RMSEs: " + str(all_folds_val_rmses), flush=True)
            print("All MAEs: " + str(all_folds_val_maes), flush=True)
            print('Standard deviation (RMSE): ' + str(std_rmse) + ' (MAE): ' + str(np.std(all_folds_val_maes)),
                  flush=True)
            print('Average RMSE: ' + str(average_rmse) + ' MAE: ' + str(np.mean(all_folds_val_maes)), flush=True)
            print('Median RMSE: ' + str(median_rmse) + ' MAE: ' + str(np.median(all_folds_val_maes)), flush=True)
            print('Minimum RMSE: ' + str(minimum_rmse) + ' MAE: ' + str(np.min(all_folds_val_maes)), flush=True)
            print('Coefficient of variation (RMSE): ' + str(cv) + ' (MAE): ' + str(cv_mae), flush=True)
            print("\n", flush=True)

            df.loc[len(df)] = ['No Pretraining', args.features, model_name, args.global_pooling, args.hidden_dim, args.num_layers,
                               args.batch_norm, target_property_downstream, average_rmse, median_rmse, minimum_rmse,
                               std_rmse, cv]

            all_folds_val_rmses = []
            all_folds_val_maes = []

            # Write results
            # Save df to csv (append, and only include header if it's the first time)
            if os.path.exists('results/functional_groups/' + args.features + '_' + args.split + '_split.csv'):
                df.to_csv('results/functional_groups/' + args.features + '_' + args.split + '_split.csv', mode='a', header=False,
                          index=False)
            else:
                df.to_csv('results/functional_groups/' + args.features + '_' + args.split + '_split.csv', index=False)


if __name__ == "__main__":
    main()
