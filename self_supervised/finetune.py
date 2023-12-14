import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import argparse
import glob
import os
import torch_geometric
from model import GINConcat, GIN, GCN


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
    parser.add_argument('--target_properties_downstream', type=str, default=['AiT', 'Hf', 'LD50', 'Lmv',
                                                                             'LogP', 'OSHA_TWA', 'Tb', 'Tm'],
                        help='List of downstream dataset names to train and cross-validate on, one by one. Default:'
                             ' AiT Hf LD50 Lmv LogP OSHA_TWA Tb Tm', nargs='*')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio - dropout only implemented for GNN type gin and ginconcat. (default: 0.5)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers - only considered with strategy no_pretrain (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='dimensionality of hidden layers - only considered with strategy no_pretrain (default: 300)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='batch normalization (1) or not (0) - only considered with strategy no_pretrain (default: 1)')
    parser.add_argument('--gnn_type', type=str, default="gin", help="gin, ginconcat, "
                                                                             "gcn - only used "
                                                                             "with strategy no_pretrain. Default: gin")
    parser.add_argument('--strategy', type=str, default='sup_only',
                        help='which pretraining to use: masking, masking_sup, contextpred, contextpred_sup, sup_only, or no_pretrain (default: sup_only).')
    parser.add_argument('--global_pooling', type=str, default="sum",
                        help='global pooling - only considered with strategy no_pretrain, contextpred, or masking, '
                             'otherwise uses the same setting as the pretrained model (sum, mean - default: sum)')
    parser.add_argument('--num_folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=321, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--use_embeddings', type=int, default=0, help='Use feature embeddings (1) or '
                                                                      'multi-hot encoding (0) - will only finetune pretrained models which have the same specified use_embeddings setting. (default: 0)')
    args = parser.parse_args()

    if args.use_embeddings:
        from loader_embeddings import DownstreamDataset
        print('Using feature embeddings', flush=True)
    else:
        from loader_multihot import DownstreamDataset
        print('Using multi-hot encodings', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_geometric.seed_everything(args.seed)

    # num_tasks = 1

    if args.strategy == 'masking':
        # find .pt files with 'masking' in the filename:
        if args.use_embeddings:
            pretrained_model_list = glob.glob('models/embeddings_pretrain_masking/*masking*.pt')
        else:
            pretrained_model_list = glob.glob('models/multihot_pretrain_masking/*masking*.pt')
    elif args.strategy == 'masking_sup':
        # find .pt files with 'masking_sup' in the filename:
        if args.use_embeddings:
            pretrained_model_list = glob.glob('models/embeddings_pretrain_supervised/*masking*.pt')
        else:
            pretrained_model_list = glob.glob('models/multihot_pretrain_supervised/*masking*.pt')
    elif args.strategy == 'contextpred':
        if args.use_embeddings:
            pretrained_model_list = glob.glob('models/embeddings_pretrain_contextpred/*contextpred*.pt')
        else:
            pretrained_model_list = glob.glob('models/multihot_pretrain_contextpred/*contextpred*.pt')
    elif args.strategy == 'contextpred_sup':
        if args.use_embeddings:
            pretrained_model_list = glob.glob('models/embeddings_pretrain_supervised/*contextpred*.pt')
        else:
            pretrained_model_list = glob.glob('models/multihot_pretrain_supervised/*contextpred*.pt')
    elif args.strategy == 'sup_only':
        if args.use_embeddings:
            pretrained_model_list = glob.glob('models/embeddings_pretrain_supervised/*sup_only*.pt')
        else:
            pretrained_model_list = glob.glob('models/multihot_pretrain_supervised/*sup_only*.pt')
    elif args.strategy == 'no_pretrain':
        pretrained_model_list = ['no_pretrain']
    else:
        raise ValueError('Invalid strategy specified.')

    if not pretrained_model_list:
        raise FileNotFoundError('No pretrained models found for the specified strategy.')

    for pretrained_model in pretrained_model_list:
        if pretrained_model != 'no_pretrain':
            pretrained_model_name = os.path.splitext(os.path.basename(pretrained_model))[0]
            state_dict_pretrained = torch.load(pretrained_model)
            if args.strategy == 'masking' or args.strategy == 'contextpred':
                # Split the filename into its components:
                pretrained_model_params = pretrained_model_name.split('_')
                global_pooling = args.global_pooling
                hidden_dim = int(pretrained_model_params[0])
                num_layers = int(pretrained_model_params[1])
                batch_norm = int(pretrained_model_params[2])
                gnn_type = pretrained_model_params[3]
            elif args.strategy == 'masking_sup' or args.strategy == 'contextpred_sup' or args.strategy == 'sup_only':
                # Split the filename into its components:
                pretrained_model_params = pretrained_model_name.split('_')
                global_pooling = str(pretrained_model_params[0])
                hidden_dim = int(pretrained_model_params[1])
                num_layers = int(pretrained_model_params[2])
                batch_norm = int(pretrained_model_params[3])
                gnn_type = pretrained_model_params[4]
        elif pretrained_model == 'no_pretrain':
            pretrained_model_name = 'no_pretrain'
            global_pooling = args.global_pooling
            hidden_dim = args.hidden_dim
            num_layers = args.num_layers
            batch_norm = args.batch_norm
            gnn_type = args.gnn_type
            state_dict_pretrained = None

        print("Using the following pretrained model: " + pretrained_model_name, flush=True)
        print("Parameters: hidden_dim: " + str(hidden_dim) + " num_layers: " + str(num_layers) +
              " global_pooling: " + global_pooling + " batch_norm: " + str(batch_norm), flush=True)

        df = pd.DataFrame(columns=['strategy', 'gnn_type', 'pooling', 'hidden_dim', 'num_layers', 'batch_norm',
                                   'dataset', 'average_rmse', 'median_rmse', 'minimum_rmse',
                                   'std_rmse', 'cv'])
        for target_property_downstream in args.target_properties_downstream:
            dataset = DownstreamDataset('Train/', dataset_name=target_property_downstream.lower())
            print("Fine tuning on the dataset " + target_property_downstream + " with pretrained model: "
                  + pretrained_model_name, flush=True)

            num_tasks = dataset[0].y.shape[1]
            print("num tasks: ", num_tasks, flush=True)

            norm_percent = 0
            norm_length = int(len(dataset) * norm_percent)
            mean_lst = []
            std_lst = []
            for target in range(0, len(dataset.data.y[norm_length])):
                tmp_target_data = dataset.data.y[norm_length:,target]
                mean_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).mean())
                std_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).std())

            # Normalize targets to mean = 0 and std = 1
            mean = torch.tensor(mean_lst, dtype=torch.float)
            std = torch.tensor(std_lst, dtype=torch.float)
            dataset.data.y = (dataset.data.y - mean) / std

            print('Target data mean: ' + str(mean.tolist()), flush=True)
            print('Target data standard deviation: ' + str(std.tolist()), flush=True)

            mean = mean.to(device)
            std = std.to(device)

            # Perform k-fold cross-validation with a random seed for shuffling the dataset:
            k_fold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

            # Initialize lists to store RMSE values for each fold
            all_folds_val_rmses = []
            all_folds_val_maes = []
            for fold, (train_indices, val_indices) in enumerate(k_fold.split(dataset)):
                torch_geometric.seed_everything(args.seed)

                # Create train and validation datasets for the current fold
                train_dataset = dataset[list(train_indices)]
                val_dataset = dataset[list(val_indices)]

                # Load the appropriate architecture based on which pretrained model is being considered:
                if gnn_type == 'ginconcat':
                    model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features, pool=global_pooling,
                                      drop_ratio=args.dropout_ratio, batch_norm=batch_norm,
                                      hidden_dim=hidden_dim, num_layers=num_layers, pred=True,
                                      use_embeddings=args.use_embeddings).to(device)
                elif gnn_type == 'gin':
                    model = GIN(num_tasks=num_tasks, num_features=dataset.num_features, pool=global_pooling,
                                      drop_ratio=args.dropout_ratio, batch_norm=batch_norm,
                                      hidden_dim=hidden_dim, num_layers=num_layers, pred=True,
                                      use_embeddings=args.use_embeddings).to(device)
                elif gnn_type == 'gcn':
                    model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, pool=global_pooling,
                                      drop_ratio=args.dropout_ratio, batch_norm=batch_norm,
                                      hidden_dim=hidden_dim, num_layers=num_layers, pred=True,
                                      use_embeddings=args.use_embeddings).to(device)
                else:
                    raise ValueError('Invalid GNN model specified.')

                if state_dict_pretrained is not None:
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

                # print('Length of train dataset: ' + str(len(train_dataset)), flush=True)
                # print('Length of validation dataset: ' + str(len(val_dataset)), flush=True)
                print('Using initialization ' + pretrained_model_name + ' for dataset ' + target_property_downstream,
                      flush=True)

                # Training loop
                for epoch in range(1, args.epochs + 1):
                    train_rmse = train(model, device, optimizer, train_loader)

                    print(f'Downstream Property: {target_property_downstream}, '
                          f'Fold: {fold+1}, Epoch: {epoch}, Training Loss: {train_rmse}', flush=True)

                # After training finished, check test performance of the model on the current fold
                val_mae, val_rmse = test(model, device, mean, std, val_loader)
                print("Test MAE for fold " + str(fold+1) + " of property " + target_property_downstream +
                      ": " + str(val_mae), flush=True)
                print("Test RMSE for fold " + str(fold+1) + " of property " + target_property_downstream +
                      ": " + str(val_rmse), flush=True)

                all_folds_val_rmses.append(val_rmse)
                all_folds_val_maes.append(val_mae)

            # Stats over all folds:
            std_rmse = np.std(all_folds_val_rmses)
            average_rmse = np.mean(all_folds_val_rmses)
            median_rmse = np.median(all_folds_val_rmses)
            minimum_rmse = np.min(all_folds_val_rmses)
            cv = (std_rmse / average_rmse) * 100
            cv_mae = (np.std(all_folds_val_maes) / np.mean(all_folds_val_maes)) * 100

            print("Test Stats for dataset " + target_property_downstream + " using strategy " + pretrained_model_name +
                  " over all 10 folds:\n", flush=True)
            print("All RMSEs: " + str(all_folds_val_rmses), flush=True)
            print("All MAEs: " + str(all_folds_val_maes), flush=True)
            print('Standard deviation (RMSE): ' + str(std_rmse) + ' (MAE): ' + str(np.std(all_folds_val_maes)), flush=True)
            print('Average RMSE: ' + str(average_rmse) + ' MAE: ' + str(np.mean(all_folds_val_maes)), flush=True)
            print('Median RMSE: ' + str(median_rmse) + ' MAE: ' + str(np.median(all_folds_val_maes)), flush=True)
            print('Minimum RMSE: ' + str(minimum_rmse) + ' MAE: ' + str(np.min(all_folds_val_maes)), flush=True)
            print('Coefficient of variation (RMSE): ' + str(cv) + ' (MAE): ' + str(cv_mae), flush=True)
            print("\n", flush=True)

            # Write a csv file with the results over all 10 folds. Save the pretrained_model_name as the filename.
            # Inside the file, write the strategy, gnn_type, pooling, hidden_dim, num_layers, batch_norm,
            # dataset,average_rmse, median_rmse, minimum_rmse, std_rmse, and cv in a row.
            # Write this to the df dataframe created earlier.

            if "contextpred" in pretrained_model_name:
                current_strategy = "ContextPred"
            elif "masking_all" in pretrained_model_name:
                current_strategy = "AttrMask (Atom, FG)"
            elif "masking_func" in pretrained_model_name:
                current_strategy = "AttrMask (FG)"
            elif "masking_atom" in pretrained_model_name:
                current_strategy = "AttrMask (Atom)"
            if "_sup" in args.strategy:
                current_strategy = current_strategy + " + Supervised"
            if args.strategy == "sup_only":
                current_strategy = "Supervised"
            if args.strategy == "no_pretrain":
                current_strategy = "No Pretraining"
            df.loc[len(df)] = [current_strategy, gnn_type, global_pooling, hidden_dim, num_layers, batch_norm,
                               target_property_downstream, average_rmse, median_rmse, minimum_rmse,
                               std_rmse, cv]

            all_folds_val_rmses = []
            all_folds_val_maes = []

        # Save the dataframe as a csv file
        name = 'embeddings' if args.use_embeddings else 'multihot'
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('results/self_supervised'):
            os.makedirs('results/self_supervised')
        # Save df to csv (append, and only include header if it's the first time)
        if os.path.exists('results/self_supervised/' + args.strategy + '_' + name + '.csv'):
            df.to_csv('results/self_supervised/' + args.strategy + '_' + name + '.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('results/self_supervised/' + args.strategy + '_' + name + '.csv', index=False)


if __name__ == "__main__":
    main()
