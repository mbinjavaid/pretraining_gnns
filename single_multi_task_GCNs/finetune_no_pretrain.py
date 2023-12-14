import argparse
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric
from loader import DownstreamDataset
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


def test(model, loader, device, mean, std):
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
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--global_pooling', type=str, default="sum",
                        help='global pooling (sum, mean - default: sum)')
    parser.add_argument('--num_folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--batch_norm', type=int, default=1, help='Batch Normalization off (0) or on (1) (default: 1)')
    parser.add_argument('--seed', type=int, default=321, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.DataFrame(columns=['pretraining', 'pooling', 'hidden_dim', 'num_layers', 'batch_norm',
                               'dataset', 'average_rmse', 'median_rmse', 'minimum_rmse',
                               'std_rmse', 'cv'])

    for target_property_downstream in args.target_properties_downstream:
        dataset = DownstreamDataset('Train/', dataset_name=target_property_downstream.lower())
        print("Training on the dataset " + target_property_downstream + " without pre-training -- Pooling: "
              + args.global_pooling, flush=True)

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

        num_tasks = dataset[0].y.shape[1]
        print("num tasks: ", num_tasks, flush=True)

        # Perform k-fold cross-validation with a random seed for shuffling the dataset before splitting it into folds:
        k_fold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

        # Initialize lists to store RMSE values for each fold
        all_folds_val_rmses = []
        all_folds_val_maes = []
        for fold, (train_indices, val_indices) in enumerate(k_fold.split(dataset)):
            torch_geometric.seed_everything(args.seed)
            # Create train and validation datasets for the current fold
            train_dataset = dataset[list(train_indices)]
            val_dataset = dataset[list(val_indices)]

            model = GCN(num_tasks=num_tasks, num_features=dataset.num_features, batch_norm=args.batch_norm,
                        num_layers=args.num_layers, pool=args.global_pooling, hidden_dim=args.hidden_dim,
                        pred=True).to(device)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

            # Create train and validation loaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      pin_memory_device='cuda', num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True,
                                    pin_memory_device='cuda')

            print('Length of train dataset: ' + str(len(train_dataset)), flush=True)
            print('Length of validation dataset: ' + str(len(val_dataset)), flush=True)

            # Training loop
            for epoch in range(1, args.epochs + 1):
                train_rmse = train(model, optimizer, train_loader, device)
                # if epoch % 100 == 0:
                #     print(f'Downstream Property: {target_property_downstream}, '
                #           f'Fold: {fold+1}, Epoch: {epoch}, Training Loss: {train_rmse}', flush=True)

            # After training finished, check test performance of the model on the current fold
            val_mae, val_rmse = test(model, val_loader, device, mean, std)
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

        print("Test Stats for property " + target_property_downstream + " pretrained on all QM9 properties over"
                                                                        " all 10 folds:\n", flush=True)
        print("All RMSEs: " + str(all_folds_val_rmses), flush=True)
        print("All MAEs: " + str(all_folds_val_maes), flush=True)
        print('Standard deviation (RMSE): ' + str(std_rmse) + ' (MAE): ' + str(np.std(all_folds_val_maes)), flush=True)
        print('Average RMSE: ' + str(average_rmse) + ' MAE: ' + str(np.mean(all_folds_val_maes)), flush=True)
        print('Median RMSE: ' + str(median_rmse) + ' MAE: ' + str(np.median(all_folds_val_maes)), flush=True)
        print('Minimum RMSE: ' + str(minimum_rmse) + ' MAE: ' + str(np.min(all_folds_val_maes)), flush=True)
        print('Coefficient of variation (RMSE): ' + str(cv) + ' (MAE): ' + str(cv_mae), flush=True)
        print("\n", flush=True)

        all_folds_val_rmses = []
        all_folds_val_maes = []

        print(len(df))
        df.loc[len(df)] = ['No Pretraining', args.global_pooling, args.hidden_dim, args.num_layers, args.batch_norm,
                           target_property_downstream, average_rmse, median_rmse, minimum_rmse,
                           std_rmse, cv]

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/single_multi_task'):
        os.makedirs('results/single_multi_task')
    if os.path.exists('results/single_multi_task/gcn_no_pretrain.csv'):
        df.to_csv('results/single_multi_task/gcn_no_pretrain.csv', index=False, mode='a', header=False)
    else:
        df.to_csv('results/single_multi_task/gcn_no_pretrain.csv', index=False)


if __name__ == "__main__":
    main()
