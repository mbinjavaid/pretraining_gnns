import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch_geometric
from model import GINConcat, GIN, GCN


def pool_func(x, batch, mode="mean"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    else:
        raise ValueError("Invalid pooling function! (should be sum or mean)")


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


criterion = nn.BCEWithLogitsLoss()
num_tasks = 1


def train(args, model_substruct, model_context, loader, optimizer_substruct, optimizer_context, device, num_layers):
    model_substruct.train()
    model_context.train()

    balanced_loss_accum = 0
    acc_accum = 0
    steps = 0
    for data in loader:
        data = data.to(device)
        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()
        # creating substructure representation
        substruct_rep = model_substruct(data.x_substruct, data.edge_index_substruct,
                                        data.batch)[data.center_substruct_idx]

        # creating context representations
        overlapped_node_rep = model_context(data.x_context, data.edge_index_context, data.batch)[
            data.overlap_context_substruct_idx]

        # # If using GIN with concatenation of features across different layers, then we discard
        # # an appropriate number of initial layer representations such that the number of node representations
        # # are equal for substructure and context, so they can be compared later. (Substructure network has
        # # a different number of layers than context network)
        if 'ginconcat' in args.gnn_type and args.csize > num_layers:
            overlapped_node_rep = overlapped_node_rep[:, -num_layers * int(substruct_rep.shape[1]/num_layers):]
        elif 'ginconcat' in args.gnn_type and args.csize < num_layers:
            substruct_rep = substruct_rep[:, -args.csize * int(overlapped_node_rep.shape[1]/args.csize):]

        # print(-args.csize * int(overlapped_node_rep.shape[1]/args.csize))

        # Contexts are represented by
        if args.mode == "cbow":
            # positive context representation
            context_rep = pool_func(overlapped_node_rep, data.batch_overlapped_context, mode=args.context_pooling)
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_context_rep = torch.cat(
                [context_rep[cycle_index(len(context_rep), i + 1)] for i in range(args.neg_samples)], dim=0)

            pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
            pred_neg = torch.sum(substruct_rep.repeat((args.neg_samples, 1)) * neg_context_rep, dim=1)

        elif args.mode == "skipgram":
            expanded_substruct_rep = torch.cat(
                [substruct_rep[i].repeat((data.overlapped_context_size[i], 1)) for i in range(len(substruct_rep))],
                dim=0)

            pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim=1)

            # shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(args.neg_samples):
                shifted_substruct_rep = substruct_rep[cycle_index(len(substruct_rep), i + 1)]
                shifted_expanded_substruct_rep.append(torch.cat(
                    [shifted_substruct_rep[i].repeat((data.overlapped_context_size[i], 1)) for i in
                     range(len(shifted_substruct_rep))], dim=0))

            shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim=0)
            pred_neg = torch.sum(shifted_expanded_substruct_rep * overlapped_node_rep.repeat((args.neg_samples, 1)),
                                 dim=1)

        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        loss = loss_pos + args.neg_samples * loss_neg
        loss.backward()

        optimizer_substruct.step()
        optimizer_context.step()

        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += 0.5 * (float(torch.sum(pred_pos > 0).detach().cpu().item()) / len(pred_pos) + float(
            torch.sum(pred_neg < 0).detach().cpu().item()) / len(pred_neg))
        steps += 1

    return balanced_loss_accum / steps, acc_accum / steps


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--csize', type=int, default=3,
                        help='context size (default: 3).')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio - dropout only implemented for GNN type gin and ginconcat. (default: 0)')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='number of negative contexts per positive context (default: 1)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='batch normalization (default: 1)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='dimensionality of hidden layers (default: 300)')
    parser.add_argument('--context_pooling', type=str, default="mean",
                        help='how the contexts are pooled (sum, mean, or max) - default: mean')
    parser.add_argument('--mode', type=str, default="cbow", help="cbow or skipgram")
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='name of dataset to use for pretraining (zinc, qm9) (default: qm9)')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help="ginconcat, gin, gcn - default: gin")
    parser.add_argument('--seed', type=int, default=321, help="Random seed.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--use_embeddings', type=int, default=0, help='Use feature embeddings (1) or '
                                                                      'multi-hot encoding (0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_geometric.seed_everything(args.seed)

    l1 = args.num_layers - 1
    l2 = l1 + args.csize

    print(args.mode)
    print("num layer: %d l1: %d l2: %d" % (args.num_layers, l1, l2))

    # set up dataset and transform function.
    if args.dataset == "qm9" and args.use_embeddings:
        from loader_embeddings import QM9
        from util_embeddings import ExtractSubstructureContextPair
        from dataloader_embeddings import DataLoaderSubstructContext
        dataset = QM9("Train/", transform=ExtractSubstructureContextPair(args.num_layers, l1, l2))
    elif args.dataset == "qm9" and not args.use_embeddings:
        from loader_multihot import QM9
        from util_multihot import ExtractSubstructureContextPair
        from dataloader_multihot import DataLoaderSubstructContext
        dataset = QM9("Train/", transform=ExtractSubstructureContextPair(args.num_layers, l1, l2))
    elif args.dataset == "zinc" and args.use_embeddings:
        from loader_embeddings import MoleculeDataset
        from util_embeddings import ExtractSubstructureContextPair
        from dataloader_embeddings import DataLoaderSubstructContext
        dataset = MoleculeDataset("Train/", transform=ExtractSubstructureContextPair(args.num_layers, l1, l2))
    elif args.dataset == "zinc" and not args.use_embeddings:
        from loader_multihot import MoleculeDataset
        from util_multihot import ExtractSubstructureContextPair
        from dataloader_multihot import DataLoaderSubstructContext
        dataset = MoleculeDataset("Train/", transform=ExtractSubstructureContextPair(args.num_layers, l1, l2))

    # Shuffle training data
    dataset = dataset.shuffle()
    loader = DataLoaderSubstructContext(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models, one for pre-training and one for context embeddings
    if args.gnn_type == "ginconcat":
        model_substruct = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                                    drop_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim,
                                    use_embeddings=args.use_embeddings, num_layers=args.num_layers,
                                    batch_norm=args.batch_norm).to(device)
        model_context = GINConcat(num_layers=int(l2 - l1), num_tasks=num_tasks, num_features=dataset.num_features,
                                  drop_ratio=args.dropout_ratio, use_embeddings=args.use_embeddings,
                                  hidden_dim=args.hidden_dim, batch_norm=args.batch_norm).to(device)
    elif args.gnn_type == "gin":
        model_substruct = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                              drop_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim,
                              use_embeddings=args.use_embeddings, num_layers=args.num_layers,
                              batch_norm=args.batch_norm).to(device)
        model_context = GIN(num_layers=int(l2 - l1), num_tasks=num_tasks, num_features=dataset.num_features,
                            drop_ratio=args.dropout_ratio, use_embeddings=args.use_embeddings,
                            hidden_dim=args.hidden_dim, batch_norm=args.batch_norm).to(device)
    elif args.gnn_type == "gcn":
        model_substruct = GCN(num_tasks=num_tasks, num_features=dataset.num_features,
                              drop_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim,
                              use_embeddings=args.use_embeddings, num_layers=args.num_layers,
                              batch_norm=args.batch_norm).to(device)
        model_context = GCN(num_layers=int(l2 - l1), num_tasks=num_tasks, num_features=dataset.num_features,
                            drop_ratio=args.dropout_ratio, use_embeddings=args.use_embeddings,
                            hidden_dim=args.hidden_dim, batch_norm=args.batch_norm).to(device)
    else:
        raise ValueError("Invalid GNN type (must be gin_concat, gin_modified, gcn_bn, or gcn_no_bn.")

    # set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(model_context.parameters(), lr=args.lr, weight_decay=args.decay)

    print("GNN type: " + args.gnn_type)
    print("Number of layers: " + str(args.num_layers))
    print("Hidden dim: " + str(args.hidden_dim))
    print("Batch norm: " + str(args.batch_norm))
    print("Context size: " + str(args.csize))
    print("Context pooling: " + str(args.context_pooling))
    print("Mode: " + str(args.mode))

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(args, model_substruct, model_context, loader, optimizer_substruct,
                                      optimizer_context, device, args.num_layers)
        print(train_loss, train_acc, flush=True)

    # Save model
    # output_file = args.dataset + "_" + args.gnn_type + "_contextpred.pt"
    output_file = (str(args.hidden_dim) + "_" + str(args.num_layers) + "_" +
                   str(args.batch_norm) + "_" + args.gnn_type + "_contextpred_" + str(args.csize) + "_" +
                   str(args.neg_samples) + "_" + args.context_pooling + "_" + args.mode + "_" + args.dataset + ".pt")
    if not os.path.exists("models"):
        os.makedirs("models")
    if args.use_embeddings:
        model_path = 'models/embeddings_pretrain_contextpred'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    else:
        model_path = 'models/multihot_pretrain_contextpred'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    torch.save(model_substruct.state_dict(), model_path + '/' + output_file)


if __name__ == "__main__":
    main()
