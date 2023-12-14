import os
import argparse
import torch
import torch_geometric
from model import GINConcat, GIN, GCN

# Multi class classification (an atom can only have one atom type)
criterion_atom = torch.nn.CrossEntropyLoss()

# Multi label classification (an atom can have multiple functional group labels)
criterion_fg = torch.nn.BCEWithLogitsLoss()

num_tasks = 1


def compute_accuracy(pred, target, which):
    if which == 'atom_type':
        return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target.detach()).cpu().item())/len(pred.detach())
    elif which == 'func_group':
        # check how many 1s are in target, that is the number of functional groups to predict
        num_fg = torch.sum(target.detach(), 1)
        correct = 0
        total_fg = 0
        for i in range(len(num_fg)):
            num = num_fg[i].item()
            topk = torch.topk(pred.detach()[i], k=int(num))[1]
            correct += torch.sum(topk == torch.nonzero(target.detach()[i])).item()
            total_fg += num
        return float(correct)/total_fg
    else:
        raise ValueError('"which" must be either atom_type or func_group')


def train(args, model_list, loader, optimizer_list, device):
    if args.which_mask == 'atom_type':
        model, linear_pred_atoms = model_list
        model.train()
        linear_pred_atoms.train()
    elif args.which_mask == 'func_group':
        model, linear_pred_fg = model_list
        model.train()
        linear_pred_fg.train()
    elif args.which_mask == 'all':
        model, linear_pred_atoms, linear_pred_fg = model_list
        model.train()
        linear_pred_atoms.train()
        linear_pred_fg.train()
    else:
        raise ValueError('which_mask must be either atom_type, func_group, or all')

    loss_accum = 0
    acc_atom_type_accum = 0
    acc_fg_accum = 0
    steps = 0
    for data in loader:
        data = data.to(device)

        # zero grad all optimizers:
        for num in range(len(optimizer_list)):
            optimizer_list[num].zero_grad()

        node_rep = model(data.x, data.edge_index, data.batch)
        losses = []
        # loss for nodes (atom type)
        if args.which_mask == 'atom_type' or args.which_mask == 'all':
            pred_atom_type = linear_pred_atoms(node_rep[data.masked_atom_indices])
            if args.use_embeddings:
                atom_indices = data.mask_node_label[:, 0]
                loss_atom_type = criterion_atom(pred_atom_type, atom_indices)
            else:
                # find the location of the 1 in data.mask_node_label[0:118] as the true atom type label:
                atom_indices = torch.nonzero(data.mask_node_label[:, 0:118])[:, 1]
                loss_atom_type = criterion_atom(pred_atom_type, atom_indices)
            losses.append(loss_atom_type)
            if args.train_accuracy:
                acc_atom_type = compute_accuracy(pred_atom_type, atom_indices, which='atom_type')
                acc_atom_type_accum += acc_atom_type

        # loss for nodes (functional group)
        if args.which_mask == 'func_group' or args.which_mask == 'all':
            pred_func_group = linear_pred_fg(node_rep[data.masked_atom_indices])
            if args.use_embeddings:
                fg_ground_truth = data.mask_node_label[:, 1:].float()
                loss_func_group = criterion_fg(pred_func_group, fg_ground_truth)
            else:
                fg_ground_truth = data.mask_node_label[:, 118:]
                loss_func_group = criterion_fg(pred_func_group, fg_ground_truth.to(torch.float))
            losses.append(loss_func_group)
            if args.train_accuracy:
                acc_fg = compute_accuracy(pred_func_group, fg_ground_truth, which='func_group')
                acc_fg_accum += acc_fg

        total_loss = torch.stack(losses).sum()
        total_loss.backward()

        for num_ in range(len(optimizer_list)):
            optimizer_list[num_].step()

        loss_accum += float(total_loss.cpu().item())
        steps += 1

    if args.which_mask == 'atom_type':
        return [loss_accum/steps, acc_atom_type_accum/steps]
    elif args.which_mask == 'func_group':
        return [loss_accum/steps, acc_fg_accum/steps]
    elif args.which_mask == 'all':
        return [loss_accum/steps, acc_atom_type_accum/steps, acc_fg_accum/steps]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio - dropout only implemented for GNN type gin and ginconcat. (default: 0)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='batch normalization (default: 1)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='dimensionality of hidden layers (default: 300)')
    parser.add_argument('--which_mask', type=str, default='all',
                        help='which masked atom properties to predict: atom_type, func_group, or all - default: all')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='mask ratio (default: 0.15)')
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='name of dataset to use for pretraining (zinc, qm9) (default: qm9)')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help="ginconcat, gin, or gcn - default: gin")
    parser.add_argument('--seed', type=int, default=321, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--use_embeddings', type=int, default=0, help='Use feature embeddings (1) or '
                                                                            'multi-hot encoding (0) - default: 0')
    parser.add_argument('--train_accuracy', type=int, default=0, help='Compute training accuracy (1) or not (0) - default: 0')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_geometric.seed_everything(args.seed)

    if args.dataset == 'qm9' and args.use_embeddings:
        from dataloader_embeddings import DataLoaderMasking
        from util_embeddings import MaskAtom
        from loader_embeddings import QM9
        dataset = QM9('Train/', transform=MaskAtom(mask_rate=args.mask_rate))
    elif args.dataset == 'qm9' and not args.use_embeddings:
        from dataloader_multihot import DataLoaderMasking
        from util_multihot import MaskAtom
        from loader_multihot import QM9
        dataset = QM9('Train/', transform=MaskAtom(mask_rate=args.mask_rate))
    elif args.dataset == 'zinc' and args.use_embeddings:
        from dataloader_embeddings import DataLoaderMasking
        from util_embeddings import MaskAtom
        from loader_embeddings import MoleculeDataset
        dataset = MoleculeDataset('Train/', transform=MaskAtom(mask_rate=args.mask_rate))
    elif args.dataset == 'zinc' and not args.use_embeddings:
        from dataloader_multihot import DataLoaderMasking
        from util_multihot import MaskAtom
        from loader_multihot import MoleculeDataset
        dataset = MoleculeDataset('Train/', transform=MaskAtom(mask_rate=args.mask_rate))

    # Shuffle training data
    dataset = dataset.shuffle()
    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print('Operating on following hardware: ' + str(device), flush=True)
    print('---- Target: Pre-Training of ' + args.gnn_type + ' by masking atom features ---- ', flush=True)
    print('Training is based on ' + str(dataset.num_features) + ' atom features and ' +
          str(dataset.num_edge_features) + ' edge features for a molecule.', flush=True)
    print(args.which_mask + ' masking is used with a mask rate of ' + str(args.mask_rate), flush=True)

    # set up models, one for pre-training and one for masking
    if args.gnn_type == 'ginconcat':
        model = GINConcat(num_tasks=num_tasks, num_features=dataset.num_features,
                          drop_ratio=args.dropout_ratio, pred=False,
                          hidden_dim=args.hidden_dim, use_embeddings=args.use_embeddings, num_layers=args.num_layers,
                          batch_norm=args.batch_norm).to(device)
        # Can only also concatenate the input layer to the output representation if using feature embeddings:
        if args.use_embeddings:
            emb_dim = args.hidden_dim * (args.num_layers + 1)
        else:
            emb_dim = args.hidden_dim * args.num_layers
    elif args.gnn_type == 'gin':
        model = GIN(num_tasks=num_tasks, num_features=dataset.num_features,
                    drop_ratio=args.dropout_ratio, pred=False,
                    hidden_dim=args.hidden_dim, use_embeddings=args.use_embeddings,
                    num_layers=args.num_layers, batch_norm=args.batch_norm).to(device)
        emb_dim = args.hidden_dim
    elif args.gnn_type == 'gcn':
        model = GCN(num_tasks=num_tasks, num_features=dataset.num_features,
                    drop_ratio=args.dropout_ratio, pred=False,
                    hidden_dim=args.hidden_dim, use_embeddings=args.use_embeddings,
                    num_layers=args.num_layers, batch_norm=args.batch_norm).to(device)
        emb_dim = args.hidden_dim
    else:
        raise ValueError('Invalid GNN model specified.')

    if args.which_mask == 'atom_type':
        linear_pred_atoms = torch.nn.Linear(emb_dim, 118).to(device)
        model_list = [model, linear_pred_atoms]
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer_linear_pred_atoms = torch.optim.Adam(linear_pred_atoms.parameters(), lr=args.lr,
                                                       weight_decay=args.decay)
        optimizer_list = [optimizer_model, optimizer_linear_pred_atoms]
    elif args.which_mask == 'func_group':
        if args.use_embeddings:
            linear_pred_fg = torch.nn.Linear(emb_dim, 131).to(device)
        else:
            linear_pred_fg = torch.nn.Linear(emb_dim, 130).to(device)
        model_list = [model, linear_pred_fg]
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer_linear_pred_fg = torch.optim.Adam(linear_pred_fg.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer_list = [optimizer_model, optimizer_linear_pred_fg]
    elif args.which_mask == 'all':
        linear_pred_atoms = torch.nn.Linear(emb_dim, 118).to(device)
        if args.use_embeddings:
            linear_pred_fg = torch.nn.Linear(emb_dim, 131).to(device)
        else:
            linear_pred_fg = torch.nn.Linear(emb_dim, 130).to(device)
        model_list = [model, linear_pred_atoms, linear_pred_fg]
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer_linear_pred_atoms = torch.optim.Adam(linear_pred_atoms.parameters(), lr=args.lr,
                                                       weight_decay=args.decay)
        optimizer_linear_pred_fg = torch.optim.Adam(linear_pred_fg.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_fg]
    else:
        raise ValueError('Invalid mask type specified (must be atom_type, func_group, or all).')

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_metrics = train(args, model_list, loader, optimizer_list, device)
        print(train_metrics, flush=True)

    # Save model (pretrained by masking atom type and/or functional groups)
    # 300_5_0_ginconcat.pt
    output_file = (str(args.hidden_dim) + "_" + str(args.num_layers) + "_" +
                   str(args.batch_norm) + "_" + args.gnn_type + "_masking_" + args.which_mask + "_" +
                   str(args.mask_rate) + "_" + args.dataset + ".pt")

    if not os.path.exists("models"):
        os.makedirs("models")
    if args.use_embeddings:
        model_path = 'models/embeddings_pretrain_masking'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    else:
        model_path = 'models/multihot_pretrain_masking'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    torch.save(model.state_dict(), model_path + '/' + output_file)


if __name__ == "__main__":
    main()
