from torch_geometric.nn import GINConv, GCNConv
import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool, global_mean_pool

num_atom_types = 119
num_fg_types = 131


# The model used by Hu et al. (2020) in 'Strategies for Pre-training Graph Neural Networks'
class GIN(torch.nn.Module):
    def __init__(self, num_tasks, num_features, drop_ratio, num_layers=5, hidden_dim=300, pool='sum',
                 use_embeddings=False, pred=False, batch_norm=True):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        if batch_norm:
            self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0 and not use_embeddings:
                self.layers.append(GINConv(torch.nn.Sequential(torch.nn.Linear(num_features, 2 * hidden_dim),
                                                               torch.nn.ReLU(), torch.nn.Linear(2 * hidden_dim, hidden_dim))))
            else:
                self.layers.append(GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2 * hidden_dim),
                                                               torch.nn.ReLU(), torch.nn.Linear(2 * hidden_dim,
                                                                                                hidden_dim))))
        if batch_norm:
            for layer in range(num_layers):
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        self.graph_prediction_linear = torch.nn.Linear(hidden_dim, num_tasks)

        # if use_embeddings:
        #     self.atom_embedding_layer = torch.nn.Embedding(num_atom_types, hidden_dim)
        #     torch.nn.init.xavier_uniform_(self.atom_embedding_layer.weight.data)
        #
        #     self.fg_embedding_layer = torch.nn.Embedding(num_fg_types, hidden_dim)
        #     torch.nn.init.xavier_uniform_(self.fg_embedding_layer.weight.data)

        if pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'mean':
            self.pool = global_mean_pool
        else:
            raise ValueError('Undefined pooling type')

        self.use_embeddings = use_embeddings
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers
        self.pred = pred
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm

    def forward(self, x, edge_index, batch):

        # if self.use_embeddings:
        #     # Embed atom type:
        #     atom_embedding = self.atom_embedding_layer(x[:, 0])
        #
        #     # Embed functional groups, keeping track of the atoms which each functional group that is
        #     # present is associated with:
        #     nonzero_fg_indices = torch.nonzero(x[:, 1:], as_tuple=True)  # get the indices of the FGs present
        #     fg_embedding = self.fg_embedding_layer(nonzero_fg_indices[1])  # embed FGs for each atom (indices 0 to 130)
        #     num_atoms = x.shape[0]
        #     sum_fg_embeddings = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
        #     sum_fg_embeddings.scatter_add_(0, nonzero_fg_indices[0].unsqueeze(1).expand(-1, self.hidden_dim),
        #                                    fg_embedding)
        #
        #     # Add up the atom and functional group embeddings:
        #     x = atom_embedding + sum_fg_embeddings

        h_list = [x]
        for layer in range(self.num_layers):
            h = self.layers[layer](h_list[layer], edge_index)
            if self.batch_norm:
                h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Take the final layer node representations only:
        node_representation = h_list[-1]

        if self.pred:
            return self.graph_prediction_linear(self.pool(node_representation, batch))
        else:
            return node_representation


# The original GIN model proposed by Xu et al. (2019) in 'How Powerful are Graph Neural Networks?'
# This model uses concatenation of graph embeddings from all layers.
class GINConcat(torch.nn.Module):
    def __init__(self, num_tasks, num_features, drop_ratio, num_layers=5, hidden_dim=300, pool='sum',
                 use_embeddings=False, pred=False, batch_norm=True):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        if batch_norm:
            self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0 and not use_embeddings:
                self.layers.append(GINConv(torch.nn.Sequential(torch.nn.Linear(num_features, 2 * hidden_dim),
                                                 torch.nn.ReLU(), torch.nn.Linear(2 * hidden_dim, hidden_dim))))
            else:
                self.layers.append(GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2 * hidden_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * hidden_dim, hidden_dim))))

        if batch_norm:
            for layer in range(num_layers):
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        if use_embeddings:
            self.graph_prediction_linear = torch.nn.Linear(hidden_dim * (num_layers + 1), num_tasks)
        else:
            self.graph_prediction_linear = torch.nn.Linear(hidden_dim * num_layers, num_tasks)

        # if use_embeddings:
        #     self.atom_embedding_layer = torch.nn.Embedding(num_atom_types, hidden_dim)
        #     torch.nn.init.xavier_uniform_(self.atom_embedding_layer.weight.data)
        #
        #     self.fg_embedding_layer = torch.nn.Embedding(num_fg_types, hidden_dim)
        #     torch.nn.init.xavier_uniform_(self.fg_embedding_layer.weight.data)

        if pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'mean':
            self.pool = global_mean_pool
        else:
            raise ValueError('Undefined pooling type')

        self.use_embeddings = use_embeddings
        self.hidden_dim = hidden_dim
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers
        self.pred = pred
        self.batch_norm = batch_norm

    def forward(self, x, edge_index, batch):
        # if self.use_embeddings:
        #     # Embed atom type:
        #     atom_embedding = self.atom_embedding_layer(x[:, 0])
        #
        #     # Embed functional groups, keeping track of the atoms which each functional group that is
        #     # present is associated with:
        #     nonzero_fg_indices = torch.nonzero(x[:, 1:], as_tuple=True)  # get the indices of the FGs present
        #     fg_embedding = self.fg_embedding_layer(nonzero_fg_indices[1])  # embed FGs for each atom (indices 0 to 130)
        #     num_atoms = x.shape[0]
        #     # atom_counts = torch.bincount(nonzero_fg_indices[0], minlength=num_atoms)
        #     sum_fg_embeddings = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
        #     sum_fg_embeddings.scatter_add_(0, nonzero_fg_indices[0].unsqueeze(1).expand(-1, self.hidden_dim),
        #                                    fg_embedding)
        #     # fg_embedding = sum_embeddings / atom_counts.unsqueeze(1)
        #
        #     # Add up the atom and functional group embeddings:
        #     x = atom_embedding + sum_fg_embeddings

        h_list = [x]
        for layer in range(self.num_layers):
            h = self.layers[layer](h_list[layer], edge_index)
            if self.batch_norm:
                h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        h_list = h_list[1:]

        node_representation = torch.cat(h_list, dim=1)

        if self.pred:
            # print("test1: ", self.pool(node_representation, batch))
            # h_list_pooled = [self.pool(h_, batch) for h_ in h_list]
            # print("test2: ", torch.cat(h_list_pooled, dim=1))
            # # print(torch.allclose(self.pool(node_representation, batch), torch.cat(h_list_pooled, dim=1),
            # #                      rtol=0, atol=1e-04))
            return self.graph_prediction_linear(self.pool(node_representation, batch))
        else:
            return node_representation


class GCN(torch.nn.Module):
    def __init__(self, num_tasks, num_features, drop_ratio, num_layers=4, hidden_dim=32, batch_norm=True, pool='sum',
                 use_embeddings=False, pred=False):
        super().__init__()

        # GNN layers:
        self.layers = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0 and not use_embeddings:
                self.layers.append(GCNConv(num_features, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        if batch_norm:
            self.batch_norms = torch.nn.ModuleList()
            for layer in range(num_layers):
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        if num_tasks == 1:
            self.mlp_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1))
        elif num_tasks > 1:
            self.mlp_head = torch.nn.ModuleList([torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)) for _ in range(num_tasks)])
        else:
            raise ValueError('Undefined number of tasks (must be 1 or greater)')

        # if use_embeddings:
        #     self.atom_embedding_layer = torch.nn.Embedding(num_atom_types, hidden_dim)
        #     torch.nn.init.xavier_uniform_(self.atom_embedding_layer.weight.data)
        #
        #     self.fg_embedding_layer = torch.nn.Embedding(num_fg_types, hidden_dim)
        #     torch.nn.init.xavier_uniform_(self.fg_embedding_layer.weight.data)

        if pool == 'sum':
            self.pool = global_add_pool
        elif pool == 'mean':
            self.pool = global_mean_pool
        else:
            raise ValueError('Undefined pooling type')

        self.use_embeddings = use_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.batch_norm = batch_norm
        self.pred = pred
        self.drop_ratio = drop_ratio

    def forward(self, x, edge_index, batch):
        # if self.use_embeddings:
        #     # Embed atom type:
        #     atom_embedding = self.atom_embedding_layer(x[:, 0])
        #
        #     # Embed functional groups, keeping track of the atoms which each functional group that is
        #     # present is associated with:
        #     nonzero_fg_indices = torch.nonzero(x[:, 1:], as_tuple=True)  # get the indices of the FGs present
        #     fg_embedding = self.fg_embedding_layer(nonzero_fg_indices[1])  # embed FGs for each atom (indices 0 to 130)
        #     num_atoms = x.shape[0]
        #     # atom_counts = torch.bincount(nonzero_fg_indices[0], minlength=num_atoms)
        #     sum_fg_embeddings = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
        #     sum_fg_embeddings.scatter_add_(0, nonzero_fg_indices[0].unsqueeze(1).expand(-1, self.hidden_dim),
        #                                    fg_embedding)
        #     # fg_embedding = sum_embeddings / atom_counts.unsqueeze(1)
        #
        #     # Add up the atom and functional group embeddings:
        #     x = atom_embedding + sum_fg_embeddings

        h_list = [x]
        for layer in range(self.num_layers):
            h = self.layers[layer](h_list[layer].float(), edge_index)
            if self.batch_norm:
                h = self.batch_norms[layer](h)
            if layer != self.num_layers - 1:
                # remove relu for the last layer
                h = F.relu(h)
            h_list.append(h)

        # Take the final layer node representations only:
        node_representation = h_list[-1]

        if self.pred:
            if self.num_tasks == 1:
                output = self.mlp_head(self.pool(node_representation, batch))
            else:
                output = [head(self.pool(node_representation, batch)) for head in self.mlp_head]
                output = torch.cat(output, dim=1)
            return output
        else:
            return node_representation
