from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool, global_mean_pool

num_atom_types = 119
num_fg_types = 131


class GCN(torch.nn.Module):
    def __init__(self, num_tasks, num_features, num_layers=4, hidden_dim=32, batch_norm=True, pool='sum',
                 use_embeddings=False, pred=True):
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

        if use_embeddings:
            self.atom_embedding_layer = torch.nn.Embedding(num_atom_types, hidden_dim)
            torch.nn.init.xavier_uniform_(self.atom_embedding_layer.weight.data)

            self.fg_embedding_layer = torch.nn.Embedding(num_fg_types, hidden_dim)
            torch.nn.init.xavier_uniform_(self.fg_embedding_layer.weight.data)

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

    def forward(self, x, edge_index, batch):
        if self.use_embeddings:
            # Embed atom type:
            atom_embedding = self.atom_embedding_layer(x[:, 0])

            # Embed functional groups, keeping track of the atoms which each functional group that is
            # present is associated with:
            nonzero_fg_indices = torch.nonzero(x[:, 1:], as_tuple=True)  # get the indices of the FGs present
            fg_embedding = self.fg_embedding_layer(nonzero_fg_indices[1])  # embed FGs for each atom (indices 0 to 130)
            num_atoms = x.shape[0]
            sum_fg_embeddings = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
            sum_fg_embeddings.scatter_add_(0, nonzero_fg_indices[0].unsqueeze(1).expand(-1, self.hidden_dim),
                                           fg_embedding)

            # Add up the atom and functional group embeddings:
            x = atom_embedding + sum_fg_embeddings

        h_list = [x]
        for layer in range(self.num_layers):
            h = self.layers[layer](h_list[layer].float(), edge_index)
            if self.batch_norm:
                h = self.batch_norms[layer](h)
            # if layer != self.num_layers - 1:
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
