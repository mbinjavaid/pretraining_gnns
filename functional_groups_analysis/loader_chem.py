"""Read in SMILES strings from a csv file and convert them to RDKit molecular graph objects,
and assign chemical features to the atoms (nodes), as well as properties to the molecules (graphs)."""
import networkx as nx
import os
import numpy as np
import periodictable as pt
import pandas as pd
from rdkit import Chem  # To extract information of the molecules
from rdkit.Chem.rdchem import HybridizationType
import torch
import torch.nn.functional as F
from torch_geometric.data import (InMemoryDataset, Data)
import torch_geometric.utils.coalesce as coalesce

# Atom types:
types = {}
for atom_num, el in enumerate(pt.elements):
    if atom_num != 0:  # Exclude the first element, which is a neutron
        types[el.symbol] = el.number - 1  # so the index starts from hydrogen being 0.


# Output a dict for each molecule, where each key is an atom index and its value is a list of the indices of
# functional groups that atom belongs to.
def atom_functional_group_mapping(mol_, functional_groups):
    mapping = {}
    highlight_list = [mol_.GetSubstructMatches(functional_group) for functional_group in functional_groups]
    for k, highlight in enumerate(highlight_list):
        for atom_idx in [y for x in highlight for y in x]:
            if atom_idx not in mapping:
                mapping[atom_idx] = []
            if k not in mapping[atom_idx]:
                mapping[atom_idx].append(k)
    return mapping


def mol_to_graph_data_obj_simple(molecule):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param molecule: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atom features
    type_idx = []
    aromatic = []
    ring = []
    sp = []
    sp2 = []
    sp3 = []
    sp3d = []
    sp3d2 = []
    num_hs = []
    num_neighbors = []
    for atom in molecule.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        ring.append(1 if atom.IsInRing() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
        sp3d2.append(1 if hybridization == HybridizationType.SP3D2 else 0)
        num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        num_neighbors.append(len(atom.GetNeighbors()))

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([aromatic, ring, sp, sp2, sp3, sp3d, sp3d2], dtype=torch.float).t().contiguous()
    x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=6)
    x4 = F.one_hot(torch.tensor(num_hs), num_classes=5)
    x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float), x4.to(torch.float)], dim=-1)

    # Get edge indices:
    row, col = [], []
    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_index = coalesce(edge_index)

    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = Data(x=x, edge_index=edge_index)

    return data


class QM9(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9_preprocessed.csv'

    @property
    def processed_file_names(self):
        return 'qm9_chem.pt'

    def download(self):
        pass

    def process(self):
        # Read in the csv file containing the canonized SMILES strings:
        df = pd.read_csv(self.raw_paths[0])

        # Create new molecule objects based on the canonical SMILES representations
        df['mol'] = df['canon_smiles'].apply(lambda f: Chem.MolFromSmiles(f))
        # # Add the hydrogen atoms into the molecules:
        df['mol'] = df['mol'].apply(lambda f: Chem.AddHs(f))

        # Check mol.GetNumAtoms() for each element in df['mol'] and if it is <=1, then remove that row from df:
        df = df[df['mol'].apply(lambda f: f.GetNumAtoms()) > 1]

        # # Next step is to do substructure matching: for a given atom in
        # # a molecule, create a one-hot vector with 1 in an entry if the
        # # atom is part of a given functional group, and 0 otherwise.

        # # Array of all SMARTS strings of 130 functional groups:
        # with open(self.raw_paths[1]) as file:
        #     smarts_queries = [line.rstrip() for line in file]
        #
        # # Use MolFromSmarts to get mol objects from the SMARTS strings:
        # smarts_queries = [Chem.MolFromSmarts(smarts) for smarts in smarts_queries]
        # # initialize rings
        # [Chem.GetSSSR(smart) for smart in smarts_queries]

        data_list = []
        # max_len = 0
        # Atom features, bond indices (and potentially bond features)
        for mol_id, molecule in enumerate(df['mol']):
            # atom features
            type_idx = []
            aromatic = []
            ring = []
            sp = []
            sp2 = []
            sp3 = []
            sp3d = []
            sp3d2 = []
            num_hs = []
            num_neighbors = []
            for atom in molecule.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                ring.append(1 if atom.IsInRing() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
                sp3d2.append(1 if hybridization == HybridizationType.SP3D2 else 0)
                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
                num_neighbors.append(len(atom.GetNeighbors()))

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([aromatic, ring, sp, sp2, sp3, sp3d, sp3d2], dtype=torch.float).t().contiguous()
            x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=6)
            x4 = F.one_hot(torch.tensor(num_hs), num_classes=5)
            x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float), x4.to(torch.float)], dim=-1)

            # Get edge indices:
            row, col = [], []
            for bond in molecule.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_index = coalesce(edge_index)

            # # transform SMILES into ascii data type and store it in a name tensor
            # # name = str(Chem.MolToSmiles(molecule))
            # name = df['canon_smiles'][mol_id]
            # ascii_name = []
            # for c in name:
            #     ascii_name.append(int(ord(c)))
            #
            # # if len(ascii_name) > max_len:
            # #     max_len = len(ascii_name)
            #
            # # if fails, increase range
            # for i in range(len(ascii_name), 300):
            #     ascii_name.append(0)
            #
            # ascii_name = torch.tensor([ascii_name], dtype=torch.float).contiguous()

            # # read target properties data
            # # targets initialize with placeholder (infinity is excluded from loss calculation for later training)
            # tmp_mu, tmp_alfa, tmp_homo, tmp_lumo, \
            #     tmp_gap, tmp_r, tmp_zpve, tmp_u0, tmp_u, tmp_h, tmp_g, tmp_cv = \
            #     float("Inf"), float("Inf"), float("Inf"), \
            #     float("Inf"), float("Inf"), float("Inf"), \
            #     float("Inf"), float("Inf"), float("Inf"), \
            #     float("Inf"), float("Inf"), float("Inf")

            property_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                              'zpve', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']

            # In the current row of df (seen by mol_id), extract the values of the properties named
            # in property_names:
            targets = df.iloc[[mol_id]][property_names].to_numpy()[0]
            # add target data, if present in raw data file
            if targets[0] != '':
                tmp_mu = float(targets[0])
            if targets[1] != '':
                tmp_alfa = float(targets[1])
            if targets[2] != '':
                tmp_homo = float(targets[2])
            if targets[3] != '':
                tmp_lumo = float(targets[3])
            if targets[4] != '':
                tmp_gap = float(targets[4])
            if targets[5] != '':
                tmp_r = float(targets[5])
            if targets[6] != '':
                tmp_zpve = float(targets[6])
            if targets[7] != '':
                tmp_cv = float(targets[7])
            if targets[8] != '':
                tmp_u0 = float(targets[8])
            if targets[9] != '':
                tmp_u = float(targets[9])
            if targets[10] != '':
                tmp_h = float(targets[10])
            if targets[11] != '':
                tmp_g = float(targets[11])

            target = torch.tensor([tmp_mu, tmp_alfa, tmp_homo, tmp_lumo, tmp_gap,
                                   tmp_r, tmp_zpve, tmp_cv, tmp_u0, tmp_u, tmp_h, tmp_g], dtype=torch.float)
            target = target.unsqueeze(0)

            data = Data(x=x, edge_index=edge_index,
                        y=target)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

            # Just to check progress after every 1000 molecules:
            if mol_id % 1000 == 0:
                print(mol_id, flush=True)

        torch.save(self.collate(data_list), self.processed_paths[0])


class DownstreamDataset(InMemoryDataset):
    def __init__(self, root, dataset_name=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        super(DownstreamDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.dataset_name is None:
            raise ValueError('dataset_name must be specified')
        return self.dataset_name + '_preprocessed.csv'

    @property
    def processed_file_names(self):
        if self.dataset_name is None:
            raise ValueError('dataset_name must be specified')
        return self.dataset_name + '_chem.pt'

    def download(self):
        pass

    def process(self):
        # Read in the csv file containing the canonized SMILES strings:
        df = pd.read_csv(self.raw_paths[0])

        # The name of the property to be predicted (the column name in the csv file):
        property_names = [self.dataset_name]

        # Create new molecule objects based on the canonical SMILES representations
        df['mol'] = df['canon_smiles'].apply(lambda f: Chem.MolFromSmiles(f))
        # # Add the hydrogen atoms into the molecules:
        df['mol'] = df['mol'].apply(lambda f: Chem.AddHs(f))

        # Check mol.GetNumAtoms() for each element in df['mol'] and if it is <=1, then remove that row from df:
        df = df[df['mol'].apply(lambda f: f.GetNumAtoms()) > 1]

        # # Next step is to do substructure matching: for a given atom in
        # # a molecule, create a one-hot vector with 1 in an entry if the
        # # atom is part of a given functional group, and 0 otherwise.

        # # Array of all SMARTS strings of 130 functional groups:
        # with open(self.raw_paths[1]) as file:
        #     smarts_queries = [line.rstrip() for line in file]
        #
        # # Use MolFromSmarts to get mol objects from the SMARTS strings:
        # smarts_queries = [Chem.MolFromSmarts(smarts) for smarts in smarts_queries]
        # # initialize rings
        # [Chem.GetSSSR(smart) for smart in smarts_queries]

        data_list = []
        # max_len = 0
        # Atom features, bond indices (and potentially bond features)
        for mol_id, molecule in enumerate(df['mol']):
            # atom features
            type_idx = []
            aromatic = []
            ring = []
            sp = []
            sp2 = []
            sp3 = []
            sp3d = []
            sp3d2 = []
            num_hs = []
            num_neighbors = []
            for atom in molecule.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                ring.append(1 if atom.IsInRing() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
                sp3d2.append(1 if hybridization == HybridizationType.SP3D2 else 0)
                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
                num_neighbors.append(len(atom.GetNeighbors()))

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([aromatic, ring, sp, sp2, sp3, sp3d, sp3d2], dtype=torch.float).t().contiguous()
            x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=6)
            x4 = F.one_hot(torch.tensor(num_hs), num_classes=5)
            x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float), x4.to(torch.float)], dim=-1)

            # Get edge indices:
            row, col = [], []
            for bond in molecule.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_index = coalesce(edge_index)

            # # transform SMILES into ascii data type and store it in a name tensor
            # name = df['canon_smiles'][mol_id]
            # ascii_name = []
            # for c in name:
            #     ascii_name.append(int(ord(c)))
            #
            # # if len(ascii_name) > max_len:
            # #     max_len = len(ascii_name)
            #
            # # if fails, increase range
            # for i in range(len(ascii_name), 300):
            #     ascii_name.append(0)
            #
            # ascii_name = torch.tensor([ascii_name], dtype=torch.float).contiguous()

            # # read target properties data
            # # targets initialize with placeholder (infinity is excluded from loss calculation for later training)
            # tmp_lmv = float("Inf")

            # In the current row of df (seen by mol_id), extract the values of the properties named
            # in property_names:
            targets = df.iloc[[mol_id]][property_names].to_numpy()[0]
            # add target data, if present in raw data file
            if targets[0] != '':
                tmp_lmv = float(targets[0])

            target = torch.tensor([tmp_lmv], dtype=torch.float)
            target = target.unsqueeze(0)

            data = Data(x=x, edge_index=edge_index,
                        y=target)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

            # # Just to check progress after every 1000 molecules:
            # if mol_id % 100 == 0:
            #     print("Molecules done: ", mol_id, flush=True)

        torch.save(self.collate(data_list), self.processed_paths[0])


def graph_data_obj_to_nx_simple(data):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx = atom_features[i][0:118]
        fg_membership = atom_features[i][118:]
        G.add_node(i, atomic_num_idx=atomic_num_idx, fg_membership=fg_membership)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    # edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        # bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            # G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
            #            bond_dir_idx=bond_dir_idx)
            G.add_edge(begin_idx, end_idx)

    return G


def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    # num_atom_features = 2  # atom type, functional group membership
    atom_features_list = []
    for _, node in G.nodes(data=True):
        # atom_feature = np.array([atom_type for atom_type in node['atomic_num_idx']] +
        #                         [fg for fg in node['fg_membership']], dtype=np.int64)
        atom_feature = np.concatenate((node['atomic_num_idx'], node['fg_membership']))
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    # num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        # edge_features_list = []
        for i, j, edge in G.edges(data=True):
            # edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            # edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            # edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        # edge_attr = torch.tensor(np.array(edge_features_list),
        #                          dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc_standard_agent',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    # def get(self, idx):
    #     data = Data()
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         s = list(repeat(slice(None), item.dim()))
    #         s[data.cat_dim(key, item)] = slice(slices[idx],
    #                                            slices[idx + 1])
    #         data[key] = item[s]
    #     return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return self.dataset + '_chem.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        # Array of all SMARTS strings of 130 functional groups:
        # with open(self.raw_paths[1]) as file:

        # for file_name in self.raw_paths:
        #     if 'smarts_queries.txt' in file_name:
        #         with open(file_name) as file:
        #             smarts_queries = [line.rstrip() for line in file]

        # # Use MolFromSmarts to get mol objects from the SMARTS strings:
        # smarts_queries = [Chem.MolFromSmarts(smarts) for smarts in smarts_queries]
        # # initialize rings
        # [Chem.GetSSSR(mol_) for mol_ in smarts_queries]

        data_smiles_list = []
        data_list = []

        if self.dataset == 'zinc_standard_agent':
            # input_path = self.raw_paths[0]
            for file in self.raw_paths:
                if 'zinc_preprocessed' in file:
                    print("input file: ", file)
                    input_path = file
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            smiles_list = list(input_df['smiles'])
            # zinc_id_list = list(input_df['zinc_id'])
            for i in range(len(smiles_list)):
                if i % 1000 == 0:
                    print(str(i) + " molecules from ZINC processed ...", flush=True)
                s = smiles_list[i]
                # each example contains a single species
                # try:
                # rdkit_mol = Chem.AllChem.MolFromSmiles(s)
                rdkit_mol = Chem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # # manually add mol id
                    # id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                    # data.id = torch.tensor(
                    #     [id])  # id here is zinc id value, stripped of
                    # # leading zeros
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
                # except:
                #     print("ooga")
                #     continue

        elif self.dataset == 'chembl_filtered':
            ### get downstream test molecules.
            from splitters import scaffold_split

            ###
            downstream_dir = [
                'dataset/bace',
                'dataset/bbbp',
                'dataset/clintox',
                'dataset/esol',
                'dataset/freesolv',
                'dataset/hiv',
                'dataset/lipophilicity',
                'dataset/muv',
                # 'dataset/pcba/processed/smiles.csv',
                'dataset/sider',
                'dataset/tox21',
                'dataset/toxcast'
            ]

            downstream_inchi_set = set()
            for d_path in downstream_dir:
                print(d_path)
                dataset_name = d_path.split('/')[1]
                downstream_dataset = MoleculeDataset(d_path, dataset=dataset_name)
                downstream_smiles = pd.read_csv(os.path.join(d_path,
                                                             'processed', 'smiles.csv'),
                                                header=None)[0].tolist()

                assert len(downstream_dataset) == len(downstream_smiles)

                _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(downstream_dataset,
                                                                                    downstream_smiles, task_idx=None,
                                                                                    null_value=0,
                                                                                    frac_train=0.8, frac_valid=0.1,
                                                                                    frac_test=0.1,
                                                                                    return_smiles=True)

                ### remove both test and validation molecules
                remove_smiles = test_smiles + valid_smiles

                downstream_inchis = []
                for smiles in remove_smiles:
                    species_list = smiles.split('.')
                    for s in species_list:  # record inchi for all species, not just
                        # largest (by default in create_standardized_mol_id if input has
                        # multiple species)
                        inchi = create_standardized_mol_id(s)
                        downstream_inchis.append(inchi)
                downstream_inchi_set.update(downstream_inchis)

            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))

            print('processing')
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None and inchi not in downstream_inchi_set:
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            data.y = torch.tensor(labels[i, :])
                            # fold information
                            if i in folds[0]:
                                data.fold = torch.tensor([0])
                            elif i in folds[1]:
                                data.fold = torch.tensor([1])
                            else:
                                data.fold = torch.tensor([2])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data.fold = torch.tensor([folds[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'pcba':
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'pcba_pretrain':
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            downstream_inchi = set(pd.read_csv(os.path.join(self.root,
                                                            'downstream_mol_inchi_may_24_2019'),
                                               sep=',', header=None)[0])
            for i in range(len(smiles_list)):
                print(i)
                if '.' not in smiles_list[i]:  # remove examples with
                    # multiples species
                    rdkit_mol = rdkit_mol_objs[i]
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None and inchi not in downstream_inchi:
                            # # convert aromatic bonds to double bonds
                            # Chem.SanitizeMol(rdkit_mol,
                            #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            data.y = torch.tensor(labels[i, :])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        # elif self.dataset == ''

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'ptc_mr':
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', header=None, names=['id', 'label', 'smiles'])
            smiles_list = input_df['smiles']
            labels = input_df['label'].values
            for i in range(len(smiles_list)):
                print(i)
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'mutag':
            smiles_path = os.path.join(self.root, 'raw', 'mutag_188_data.can')
            # smiles_path = 'dataset/mutag/raw/mutag_188_data.can'
            labels_path = os.path.join(self.root, 'raw', 'mutag_188_target.txt')
            # labels_path = 'dataset/mutag/raw/mutag_188_target.txt'
            smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
            labels = pd.read_csv(labels_path, header=None)[0].values
            for i in range(len(smiles_list)):
                print(i)
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
