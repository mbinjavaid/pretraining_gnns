from loader_fge import QM9, DownstreamDataset
# import numpy as np
import pickle
import os

# Total number of functional groups being considered:
total_fgs = 130
# The minimum number of molecules in QM9 (pre-training dataset) that should have a functional group in order
# for molecules containing that FG in a downstream dataset to be considered for placing into a holdout (test) set:
min_present_pretraining = 1000
# Minimum percentage of the downstream dataset that can be used for testing:
min_test_percentage = 0.1
# Maximum percentage of the downstream dataset that can be used for testing:
max_test_percentage = 0.3

if not os.path.exists('functional_groups'):
    os.makedirs('functional_groups')

# list of downstream dataset names:
downstream_datasets = ['AiT', 'Hf', 'LD50', 'Lmv', 'LogP', 'OSHA_TWA', 'Tb', 'Tm']
dataset = QM9('Train/')

qm9_fgs = {i: 0 for i in range(total_fgs)}
# Check for no. of molecules in QM9 having each of the 130 functional groups:
for i in range(len(dataset)):
    molecule = dataset.get(i)
    done_fgs = []
    # Check the column index of the 1s in every row of molecule.x, those are the fgs present in the molecule:
    for row in molecule.x:
        for col in row[118:].nonzero():
            if col.item() not in done_fgs:
                qm9_fgs[col.item()] += 1
                done_fgs.append(col.item())

downstream_fgs = [{i: 0 for i in range(total_fgs)} for j in range(len(downstream_datasets))]
# Check for no. of molecules in each of the downstream datasets having each of the 130 functional groups:
for idx in range(len(downstream_datasets)):
    downstream_dataset = DownstreamDataset('Train/', downstream_datasets[idx].lower())
    for i in range(len(downstream_dataset)):
        molecule = downstream_dataset.get(i)
        done_fgs = []
        # Check the column index of the 1s in every row of molecule.x, those are the fgs present in the molecule:
        for row in molecule.x:
            for col in row[118:].nonzero():
                if col.item() not in done_fgs:
                    downstream_fgs[idx][col.item()] += 1
                    done_fgs.append(col.item())

# Next, we want to see what happens to the distribution of the number of molecules FGs if, for each downstream
# dataset, we move, for each FG, the molecules that have that FG to a 'holdout' (test) set. Then, we want to reevaluate
# the distribution of the number of molecules FGs in the remaining molecules. We want to do this for each FG, and for
# each downstream dataset.
for idx in range(len(downstream_datasets)):
    downstream_dataset = DownstreamDataset('Train/', downstream_datasets[idx].lower())
    # print('For ' + datasets[idx] + ':')
    downstream_molecules = []
    for i in range(len(downstream_dataset)):
        molecule = downstream_dataset.get(i)
        molecule_fgs = []
        # Check the column index of the 1s in every row of molecule.x, those are the fgs present in the molecule:
        for row in molecule.x:
            for col in row[118:].nonzero():
                if col.item() not in molecule_fgs:
                    downstream_fgs[idx][col.item()] += 1
                    molecule_fgs.append(col.item())
        downstream_molecules.append(molecule_fgs)

    # downstream_molecules contains the list of functional groups present in each molecule of the
    # current downstream dataset.
    # Now, for each FG, move the molecules that have that FG to a holdout set:
    fg_list = []
    holdout_molecules_done = []
    for fg in range(total_fgs):
        holdout_molecules = []
        for i in range(len(downstream_dataset)):
            # Only pick those FGs that are present in at least 1000 molecules in QM9:
            if fg in downstream_molecules[i] and qm9_fgs[fg] >= min_present_pretraining:
                holdout_molecules.append(i)
        # If the exact same set of molecules has already been moved to the holdout set for a different FG
        # (can occur if two functional groups almost always occur together), then don't move them again,
        # instead just put a placeholder inside holdout_molecules:
        if holdout_molecules in holdout_molecules_done:
            holdout_molecules = [downstream_datasets[idx] + '_' + str(fg)]
        holdout_molecules_done.append(holdout_molecules)
        if min_test_percentage*len(downstream_dataset) <= len(holdout_molecules) <= max_test_percentage*len(downstream_dataset):
            fg_list.append((fg, len(holdout_molecules), holdout_molecules))

    # We need only the top 10 functional groups with the most molecules moved to the holdout set:
    if len(fg_list) >= 10:
        fg_list.sort(key=lambda x: x[1])
        fg_list = fg_list[:10]
        print('For dataset ' + downstream_datasets[idx] + ':')
        test_indices = []
        fg_idx = []
        for fg_id, fg in enumerate(fg_list):
            print('For functional group ' + str(fg[0]) + ', ' + str(fg[1]) +
                  ' molecules are moved to the test set.')
            print('The remaining ' + str(len(downstream_dataset) - fg[1]) + ' molecules are used for training.')
            print('The indices of the molecules moved to the test set are: ' + str(fg[2]))
            print(downstream_datasets[idx].lower() + '_' + str(fg_id) + '_test_indices = ' + str(fg[2]))
            print(downstream_datasets[idx].lower() + '_' + str(fg_id) + '_fg = ' + str(fg[0]))
            fg_idx.append(fg[0])
            test_indices.append(fg[2])
            # np.save('functional_groups/' + downstream_datasets[idx].lower() + '_' + str(fg_id) + '_test_indices.npy', fg[2])
            # np.save('functional_groups/' + downstream_datasets[idx].lower() + '_' + str(fg_id) + '_fg.npy', fg[0])
        # np.save('functional_groups/' + downstream_datasets[idx].lower() + '_test_fgs.npy', fg_idx)
        # Write test_indices to a pickle file in functional_groups folder with the name of the dataset + '_test_indices'
        # and fg_idx to a pickle file in functional_groups folder with the name of the dataset + '_test_fgs'
        with open('functional_groups/' + downstream_datasets[idx].lower() + '_test_indices.pkl', 'wb') as f:
            pickle.dump(test_indices, f)
        with open('functional_groups/' + downstream_datasets[idx].lower() + '_test_fgs.pkl', 'wb') as f:
            pickle.dump(fg_idx, f)
        print(downstream_datasets[idx] + ': ' + str(fg_idx))
    else:
        print('For dataset ' + downstream_datasets[idx] + ':')
        print(len(fg_list))
        raise ValueError('Not enough functional groups with enough molecules to move to the holdout set.')
