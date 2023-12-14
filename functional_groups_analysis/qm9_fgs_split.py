from loader_fge import QM9
import random
import pickle
import copy
import os
import sys

random.seed(321)
# Total number of functional groups being considered:
total_fgs = 130
# Minimum number of molecules with a certain functional group which should be present in each "10k" pre-training set:
min_present_pretraining = 1000

dataset_names = ['AiT', 'Hf', 'LD50', 'Lmv', 'LogP', 'OSHA_TWA', 'Tb', 'Tm']

if not os.path.exists('functional_groups'):
    os.makedirs('functional_groups')

# Load the list of test indices for each dataset from the functional_groups folder:
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

    # Load the list of functional groups for each dataset from the functional_groups folder:
    with open('functional_groups/ait_test_fgs.pkl', 'rb') as f:
        ait_test_fgs = pickle.load(f)
    with open('functional_groups/hf_test_fgs.pkl', 'rb') as f:
        hf_test_fgs = pickle.load(f)
    with open('functional_groups/ld50_test_fgs.pkl', 'rb') as f:
        ld50_test_fgs = pickle.load(f)
    with open('functional_groups/lmv_test_fgs.pkl', 'rb') as f:
        lmv_test_fgs = pickle.load(f)
    with open('functional_groups/logp_test_fgs.pkl', 'rb') as f:
        logp_test_fgs = pickle.load(f)
    with open('functional_groups/osha_twa_test_fgs.pkl', 'rb') as f:
        osha_twa_test_fgs = pickle.load(f)
    with open('functional_groups/tb_test_fgs.pkl', 'rb') as f:
        tb_test_fgs = pickle.load(f)
    with open('functional_groups/tm_test_fgs.pkl', 'rb') as f:
        tm_test_fgs = pickle.load(f)

except FileNotFoundError:
    print('Please run functional_group_split.py first to generate the test molecule indices and split-defining functional '
          'group indices for each dataset.')
    sys.exit()

ait_test_indices = [len(x) for x in ait_test_indices]
hf_test_indices = [len(x) for x in hf_test_indices]
ld50_test_indices = [len(x) for x in ld50_test_indices]
lmv_test_indices = [len(x) for x in lmv_test_indices]
logp_test_indices = [len(x) for x in logp_test_indices]
osha_twa_test_indices = [len(x) for x in osha_twa_test_indices]
tb_test_indices = [len(x) for x in tb_test_indices]
tm_test_indices = [len(x) for x in tm_test_indices]

dataset_test_indices = [ait_test_indices, hf_test_indices, ld50_test_indices, lmv_test_indices, logp_test_indices,
                        osha_twa_test_indices, tb_test_indices, tm_test_indices]

dataset_test_fgs = [ait_test_fgs, hf_test_fgs, ld50_test_fgs, lmv_test_fgs, logp_test_fgs, osha_twa_test_fgs,
                    tb_test_fgs, tm_test_fgs]

qm9 = QM9('Train/')

# A dictionary to contain the molecule indices in QM9 which contain each functional group:
qm9_fgs = {i: [] for i in range(total_fgs)}
# Populate the dictionary:
for i in range(len(qm9)):
    molecule = qm9.get(i)
    done_fgs = []
    # Check the column index of the 1s in every row of molecule.x, those are the fgs present in the molecule:
    for row in molecule.x:
        for col in row[118:].nonzero():
            if col.item() not in done_fgs:
                qm9_fgs[col.item()].append(i)
                done_fgs.append(col.item())


qm9_fg_samples_list = []
for idx, dataset in enumerate(dataset_names):
    # Create a deep copy of qm9 dictionary qm9_fgs, such that the original remains intact:
    qm9_fgs_ = copy.deepcopy(qm9_fgs)
    qm9_fg_lengths = []
    for i in range(len(dataset_test_indices[idx])):
        fg_length = dataset_test_indices[idx][i]
        fg = dataset_test_fgs[idx][i]
        # Check how many molecules in QM9 contain this FG:
        qm9_fg_length = len(qm9_fgs_[fg])
        qm9_fg_lengths.append(qm9_fg_length)

    # Check if there are at least min_present_pretraining molecules in QM9 which contain each of the fgs listed in
    # dataset_test_fgs[idx]:
    for i in range(len(qm9_fg_lengths)):
        if qm9_fg_lengths[i] < min_present_pretraining:
            raise ValueError('Failed to find ' + str(min_present_pretraining) + ' molecules in QM9 which contain the '
                             'functional group ' + str(i) + '  for dataset ' + dataset + '.')

    print('Success in finding ' + str(min_present_pretraining) + ' molecules in QM9 for each of the '
                                                                                             'functional groups defining '
                                                                                             'the test sets of dataset '
                                                                                             '' + dataset + '.')
    # Next, sample 1000 QM9 molecules containing each of the fgs listed in dataset_test_fgs[idx]
    # also, make sure that new sampled molecules are not already in qm9_fg_samples.
    # at the end, qm9_fg_samples is a list of 10000 molecules from QM9 which contains
    # all the fgs listed in dataset_test_fgs[idx]
    qm9_fg_samples = []
    for i in range(len(dataset_test_indices[idx])):
        break_indicator = False
        fg_length = dataset_test_indices[idx][i]
        fg = dataset_test_fgs[idx][i]
        # Sample a bit more than 1000 molecules, so that we can remove duplicates later
        # (e.g. molecules extracted for other functional groups that happen to be the same) and
        # still have at least 1000 molecules per FG:
        qm9_fg_sample = random.sample(qm9_fgs_[fg], 1500)
        # qm9_fg_sample = qm9_fgs_[fg]

        if i >= 1:
            # the goal of the following code is to make sure that the molecules in qm9_fg_sample are not already in
            # qm9_fg_samples. If they are, then replace them with molecules which are not in qm9_fg_samples.
            # this is so that we can have 1000 molecules from each of the fgs listed in dataset_test_fgs[idx], and
            # we don't end up with a lot of duplicates that, when removed, lead to less than 10000 total molecules and
            # less than 1000 molecules for some of the fgs.
            already_tried = []
            while any(x in qm9_fg_samples for x in qm9_fg_sample):
                if break_indicator:
                    break_indicator = False
                    break
                # Sample one by one from qm9_fgs_[fg] until a molecule is sampled which is not in qm9_fg_samples:
                while True:
                    # Check if the sorted version of already_tried is exactly the same list as qm9_fgs_[fg]:
                    if set(already_tried) == set(qm9_fgs_[fg]):
                        break_indicator = True
                        break
                    qm9_fg_sample_replacement = random.sample(qm9_fgs_[fg], 1)
                    if qm9_fg_sample_replacement[0] not in already_tried:
                        already_tried.append(qm9_fg_sample_replacement[0])
                    if qm9_fg_sample_replacement[0] not in qm9_fg_samples and qm9_fg_sample_replacement[0] not in qm9_fg_sample:
                        qm9_fg_sample_bool = [x in qm9_fg_samples for x in qm9_fg_sample]
                        # At the index where the first True appears in qm9_fg_sample_bool, replace the corresponding
                        # element (at the same index) in qm9_fg_sample with the sampled molecule:
                        qm9_fg_sample[qm9_fg_sample_bool.index(True)] = qm9_fg_sample_replacement[0]
                        break
        qm9_fg_samples.extend(qm9_fg_sample)

    # remove duplicates (if any):
    qm9_fg_samples = list(set(qm9_fg_samples))
    # print(len(qm9_fg_samples))
    qm9_fg_samples_list.append(qm9_fg_samples)


# For the molecule indices in qm9_fg_samples, confirm that there are at least 1000 of each of the fgs listed in
# dataset_test_fgs[idx]:
for idx in range(len(dataset_names)):
    qm9_fg_lengths = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    while any(qm9_fg_lengths[j] < min_present_pretraining for j in range(len(qm9_fg_lengths))):
        # qm9_fg_samples_ = random.sample(qm9_fg_samples_list[idx], min(len(x) for x in qm9_fg_samples_list))
        qm9_fg_samples_ = random.sample(qm9_fg_samples_list[idx], 10000)
        for i in range(len(dataset_test_indices[idx])):
            fg = dataset_test_fgs[idx][i]
            qm9_fg_length = len([y for y in qm9_fg_samples_ if y in qm9_fgs[fg]])
            qm9_fg_lengths[i] = qm9_fg_length

    print(len(qm9_fg_samples_))
    print('qm9_' + dataset_names[idx].lower() + '_10k_indices = ' + str(qm9_fg_samples_))

    # Save these indices of QM9 molecules for the currently considered downstream dataset to a file:
    # these are the "10k pretraining" set of molecules in the thesis
    with open('functional_groups/qm9_' + dataset_names[idx].lower() + '_10k_indices.pkl', 'wb') as f:
        pickle.dump(qm9_fg_samples_, f)

    # Next, find molecules in QM9 which DO NOT contain the functional groups in the current dataset, i.e. in
    # ait_test_fgs, hf_test_fgs, etc.
    # these are the "10k excluded" set of molecules in the thesis.
    qm9_not_present_indices = [[] for i in range(len(qm9))]
    # Populate the dictionary:
    for i in range(len(qm9)):
        continue_marker = False
        molecule = qm9.get(i)
        done_fgs = []
        # Check the column index of the 1s in every row of molecule.x, those are the fgs present in the molecule:
        for row in molecule.x:
            if continue_marker:
                break
            for col in row[118:].nonzero():
                if col.item() in dataset_test_fgs[idx]:
                    qm9_not_present_indices[i] = []
                    continue_marker = True
                    break
                if col.item() not in done_fgs:
                    qm9_not_present_indices[i].append(i)
                    done_fgs.append(col.item())

    # flatten and remove duplicates:
    qm9_not_present_indices = list(set([item for sublist in qm9_not_present_indices for item in sublist]))

    # If more than 10000 "excluded" molecules found, i.e. molecules with FGs not present
    # in the test set of the downstream dataset, then sample 10000 of them:
    if len(qm9_not_present_indices) >= 10000:
        # qm9_not_present_indices = random.sample(qm9_not_present_indices, min(len(x) for x in qm9_fg_samples_list))
        qm9_not_present_indices = random.sample(qm9_not_present_indices, 10000)
        print(len(qm9_not_present_indices))
        print('qm9_' + dataset_names[idx].lower() + '_10k_excluded_indices = ' + str(qm9_not_present_indices))
        # Save these to a file:
        with open('functional_groups/qm9_' + dataset_names[idx].lower() + '_10k_excluded_indices.pkl', 'wb') as f:
            pickle.dump(qm9_not_present_indices, f)
    # the above condition is not fulfilled for AiT, Hf, and Tb, so for them, sample the minimum number of
    # molecules in the "excluded" set of any of these three datasets, which we found to be 2763.
    else:
        # print(len(qm9_not_present_indices))
        qm9_not_present_indices = random.sample(qm9_not_present_indices, 2763) # ait, hf, and tb
        print(len(qm9_not_present_indices))
        print('qm9_' + dataset_names[idx].lower() + '_10k_excluded_indices = ' + str(qm9_not_present_indices))
        # Save these to a file:
        with open('functional_groups/qm9_' + dataset_names[idx].lower() + '_10k_excluded_indices.pkl', 'wb') as f:
            pickle.dump(qm9_not_present_indices, f)

    # print(any(x in qm9_fg_samples_ for x in qm9_not_present_indices))
