import os
import pandas as pd
from rdkit import Chem


# # Canonize each molecule's SMILES representation:
def canonize(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)


if not os.path.exists('Train'):
    os.makedirs('Train')
if not os.path.exists('Train/raw'):
    os.makedirs('Train/raw')


ait = pd.read_excel('datasets/AiT.xlsx')
hf = pd.read_excel('datasets/Hf.xlsx')
ld50 = pd.read_excel('datasets/LD50.xlsx')
lmv = pd.read_excel('datasets/Lmv.xlsx')
logp = pd.read_excel('datasets/LogP.xlsx')
osha = pd.read_excel('datasets/OSHA-TWA.xlsx')
tb = pd.read_excel('datasets/Tb.xlsx')
tm = pd.read_excel('datasets/Tm.xlsx')

datasets = [ait, hf, ld50, lmv, logp, osha, tb, tm]
names = ['AiT', 'Hf', 'LD50', 'Lmv', 'LogP', 'OSHA_TWA', 'Tb', 'Tm']
# preprocessed_datasets = []

for idx, dataset in enumerate(datasets):
    dataset.columns = dataset.columns.str.lower()
    pre_drop = len(dataset)
    dataset['mol'] = dataset['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    # Remove rows with Null molecule representation in the Lmv dataset:
    dataset = dataset.dropna(subset=['mol'])
    post_drop = len(dataset)
    print("Number of N/A molecules dropped in ", names[idx], ": ", pre_drop - post_drop)
    # Remove the 'mol' column:
    dataset = dataset.drop(columns=['mol'])
    dataset['canon_smiles'] = dataset['smiles'].apply(lambda x: canonize(x))
    # Remove molecule duplicates having the same canonical SMILES representations
    pre_drop = len(dataset)

    property_name = str(dataset.columns[1])
    dataset = dataset.rename(columns={property_name: names[idx].lower()})
    property_name = str(dataset.columns[1])

    dataset = dataset.drop_duplicates(subset=['canon_smiles'], keep='first')
    post_drop = len(dataset)
    print("Number of duplicate molecules dropped in ", names[idx], ": ", pre_drop - post_drop)

    # For the Tb dataset, remove an outlier:
    if property_name.lower() == 'tb':
        dataset = dataset[dataset['tb'] <= 2535]

    # preprocessed_datasets.append(dataset)
    print('Final number of molecules in ', names[idx], ': ', len(dataset))

    dataset.to_csv('Train/raw/' + names[idx].lower() + '_preprocessed.csv', index=False)

print("Done preprocessing all downstream datasets!")
