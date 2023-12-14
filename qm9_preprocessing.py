import os
import re
import pandas as pd
from rdkit import Chem  # To extract information of the molecules


# Function to canonize each molecule's SMILES representation:
def canonize(smiles_):
    mol_ = Chem.MolFromSmiles(smiles_)
    return Chem.MolToSmiles(mol_, isomericSmiles=True, canonical=True)


if not os.path.exists('Train'):
    os.makedirs('Train')
if not os.path.exists('Train/raw'):
    os.makedirs('Train/raw')

df = pd.read_csv('datasets/qm9.csv')
df['index'] = range(1, len(df) + 1)

# Remove the list of uncharacterized molecules:
# Extract the index values of these uncharacterized molecules from the provided text file:
with open('datasets/uncharacterized.txt', 'r') as file:
    # Skip the header lines
    for _ in range(3):
        next(file)
    # Extract the integers from the "Index" column
    indices = []
    for line in file:
        match = re.search(r'\s+(\d+)\s+', line)
        if match:
            index = int(match.group(1))
            indices.append(index)
# print(indices)

print("Original Size: ", len(df))

# Remove the uncharacterized molecules from the dataframe:
df = df[~df['index'].isin(indices)]

df = df.drop(columns=['index'])

print("Size after dropping uncharacterized molecules: ", len(df))

# Remove rows with Null SMILES strings, if any:
df = df.dropna(subset=['smiles'])

# Creating molecule graphs via SMILES strings
df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

# Checking if there are None values
print("Number of N/A molecules: ", df['mol'].isnull().sum())
df = df.dropna(subset=['mol'])

# Drop the column 'mol' from df since we no longer have any use for it:
df = df.drop(columns=['mol'])

# Instead, apply the canonize function to the 'smiles' column directly:
df['canon_smiles'] = df['smiles'].apply(lambda x: canonize(x))

# Drop the column 'smiles' from df since we no longer have any use for it:
df = df.drop(columns=['smiles'])

pre_dup_drop_len = len(df)

# Remove molecule duplicates having the same canonical SMILES representations
df = df.drop_duplicates(subset=['canon_smiles'], keep='first')

post_dup_drop_len = len(df)

print('Number of duplicates dropped: ', pre_dup_drop_len - post_dup_drop_len)

# Remove molecules in QM9 that are also contained in downstream datasets
# (as these will later be used to fine-tune the model on):
ait = pd.read_excel('datasets/AiT.xlsx')
hf = pd.read_excel('datasets/Hf.xlsx')
ld50 = pd.read_excel('datasets/LD50.xlsx')
lmv = pd.read_excel('datasets/Lmv.xlsx')
logp = pd.read_excel('datasets/LogP.xlsx')
osha = pd.read_excel('datasets/OSHA-TWA.xlsx')
tb = pd.read_excel('datasets/Tb.xlsx')
tm = pd.read_excel('datasets/Tm.xlsx')

datasets = [ait, hf, ld50, lmv, logp, osha, tb, tm]

pre_dup_drop_len = len(df)
for dataset in datasets:
    dataset.columns = dataset.columns.str.lower()
    dataset['mol'] = dataset['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    # Remove rows with Null molecule representation in the Lmv dataset:
    dataset = dataset.dropna(subset=['mol'])
    # Remove the 'mol' column:
    dataset = dataset.drop(columns=['mol'])
    dataset['canon_smiles'] = dataset['smiles'].apply(lambda x: canonize(x))
    # Remove molecule duplicates having the same canonical SMILES representations
    dataset = dataset.drop_duplicates(subset=['canon_smiles'], keep='first')
    # Remove molecules in QM9 that are also contained in 'dataset':
    df = df[~df.canon_smiles.isin(dataset.canon_smiles)]

post_dup_drop_len = len(df)
# Save the dataframe as a csv file:
df.to_csv('Train/raw/qm9_preprocessed.csv', index=False)

print("Number of occurrences from downstream datasets dropped: ", pre_dup_drop_len - post_dup_drop_len)
print("Final length of QM9: ", len(df))
print("Done pre-processing QM9!")
