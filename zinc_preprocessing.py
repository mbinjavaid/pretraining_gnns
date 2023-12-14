import pandas as pd
import os
from rdkit import Chem  # To extract information of the molecules


if not os.path.exists('Train'):
    os.makedirs('Train')
if not os.path.exists('Train/raw'):
    os.makedirs('Train/raw')

df = pd.read_csv('datasets/zinc_combined_apr_8_2019.csv.gz', compression='gzip')

# Apply the function Chem.CanonSmiles(smiles) on each element of the column 'smiles' and overwrite the column with its
# result:
df['smiles'] = df['smiles'].apply(lambda a: Chem.CanonSmiles(a))

pre_len = len(df)
# Remove rows with empty or N/A 'smiles' values:
df = df.dropna(subset=['smiles'])
df = df[df['smiles'] != '']
post_len = len(df)

print("Number of faulty SMILES representations dropped: ", pre_len - post_len)

# Remove molecule duplicates having the same canonical SMILES representations
pre_len = len(df)
df = df.drop_duplicates(subset=['smiles'], keep='first')
post_len = len(df)

print("Number of duplicate molecules dropped: ", pre_len - post_len)

# Remove molecules in ZINC that are also contained in downstream datasets
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

pre_len = len(df)
for dataset in datasets:
    dataset.columns = dataset.columns.str.lower()
    dataset['mol'] = dataset['smiles'].apply(lambda b: Chem.MolFromSmiles(b))
    # Remove rows with Null molecule representation in the Lmv dataset:
    dataset = dataset.dropna(subset=['mol'])
    # Remove the 'mol' column:
    dataset = dataset.drop(columns=['mol'])
    dataset['canon_smiles'] = dataset['smiles'].apply(lambda c: Chem.CanonSmiles(c))
    # Remove molecule duplicates having the same canonical SMILES representations
    dataset = dataset.drop_duplicates(subset=['canon_smiles'], keep='first')
    # Remove molecules in ZINC that are also contained in 'dataset':
    df = df[~df.smiles.isin(dataset.canon_smiles)]
post_len = len(df)

print("Number of molecules in ZINC that are also contained in downstream datasets dropped: ", pre_len - post_len)
# Save the dataframe as a csv file:
df.to_csv('Train/raw/zinc_preprocessed.csv.gz', index=False, compression='gzip')

print("Final size of ZINC: ", len(df))
print("Finished pre-processing ZINC!")
