# pretraining_gnns

Link to thesis report: https://drive.google.com/file/d/1b59pnE5rmfEuOURAaISTkfGsc3k0nVjo/view?usp=sharing

## Structure of repository

- [ ] **datasets:** contains the original (non-preprocessed) datasets in the form of XLSX or CSV files with SMILES strings and molecular property labels.
- [ ] **single_multi_task_GCNs:** contains the code for reproducing the results of Section 6.1 in the thesis; i.e single/multi-task supervised pre-training with GCNs, as well as the option of not using pre-training. The results of the GCNs in Section 6.2 of the thesis can also be reproduced using these.
- [ ] **self_supervised:** contains the code for reproducing the results of Section 6.2 of the thesis with the GIN models as well as Section 6.3 of the thesis, using variants of the self-supervised pre-training techniques (Attribute Masking and Context Prediction) for GIN and GCN models. Allows to fine-tune on the downstream datasets with multi-task supervised pre-training alone, a self-supervised pre-training technique alone, a self-supervised pre-training technique followed by multi-task supervised pre-training, or no pre-training.
- [ ] **functional_groups_analysis:** contains the code for reproducing the results of Section 6.5 of the thesis.

## Instructions

Please note that the pre-training and fine-tuning scripts require the presence of the `Train/raw/` directory with the pre-processed dataset CSV files present in it. Pre-processing involves removing duplicates, molecules with faulty SMILES, and, in the case of the pre-training datasets, removing molecules which are also present in the downstream datasets. The same directory also contains smarts_queries.txt, which contains the SMARTS strings of 130 functional groups, which is required to generate graph objects from the pre-processed CSV files.

Therefore, please first download the dataset files from the Releases section, and place them in the `datasets` directory. Then, run **downstream_datasets_preprocessing.py**, **qm9_preprocessing.py**, and optionally, **zinc_preprocessing.py** to generate the pre-processed datasets inside `Train/raw/`. Then, the Train folder can be placed inside the other folders (single_multi_task_GCNs, self_supervised, functional_groups_analysis) and the scripts in them may be run as usual.

Alternatively, after data pre-processing, the scripts from a folder may be moved to the main (parent) directory instead, and then run (however, please keep an eye out for possible filename conflicts between scripts from different folders!)
