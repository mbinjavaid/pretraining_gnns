# pretraining_gnns

## Structure of repository

- [ ] **pretrain_singletask:** run this to carry out single-task supervised pre-training with the 12 QM9 targets specified in Section 5.5.1 of the thesis. Upon running this, for each target, a GCN model is trained and saved in the `models_gcn` folder.
- [ ] **pretrain_multitask:** run this to carry out multi-task supervised pre-training with the 12 QM9 targets. Upon running, a single GCN model is trained and saved in the `models_gcn` folder.
- [ ] **finetune_singletask:** run this to fine-tune on the downstream datasets using the trained model(s) saved by **pretrain_singletask** as initialization. Needs the latter to be run first. Will fine-tune and test sequentially on all the specified downstream datasets for each single-task pre-trained model present in `models_gcn/singletask`. The test results along with the model parameters used will be saved in `results/single_multi_task/gcn_singletask.csv` (previously saved results are not overwritten).
- [ ] **finetune_multitask:** run this to fine-tune on the downstream datasets using the trained model(s) saved by **pretrain_multitask** as initialization. Needs the latter to be run first. Will fine-tune and test sequentially on all the specified downstream datasets for each multi-task pre-trained model present in `models_gcn/multitask`. The test results along with the model parameters used will be saved in `results/single_multi_task/gcn_multitask.csv`.
- [ ] **finetune_no_pretrain:** run this to train and test on the downstream datasets without using a pre-trained model as initialization. The test results along with the model parameters used will be saved in `results/single_multi_task/gcn_no_pretrain.csv`.

## Instructions

**pretrain_multitask** must be run before **finetune_multitask**, and likewise **pretrain_singletask** must be run before **finetune_singletask**. However, **finetune_no_pretrain** can be run without any pre-training script being run prior.

The training and GCN parameters (and the downstream datasets to use during fine-tuning) can be specified and changed from the default, please check details with `python filename.py --help` for the file you wish to run. However, for the finetuning scripts (except for finetune_no_pretrain), the GCN model parameters cannot be specified, as the model parameters of the saved pre-trained GCN model (generated by pretrain_singletask or pretrain_multitask) currently being fine-tuned will automatically be adopted; these parameters include the hidden dimension, number of GCN layers, choice of graph pooling function, and the presence or absence of Batch Normalization.

**Reproducing the thesis results**: To reproduce the results of the GCNs in Section 6.1 and 6.2 of the thesis, please keep the default values for the parameters when running a pre-training script or finetune_no_pretrain, only varying `--batch_norm` and `--global_pooling` according to which particular GCN's result should be reproduced (i.e. GCN with/without Batch Norm with sum or mean graph pooling). When running finetune_multitask.py or finetune_singletask.py, no parameters need to be manually specified, as the model parameters of each saved pre-trained model being finetuned will be adopted automatically.