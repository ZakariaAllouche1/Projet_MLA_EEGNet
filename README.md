---

# Project Objectives

- Reproduce the EEGNet-8,2 architecture proposed by Lawhern et al. (2018)
- Follow the within-subject and cross-subject protocols described in the paper
- Validate the reproduction by re-training the authors’ model
- Propose a modified architecture (EEGNetMSSE)
- Compare baseline and proposed model performances across multiple EEG paradigms

---

# Data Preprocessing (`data/`)

The `data/` directory contains preprocessing scripts for each paradigm:
- `pretraitement_ern.py`
- `pretraitement_mrcp.py`
- `pretraitement_p300.py`
- `pretraitement_smr.py`

These scripts prepare raw EEG data and export them to `.npz` files used in the experiments.

Raw EEG datasets are not included in this repository.

# Commands (preprocessing)

Run from the repository root:

~~~bash
python data/ERN/pretraitement_ern.py
python data/MRCP/pretraitement_mrcp.py
python data/P300/pretraitement_p300.py
python data/SMR/pretraitement_smr.py
~~~

---

# Baseline Model — EEGNet Paper (`EEGNet_paper/`)

This part contains scripts used to re-train the original EEGNet-8,2 model using TensorFlow/Keras and the official EEGNet implementation.

Characteristics:
- Architecture: EEGNet-8,2 (`F1 = 8`, `D = 2`, `F2 = 16`)
- Training length: 500 epochs
- Model selection: minimum validation loss
- Evaluation protocols strictly follow the original paper

Paradigms:
- ERN
- MRCP
- P300
- SMR

Each paradigm directory contains training scripts and CSV files reporting experimental results.

# Training commands (paper baseline)

**Important:** some scripts require the official `EEGModels.py` from `arl-eegmodels`.
Use `<EEGMODELS_DIR>` to point to the folder that contains `EEGModels.py`.

### ERN

**Within-subject (4-fold):**
~~~bash
python EEGNet_paper/ERN/4_fold_within_subject/train_ern_within_subject.py ^
  --train_npz data/ERN/ern_train_eegnet.npz
~~~

**Cross-subject:**
~~~bash
python EEGNet_paper/ERN/cross_subject/train_ern_cross_subject.py ^
  --train_npz data/ERN/ern_train_eegnet.npz ^
  --test_npz  data/ERN/ern_test_eegnet.npz
~~~

### SMR

**Within-subject (4-fold):**
~~~bash
python EEGNet_paper/SMR/4_fold_within_subject/train_smr_within_subject_4fold.py ^
  --train_npz data/SMR/smr_train_eegnet.npz
~~~

**Cross-subject:**
~~~bash
python EEGNet_paper/SMR/cross_subject/train_smr_cross_subject.py ^
  --train_npz data/SMR/smr_train_eegnet.npz ^
  --test_npz  data/SMR/smr_test_eegnet.npz
~~~

### MRCP

MRCP paper scripts expect a directory containing per-subject files named `mrcp_P*_eegnet.npz`
(e.g., `mrcp_P01_eegnet.npz`, `mrcp_P02_eegnet.npz`, ...).

**Within-subject (4-fold):**
~~~bash
python EEGNet_paper/MRCP/4_fold_within_subject/train_mrcp_within_subject.py ^
  --data_dir data/MRCP ^
  --eegmodels_dir <EEGMODELS_DIR>
~~~

**Cross-subject:**
~~~bash
python EEGNet_paper/MRCP/cross_subject/train_mrcp_cross_subject.py ^
  --data_dir data/MRCP ^
  --eegmodels_dir <EEGMODELS_DIR>
~~~

### P300

**Within-subject (4-fold):**
~~~bash
python EEGNet_paper/P300/4_fold_within_subject/train_p300_within_subject.py ^
  --data_npz data/P300/p300_won2022_train_eegnet.npz ^
  --eegmodels_dir <EEGMODELS_DIR>
~~~

**Cross-subject:**
The script `EEGNet_paper/P300/cross_subject/train_eegnet_p300_cross_subject.py` uses hard-coded paths at the top
(e.g., `TRAIN_PATH`, `TEST_PATH`, `LOG_DIR`).
Edit them to point to:

- `data/P300/p300_won2022_train_eegnet.npz`
- `data/P300/p300_won2022_test_eegnet.npz`

Then run:
~~~bash
python EEGNet_paper/P300/cross_subject/train_eegnet_p300_cross_subject.py
~~~

---

# Our Model — EEGNetMSSE (`EEGNet_ours/`)

This part contains our proposed model, implemented in PyTorch.

Main files:
- `models_eegnet.py`: definition of the EEGNetMSSE architecture
- `run_eegnet.py`: unified training and evaluation script

Model characteristics:
- derived from EEGNet-8,2
- multi-scale temporal convolutions
- squeeze-and-excitation (SE) blocks
- same evaluation protocols as the baseline model

Final results are summarized in the following CSV files:
- `results_ERN_SMR_MRCP.csv`
- `results_p300_cross.csv`
- `results_p300_within.csv`

# Training commands (EEGNetMSSE)

### ERN

**Within-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset ern ^
  --protocol within ^
  --npz-train data/ERN/ern_train_eegnet.npz ^
  --use-class-weights
~~~

**Cross-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset ern ^
  --protocol cross ^
  --npz-train data/ERN/ern_train_eegnet.npz ^
  --npz-test  data/ERN/ern_test_eegnet.npz ^
  --use-class-weights
~~~

### SMR

**Within-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset smr ^
  --protocol within ^
  --npz-train data/SMR/smr_train_eegnet.npz
~~~

**Cross-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset smr ^
  --protocol cross ^
  --npz-train data/SMR/smr_train_eegnet.npz ^
  --npz-test  data/SMR/smr_test_eegnet.npz
~~~

### MRCP

For our runner, we use the aggregated file:

- `data/MRCP/mrcp_all_eegnet.npz`

**Within-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset mrcp ^
  --protocol within ^
  --npz-train data/MRCP/mrcp_all_eegnet.npz
~~~

**Cross-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset mrcp ^
  --protocol cross ^
  --npz-train data/MRCP/mrcp_all_eegnet.npz
~~~

### P300

**Within-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset p300 ^
  --protocol within ^
  --npz-train data/P300/p300_won2022_train_eegnet.npz ^
  --use-class-weights
~~~

**Cross-subject:**
~~~bash
python EEGNet_ours/run_eegnet.py ^
  --dataset p300 ^
  --protocol cross ^
  --npz-train data/P300/p300_won2022_train_eegnet.npz ^
  --npz-test  data/P300/p300_won2022_test_eegnet.npz ^
  --use-class-weights
~~~

---

# Input Data Format

EEG data are stored in `.npz` files.

Typical required keys:
- `X`: EEG data (N, C, T)
- `y`: labels
- `subject`: subject identifiers

Additional keys may be present depending on the paradigm :
`run`, `trial`, `session`, `fs`, `id_feedback`

---

# Execution Notes

- All experiments were executed using the scripts provided in this repository
- Scripts rely on command-line arguments defined internally
- Dataset paths depend on the local data organization
- No dataset paths are hard-coded in the repository (except the P300 cross-subject paper script, which must be edited)

---

# Evaluation Metrics

- ERN / P300 / MRCP: AUC ROC
- SMR: classification accuracy

Metrics are computed on the test set using the model selected by minimum validation loss.

---

# Reference

Lawhern et al.,  
EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces,  
Journal of Neural Engineering, 2018.

---

# Authors

- Sarah ABDALLAH
- Zakaria ALLOUCHE
- Alicia BERROUANE
- Mohamad ABBAS