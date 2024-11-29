

### 1. Note on datasets and directories
Due to the large size of datasets *ML-10M*, *Amazon* and *Tmall*, we have compressed them into zip files. Please unzip them before running the model on these datasets. For *Yelp* and *Gowalla*, keeping the current directory structure is fine.

Before running the codes, please ensure that two directories `log/` and `saved_model/` are created under the root directory. They are used to store the training results and the saved model and optimizer states.

### 2. Running environment

We develope our codes in the following environment:

```
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```
