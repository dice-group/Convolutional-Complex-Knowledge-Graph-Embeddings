# Convolutional Complex Knowledge Graph Embeddings

This open-source project contains the Pytorch implementation of our approach (ConEx), training and evaluation scripts as well as pretrained models.


## Installation

First clone the repository:
```
git clone https://github.com/dice-group/Convolutional-Complex-Knowledge-Graph-Embeddings.git
```
Then obtain the required libraries:
```
conda env create -f environment.yml
source activate conex
```
The code is compatible with Python 3.6.4.


## Usage
+ ```run_script.py``` can be used to train ConEx on a desired dataset.
+ ```grid_search.py``` can be used to rerun our experiments.

## Reproduce link prediction results
Please follow the next steps to reproduce all reported results.
- Unzip the datasets: ```unzip KGs.zip```
- Download pretrained models via [Google Drive](https://drive.google.com/drive/folders/1QkI6C3otXU7xylt_JDtFTf6VybU0Q2bH?usp=sharing)
- ```unzip PretrainedModels.zip```
- ```python reproduce_lp.py``` reproduces link prediction results on the FB15K-237, FB15K, WN18, WN18RR and YAGO3-10 benchmark datasets.
- ```python reproduce_baselines.py``` reproduces link prediction results of DistMult, ComplEx and TuckER on the FB15K-237, WN18RR and YAGO3-10 benchmark datasets.
- ```settings.json``` files store the hyperparameter setting for each model.
- ```python reproduce_ensemble.py``` reports link prediction results of ensembled models.
- ```python reproduce_lp_new.py``` reports link prediction results on WN18RR*, FB15K-237* and YAGO3-10*.

## Acknowledgement 
We based our implementation on the open source implementation of [TuckER](https://github.com/ibalazevic/TuckER). We would like to thank for the readable codebase.

## How to cite
```
@inproceedings{
  XXX,
  title={Convolutional Complex Knowledge Graph Embeddings},
  author={Caglar Demir and and Axel-Cyrille Ngonga Ngomo},
  booktitle={XXX},
  year={2021},
}
```

For any further questions, please contact:  ```caglar.demir@upb.de```