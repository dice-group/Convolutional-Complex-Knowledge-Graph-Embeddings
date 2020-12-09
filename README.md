# Convolutional Complex Knowledge Graph Embeddings

This open-source project contains the Pytorch implementation of our approach (ConEx), training and evaluation scripts.
To foster further reproducible research and alleviate hardware requirements to reproduce the reported results, we provide pretrained ConEx on FB15K, FB15K-237, WN18, WN18RR and YAGO3-10.

## Link Prediction Results

#### WN18 (Wordnet)

|         |   MRR | Hits@1 | Hits@3 | Hits@10 |                                                            
|---------|------:|-------:|-------:|--------:|
| ConEx   | 0.976 | 0.973  | 0.978  |   0.980 |   

#### FB15K (Freebase)

|         |   MRR | Hits@1 | Hits@3 | Hits@10 |
|---------|------:|-------:|-------:|--------:|
| ConEx   | 0.872 | 0.837  | 0.897  |   0.930 |   

#### FB15K-237 (Freebase)
|         |   MRR | Hits@1 | Hits@3 | Hits@10 |
|---------|------:|-------:|-------:|--------:|
| ConEx   | 0.352 |  0.260 |  0.388 |   0.538 |   

#### WN18RR (Wordnet)
|         |   MRR | Hits@1 | Hits@3 | Hits@10 |
|---------|------:|-------:|-------:|--------:|
| ConEx   | 0.481 |  0.448 |  0.493 |   0.550 | 

#### WN18RR* (Wordnet)
|         |   MRR | Hits@1 | Hits@3 | Hits@10 |
|---------|------:|-------:|-------:|--------:|
| ConEx   | 0.51 |  0.48 |  0.52 |   0.58 | 


#### YAGO3-10
|         |   MRR | Hits@1 | Hits@3 | Hits@10 |
|---------|------:|-------:|-------:|--------:|
| ConEx   | 0.55  |  0.47  |  0.60  |  0.70   | 

## WN18RR* dataset
We spot flaws on WN18RR, FB15K-237 and YAGO3-10. More specifacally, the validation and test splits of the dataset contain entities that do not occur in the training split


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
The code is compatible with Python 3.6.4


## Reproduce link prediction results
- Unzip the datasets: ```unzip KGs.zip```
- Download pretrained models via [Google Drive](https://drive.google.com/drive/folders/1QkI6C3otXU7xylt_JDtFTf6VybU0Q2bH?usp=sharing) :```unzip PretrainedModels.zip```
- Reproduce reported link prediction results: ``` python reproduce_lp.py```
- Reproduce reported link prediction results on WN18RR*, FB15K-237* and YAGO3-10*:``` python reproduce_lp_new.py```



## Acknowledgement 
We based our implementation on the open source implementation of [TuckER](https://github.com/ibalazevic/TuckER).

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