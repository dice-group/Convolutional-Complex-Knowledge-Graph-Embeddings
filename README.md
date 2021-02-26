# Convolutional Complex Knowledge Graph Embeddings

This open-source project contains the Pytorch implementation of our approach (ConEx), training and evaluation scripts.
To foster further reproducible research and alleviate hardware requirements to reproduce the reported results, we provide pretrained ConEx on FB15K, FB15K-237, WN18, WN18RR and YAGO3-10.

## Link Prediction Results
In the below, we provide a brief overview of the link prediction results. Results are sorted in descending order of the size of the respective dataset.

#### YAGO3-10 ####
|         |   MRR | Hits@10 | Hits@3 | Hits@1  |
|---------|------:|--------:|-------:|--------:|
| DistMult| .340  | .540    |  .380  |  .240   |
| ComplEx | .360  | .550    |  .400  |  .260   |
| ConvE   | .400  | .620    |  .490  |  .350   |
| HypER   | .530  | .680    |  .580  |  .460   |
| RotatE  | .500  | .670    |  .550  |  .400   |
| ConEx   | ***.553***  | ***.696***    |  ***.601***  |  ***.477***   | 

#### FB15K (Freebase) ####
|         |   MRR | Hits@10 | Hits@3 | Hits@1 |
|---------|------:|-------:|-------:|--------:|
DistMult  | .654  | .824   | .733   | .546 
ComplEx   | .692  | .840   | .759   | .599 
ANALOGY   | .725  | .854   |  .785  | .646 
R-GCN     | .696  | .842   | .760   | .601 
TorusE    | .733  | .832   | .771   | .674 
ConvE     | .657  | .831   |  .723  | .558  
HypER     | .790  | .885   | .829   | .734 
SimplE    | .727  | .838   | .773   | .660 
TuckER    | .795  | .892   | .833   | .741 
ConEx     | ***.872***  | ***.930***   | ***.896***   | ***.837***

#### FB15K-237 (Freebase)

(*) denotes the newly reported link prediction results.

|         |   MRR | Hits@10 | Hits@3 | Hits@1 |
|---------|------:|-------:|-------:|--------:|
DistMult  |.241   | .419   | .263   | .155  
ComplEx   |.247   | .428   | .275   | .158 
ConvE     |.335   | .501   | .356   | .237 
RESCAL*   |.357   | .541   | .393   | .263  
DistMult* |.343   | .531   | .378   | .250  
ComplEx*  |.348   | .536   | .384   | .253 
ConvE*    |.339   | .521   | .369   | .248 
HypER     |.341   | .520   | .376   | .252 
NKGE      |.330   | .510   | .365   | .241 
RotatE    |.338   | .533   | .375   | .241
TuckER    |.358   | .544   | .394   | .266 
QuatE     |.366   | .556   | .401   | .271   
ConEx     |.360   | .547   | .397   | .266
ConEx-TuckER| ***.373***| ***.566*** | ***.411*** | ***.278***

#### WN18 (Wordnet)
|         |   MRR | Hits@10 | Hits@3 | Hits@1 |                                                            
|---------|------:|-------:|-------:|--------:|
DistMult  | .822 | .936    | .914   | .728    | 
ComplEx   | .941 | .947    | .936   | .936    | 
ANALOGY   | .942 | .947    | .944   | .939    |
R-GCN     | .819 | .964    | .929   | .697    |
TorusE    | .947 | .954    | .950   | .943    |
ConvE     | .943 | .956    | .946   | .935    | 
HypER     | .951 | .958    | .955   | .947    |
SimplE    | .942 | .947    | .944   | .939    |
TuckER    | .953 | .958    | .955   | .949    |
QuatE     | .950 | .962    | .954   | .944    |
ConEx     | ***.976*** | ***.980***    | ***.978***   | ***.976***    |



#### WN18RR (Wordnet)
(*) denotes the newly reported link prediction results.

|         |   MRR | Hits@10 | Hits@3 | Hits@1 |
|---------|------:|-------:|-------:|--------:|
DistMult  | .430  | .490   | .440   | .390 
ComplEx   |.440   | .510   | .460   | .410 
ConvE     |.430   | .520   | .440   | .400 
RESCAL*   |.467   | .517   | .480   | .439  
DistMult* |.452   | .530   | .466   | .413  
ComplEx*  |.475   | .547   | .490   | .438 
ConvE*    |.442   | .504   | .451   | .411 
HypER     |.465   | .522   | .477   | .436 
NKGE      |.450   | .526   | .465   | .421 
RotatE    |.476   | .571   | .492   | .428 
TuckER    |.470   | .526   | .482   | .443 
QuatE     |.482   | .572   | .499   | .436 
ConEx     |.481   | .550   | .493   | .448 
ConEx-TuckER| ***.483***| ***.549*** | ***.494*** | ***.449***


## WN18RR* dataset
We spot flaws on WN18RR, FB15K-237 and YAGO3-10. More specifically, the validation and test splits of the dataset contain entities that do not occur in the training split

## Visualisation of Embeddings
A 2D PCA projection of relation embeddings on the FB15K-237 dataset.
![alt text](util/rel_emb_fb15k_237-1.png)


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
Please use Python 3.7.1, if you would like to use cuda 11 as  Python 3.6.x and 3.8.x cause stalling when a GPU.


## Usage
+ ```run_script.py``` can be used to train ConEx on a desired dataset.
+ ```grid_search.py``` can be used to rerun our experiments.
+ ```reproduce_script.py``` can be  used to reproduce any pretrained models.

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