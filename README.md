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
- Create a folder for pretrained models: ```mkdir PretrainedModels```
- Download pretrained models via [hobbitdata](https://hobbitdata.informatik.uni-leipzig.de/KGE/conex/) into ```PretrainedModels```.
- ```python reproduce_lp.py``` reproduces link prediction results on the FB15K-237, FB15K, WN18, WN18RR and YAGO3-10 benchmark datasets.
- ```python reproduce_baselines.py``` reproduces link prediction results of DistMult, ComplEx and TuckER on the FB15K-237, WN18RR and YAGO3-10 benchmark datasets.
- ```settings.json``` files store the hyperparameter setting for each model.
- ```python reproduce_ensemble.py``` reports link prediction results of ensembled models.
- ```python reproduce_lp_new.py``` reports link prediction results on WN18RR*, FB15K-237* and YAGO3-10*.


## Link Prediction Results
In the below, we provide a brief overview of the link prediction results.
#### YAGO3-10 ####
|         |   MRR | Hits@10 | Hits@3 | Hits@1  |
|---------|------:|--------:|-------:|--------:|
| [DistMult](https://arxiv.org/pdf/1707.01476.pdf) | .340     | .540     |  .380    |  .240    |
| [ComplEx](https://arxiv.org/pdf/1707.01476.pdf)  | .360     | .550     |  .400    |  .260    |
| [ConvE](https://arxiv.org/pdf/1707.01476.pdf)    | .440     | .620     |  .490    |  .350    |
| [HypER](https://arxiv.org/pdf/1808.07018.pdf)    | .530     | .678     |  .580    |  .455    |
| [RotatE](https://arxiv.org/pdf/1902.10197.pdf)   | .495     | .670     |  .550    |  .400    |
| DistMult                                         | .543     | .683     |  .590    |  .466    |
| ComplEx                                          | .547     | .690     |  .594    |  .468    | 
| TuckER                                           | .427     | .609     |  .476    |  .331    |
| ConEx                                            |***.553***|***.696***|***.601***|***.474***| 

#### FB15K-237 ####
|         |   MRR | Hits@10 | Hits@3 | Hits@1  |
|---------|------:|--------:|-------:|--------:|
| [DistMult](https://arxiv.org/pdf/1707.01476.pdf) | .241     | .419     |  .263    |  .155    |
| [ComplEx](https://arxiv.org/pdf/1707.01476.pdf)  | .247     | .428     |  .275    |  .158    |
| [ConvE](https://arxiv.org/pdf/1707.01476.pdf)    | .335     | .501     |  .356    |  .237    |
| [DistMult](https://github.com/uma-pi1/kge-iclr20)| .343     | .531     |  .378    |  .250    |
| [ComplEx](https://github.com/uma-pi1/kge-iclr20) | .348     | .536     |  .384    |  .253    |
| [ConvE](https://github.com/uma-pi1/kge-iclr20)   | .339     | .521     |  .369    |  .248    |
| [RotatE](https://arxiv.org/pdf/1902.10197.pdf)   | .338     | .533     |  .375    |  .241    |
| [HypER](https://arxiv.org/pdf/1808.07018.pdf)    | .341     | .520     |  .376    |  .252    |
| DistMult                                         | .353     | .539     |  .390    |  .260    |
| ComplEx                                          | .332     | .509     |  .366    |  .244    |
| TuckER                                           | .363     | .553     |  .400    |  .268    |
| ConEx                                            | .366     | .555     |  .403     |  .271    | 
| Ensemble of ConEx                                |***.376***|***.570***|***.415***|***.279***| 


## Acknowledgement 
We based our implementation on the open source implementation of [TuckER](https://github.com/ibalazevic/TuckER). We would like to thank for the readable codebase.

## How to cite
```
@inproceedings{demir2021convolutional,
               title={Convolutional Complex Knowledge Graph Embeddings},
               author={Caglar Demir and Axel-Cyrille Ngonga Ngomo},
               booktitle={Eighteenth Extended Semantic Web Conference - Research Track},
               year={2021},
               url={https://openreview.net/forum?id=6T45-4TFqaX}}
```

For any further questions or suggestions, please contact:  ```caglar.demir@upb.de```