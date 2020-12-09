import json
from util.data import Data
from util.helper_funcs import *
from models.complex_models import Complex, ConEx
from models.real_models import Distmult, Tucker
from torch.optim.lr_scheduler import ExponentialLR
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
# CUDA for PyTorch
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


class HeadAndRelationBatchLoader(torch.utils.data.Dataset):
    def __init__(self, er_vocab, num_e):
        self.num_e = num_e
        head_rel_idx = torch.Tensor(list(er_vocab.keys())).long()
        self.head_idx = head_rel_idx[:, 0]
        self.rel_idx = head_rel_idx[:, 1]
        self.tail_idx = list(er_vocab.values())
        assert len(self.head_idx) == len(self.rel_idx) == len(self.tail_idx)

    def __len__(self):
        return len(self.tail_idx)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.num_e)
        y_vec[self.tail_idx[idx]] = 1  # given head and rel, set 1's for all tails.
        return self.head_idx[idx], self.rel_idx[idx], y_vec


class RelationAndTailBatchLoader(torch.utils.data.Dataset):
    """
    DO NOT USE YET.

    The idea of implementing this class as follows:
    1. given (s,p,o) regardless of p is inverse or not, we perform KvsAll/1vsN on (s,p,?) during training.
    2. We were wondering the possible impact of performing KvsAll on (?,p,o).
    3. Note that we do not distinguish whether p is inverse or not.
    4. Our initial experiments on UMLS and KINSHIP show that training with (s,p,?) and (?,p,o) yield better results than training only
    on (s,p,?). However, this superior performance may stem from the fact that the former case updates weights twice more.
    5. We aimed to test this change on larger datasets but we obtain ****CUDA ASSERTION EXCEPTION****.
    6. We have not investigate yet the reason causing such exception.
    """

    def __init__(self, re_vocab, num_e):
        self.num_e = num_e
        rel_head_idx = torch.Tensor(list(re_vocab.keys())).long()
        self.rel_idx = rel_head_idx[:, 0]
        self.tail_idx = rel_head_idx[:, 1]
        self.head_idx = list(re_vocab.values())
        assert len(self.head_idx) == len(self.rel_idx) == len(self.tail_idx)

    def __len__(self):
        return len(self.tail_idx)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.num_e)
        y_vec[self.head_idx[idx]] = 1  # given head and rel, set 1's for all tails.
        return self.rel_idx[idx], self.tail_idx[idx], y_vec


class Reproduce:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.file_path = None
        self.kwargs = None

        self.entity_idxs = None
        self.relation_idxs = None

        self.cuda = torch.cuda.is_available()

        self.batch_size = None
        self.negative_label = 0
        self.positive_label = 1

    @staticmethod
    def get_er_vocab(data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    @staticmethod
    def get_head_tail_vocab(data):
        head_tail_vocab = defaultdict(list)
        for triple in data:
            head_tail_vocab[(triple[0], triple[2])].append(triple[1])
        return head_tail_vocab

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    def get_batch_1_to_N(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.ones((len(batch), len(self.dataset.entities))) * self.negative_label
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = self.positive_label
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate_link_prediction(self, model, data, per_rel_flag_=True):
        hits = []
        ranks = []

        rank_per_relation = dict()
        for i in range(10):
            hits.append([])
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(self.dataset.data))
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch_1_to_N(er_vocab, test_data_idxs, i)

            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward_head_batch(e1_idx=e1_idx, rel_idx=r_idx)
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()

            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                rank_per_relation.setdefault(self.dataset.relations[r_idx[j]], []).append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)

        print('Hits @10: {0}'.format(sum(hits[9]) / (float(len(data)))))
        print('Hits @3: {0}'.format(sum(hits[2]) / (float(len(data)))))
        print('Hits @1: {0}'.format(sum(hits[0]) / (float(len(data)))))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('MRR: {0}'.format(np.mean(1. / np.array(ranks))))
        print('###############################')
        if per_rel_flag_:
            for k, v in rank_per_relation.items():
                if '_reverse' in k:
                    continue
                # Given (h,r,t)
                reciprocal_tail_entity_rankings = 1. / np.array(v)  # ranks_t => RANKS of true tails
                reciprocal_head_entity_rankings = 1. / np.array(
                    rank_per_relation[k + '_reverse'])  # ranks_t => RANKS of true
                assert len(reciprocal_head_entity_rankings) == len(reciprocal_tail_entity_rankings)
                sum_reciprocal_ranks = np.sum(reciprocal_head_entity_rankings + reciprocal_tail_entity_rankings)
                print('MRR:{0}: {1}'.format(k, sum_reciprocal_ranks / ((float(len(v))) * 2)))
            # with directions
            # for k, v in rank_per_relation.items():
            #    print('MRR:{0}: {1}'.format(k, np.mean(1. / np.array(v))))

    def reproduce(self, model_path, data_path, model_name, per_rel_flag_=False, tail_pred_constraint=False,
                  out_of_vocab_flag=False):
        with open(model_path + '/settings.json', 'r') as file_descriptor:
            self.kwargs = json.load(file_descriptor)

        self.dataset = Data(data_dir=data_path, tail_pred_constraint=tail_pred_constraint,
                            out_of_vocab_flag=out_of_vocab_flag)
        model = self.load_model(model_path=model_path, model_name=model_name)
        print('Evaluate:', self.model)
        print('Number of free parameters: ', sum([p.numel() for p in model.parameters()]))

        self.entity_idxs = {self.dataset.entities[i]: i for i in range(len(self.dataset.entities))}
        self.relation_idxs = {self.dataset.relations[i]: i for i in range(len(self.dataset.relations))}
        self.batch_size = self.kwargs['batch_size']
        print('Link Prediction Results on Testing')
        self.evaluate_link_prediction(model, self.dataset.test_data, per_rel_flag_)

    def load_model(self, model_path, model_name):
        self.model = model_name
        with open(model_path + '/settings.json', 'r') as file_descriptor:
            self.kwargs = json.load(file_descriptor)

        model = None
        if self.model == 'ConEx':
            model = ConEx(self.kwargs)
        elif self.model == 'OMult':
            model = OMult(self.kwargs)
        elif self.model == 'ConvO':
            model = ConvO(self.kwargs)
        elif self.model == 'QMult':
            model = QMult(self.kwargs)
        elif self.model == 'ConvQ':
            model = ConvQ(self.kwargs)
        elif self.model == 'Tucker':
            model = Tucker(self.kwargs)
        elif self.model == 'Rescal':
            model = Rescal(self.kwargs)
        elif self.model == 'Distmult':
            model = Distmult(self.kwargs)
        elif self.model == 'Complex':
            model = Complex(self.kwargs)
        elif self.model == 'Conve':
            model = Conve(self.kwargs)
        else:
            print(self.model, ' is not valid name')
            raise ValueError

        m = torch.load(model_path + '/model.pt', torch.device('cpu'))

        model.load_state_dict(m)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        if self.cuda:
            model.cuda()

        return model

    def reproduce_ensemble(self, model, data_path, per_rel_flag_=False, tail_pred_constraint=False,
                           out_of_vocab_flag=False):
        """
        per_rel_flag_ reports link prediction results per relations.
        flag_of_removal  -> removes triples from testing split containing entities that did not occur during training  at testing time.

        lp_based_on_head_and_tail_entity_rankings-> computes rank of missing entities based on head and tail entity.
        """
        self.dataset = Data(data_dir=data_path, tail_pred_constraint=tail_pred_constraint,
                            out_of_vocab_flag=out_of_vocab_flag)
        self.batch_size = 1024
        self.entity_idxs = {self.dataset.entities[i]: i for i in range(len(self.dataset.entities))}
        self.relation_idxs = {self.dataset.relations[i]: i for i in range(len(self.dataset.relations))}
        print('Link Prediction Results of Ensemble of {0} on Testing'.format(model.name))
        self.evaluate_link_prediction(model, self.dataset.test_data, per_rel_flag_)
