import json
from util.helper_funcs import *
from util.helper_classes import HeadAndRelationBatchLoader
from models.complex_models import *
from models.real_models import *
from torch.optim.lr_scheduler import ExponentialLR
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Fixing the random sees.
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)




class Experiment:
    """
    Experiment class for training and evaluation
    """

    def __init__(self, *, dataset, model, parameters, ith_logger, store_emb_dataframe=False):

        self.dataset = dataset
        self.model = model
        self.store_emb_dataframe = store_emb_dataframe

        self.embedding_dim = parameters['embedding_dim']
        self.num_of_epochs = parameters['num_of_epochs']
        self.learning_rate = parameters['learning_rate']
        self.batch_size = parameters['batch_size']
        self.decay_rate = parameters['decay_rate']
        self.label_smoothing = parameters['label_smoothing']
        self.cuda = torch.cuda.is_available()
        self.num_of_workers = parameters['num_workers']
        self.optimizer = None
        self.entity_idxs, self.relation_idxs, self.scheduler = None, None, None

        self.negative_label = 0.0
        self.positive_label = 1.0

        # Algorithm dependent hyper-parameters
        self.kwargs = parameters
        self.kwargs['model'] = self.model

        self.storage_path, _ = create_experiment_folder()
        self.logger = create_logger(name=self.model + ith_logger, p=self.storage_path)

        self.logger.info('Cuda available:{0}'.format(self.cuda))
        if 'norm_flag' not in self.kwargs:
            self.kwargs['norm_flag'] = False

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    @staticmethod
    def get_er_vocab(data):
        # head entity and relation
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    @staticmethod
    def get_re_vocab(data):
        # relation and tail entity
        re_vocab = defaultdict(list)
        for triple in data:
            re_vocab[(triple[1], triple[2])].append(triple[0])
        return re_vocab

    def get_batch_1_to_N(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.ones((len(batch), len(self.dataset.entities))) * self.negative_label
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = self.positive_label
        return np.array(batch), torch.FloatTensor(targets)

    def evaluate_one_to_n(self, model, data, log_info='Evaluate one to N.'):
        """
         Evaluate model
        """
        self.logger.info(log_info)
        hits = []
        ranks = []
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
            #total_test_loss += model.loss(predictions, _).cpu().detach().numpy()  # store also test error.
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

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)

        hit_1 = sum(hits[0]) / (float(len(data)))
        hit_3 = sum(hits[2]) / (float(len(data)))
        hit_10 = sum(hits[9]) / (float(len(data)))
        mean_rank = np.mean(ranks)
        mean_reciprocal_rank = np.mean(1. / np.array(ranks))

        self.logger.info(f'Hits @10: {hit_10}')
        self.logger.info(f'Hits @3: {hit_3}')
        self.logger.info(f'Hits @1: {hit_1}')
        self.logger.info(f'Mean rank: {mean_rank}')
        self.logger.info(f'Mean reciprocal rank: {mean_reciprocal_rank}')
        #self.logger.info(f'Total Test Loss: {total_test_loss}')

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
                   'MR': mean_rank, 'MRR': mean_reciprocal_rank}#, 'TestLoss': total_test_loss}

        return results

    def __describe_exp(self):
        self.logger.info("Info pertaining to dataset:{0}".format(self.dataset.info))
        self.logger.info("Number of triples in training data:{0}".format(len(self.dataset.train_data)))
        self.logger.info("Number of triples in validation data:{0}".format(len(self.dataset.valid_data)))
        self.logger.info("Number of triples in testing data:{0}".format(len(self.dataset.test_data)))
        self.logger.info("Number of entities:{0}".format(len(self.entity_idxs)))
        self.logger.info("Number of relations:{0}".format(len(self.relation_idxs)))
        self.logger.info("HyperParameter Settings:{0}".format(self.kwargs))

        if self.kwargs['norm_flag']:
            try:
                assert self.kwargs['input_dropout'] == 0.0
                assert self.kwargs['hidden_dropout'] == 0.0
            except AssertionError:
                self.logger.info('No use of dropout allowed if unit norm used.')
                self.logger.info('Dropouts will be set to .0')
                self.kwargs['input_dropout'] = 0
                self.kwargs['hidden_dropout'] = 0

    def eval(self, model):
        """
        trained model
        """
        if self.dataset.test_data:
            results = self.evaluate_one_to_n(model, self.dataset.test_data,
                                             'Standard Link Prediction evaluation on Testing Data')
            with open(self.storage_path + '/results.json', 'w') as file_descriptor:
                num_param = sum([p.numel() for p in model.parameters()])
                results['Number_param'] = num_param
                results.update(self.kwargs)
                json.dump(results, file_descriptor)

    def train(self, model):
        """ Training."""
        if self.cuda:
            model.cuda()

        model.init()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            self.scheduler = ExponentialLR(self.optimizer, self.decay_rate)

        self.logger.info("{0} starts training".format(model.name))
        num_param = sum([p.numel() for p in model.parameters()])
        self.logger.info("'Number of free parameters: {0}".format(num_param))
        # Store the setting.
        with open(self.storage_path + '/settings.json', 'w') as file_descriptor:
            json.dump(self.kwargs, file_descriptor)

        if self.kwargs['scoring_technique'] == 'KvsAll':
            model = self.k_vs_all_training_schema(model)
        elif self.kwargs['scoring_technique'] == 'AllvsAll':
            raise NotImplementedError('Implementation of AllvsAll raises an exception if cuda used.')
        # We may implement the negative sampling technique.
        else:
            raise ValueError

        # Save the trained model.
        torch.save(model.state_dict(), self.storage_path + '/model.pt')
        # Save embeddings of entities and relations in csv file.
        if self.store_emb_dataframe:
            entity_emb, emb_rel = model.get_embeddings()
            # pd.DataFrame(index=self.dataset.entities, data=entity_emb.numpy()).to_csv(TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            pd.DataFrame(index=self.dataset.entities, data=entity_emb.numpy()).to_csv(
                '{0}/{1}_entity_embeddings.csv'.format(self.storage_path, model.name))
            pd.DataFrame(index=self.dataset.relations, data=emb_rel.numpy()).to_csv(
                '{0}/{1}_relation_embeddings.csv'.format(self.storage_path, model.name))

    def train_and_eval(self):
        """
        Train and evaluate phases.
        """

        self.entity_idxs = {self.dataset.entities[i]: i for i in range(len(self.dataset.entities))}
        self.relation_idxs = {self.dataset.relations[i]: i for i in range(len(self.dataset.relations))}

        self.kwargs.update({'num_entities': len(self.entity_idxs),
                            'num_relations': len(self.relation_idxs)})
        self.kwargs.update(self.dataset.info)
        model = None
        if self.model == 'ConEx':
            model = ConEx(self.kwargs)
        elif self.model == 'ConExNeg':
            model = ConExNeg(self.kwargs)
        elif self.model == 'Distmult':
            model = Distmult(self.kwargs)
        elif self.model == 'Tucker':
            model = Tucker(self.kwargs)
        elif self.model == 'Complex':
            model = Complex(self.kwargs)
        else:
            print(self.model, ' is not valid name')
            raise ValueError

        self.train(model)
        self.eval(model)

    def k_vs_all_training_schema(self, model):
        self.logger.info('k_vs_all_training_schema starts')

        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        losses = []

        head_to_relation_batch = DataLoader(
            HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(train_data_idxs), num_e=len(self.dataset.entities)),
            batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)

        # To indicate that model is not trained if for if self.num_of_epochs=0
        loss_of_epoch, it = -1, -1

        for it in range(1, self.num_of_epochs + 1):
            loss_of_epoch = 0.0
            # given a triple (e_i,r_k,e_j), we generate two sets of corrupted triples
            # 1) (e_i,r_k,x) where x \in Entities AND (e_i,r_k,x) \not \in KG
            for head_batch in head_to_relation_batch:  # mini batches
                e1_idx, r_idx, targets = head_batch
                if self.cuda:
                    targets = targets.cuda()
                    r_idx = r_idx.cuda()
                    e1_idx = e1_idx.cuda()

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

                self.optimizer.zero_grad()
                loss = model.forward_head_and_loss(e1_idx, r_idx, targets)
                loss_of_epoch += loss.item()
                loss.backward()
                self.optimizer.step()
            if self.decay_rate:
                self.scheduler.step()
            losses.append(loss_of_epoch)
        self.logger.info('Loss at {0}.th epoch:{1}'.format(it, loss_of_epoch))
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
        model.eval()
        return model

    def all_vs_all_training_schema(self, model):
        # We investigated all vs all training schema inspired by 1vsN/ KvsAll. However we did not include it into
        # our experiments.
        raise NotImplementedError
        self.logger.info('all_vs_all_training_schema starts')

        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        losses = []

        loss_of_epoch = 0
        it = 0
        head_to_relation_batch = DataLoader(
            HeadAndRelationBatchLoader(er_vocab=self.get_er_vocab(train_data_idxs), num_e=len(self.dataset.entities)),
            batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)
        relation_to_tail_batch = DataLoader(
            RelationAndTailBatchLoader(re_vocab=self.get_re_vocab(train_data_idxs), num_e=len(self.dataset.entities)),
            batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)

        for it in range(1, self.num_of_epochs + 1):
            loss_of_epoch = 0.0
            # given a triple (e_i,r_k,e_j), we generate two sets of corrupted triples
            # 1) (e_i,r_k,x) where x \in Entities AND (e_i,r_k,x) \not \in KG
            # 2) (x,r_k,e_j) where x \in Entities AND (x,r_k,e_j) \not \in KG.
            for (head_batch, tail_batch) in zip(head_to_relation_batch, relation_to_tail_batch):  # mini batches
                e1_idx, r_idx, targets = head_batch
                r_idx_2, e2_idx, targets_2 = tail_batch

                if self.cuda:
                    targets = targets.cuda()
                    r_idx = r_idx.cuda()
                    e1_idx = e1_idx.cuda()
                    targets_2 = targets_2.cuda()
                    r_idx_2 = r_idx_2.cuda()
                    e2_idx = e2_idx.cuda()

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                    targets_2 = ((1.0 - self.label_smoothing) * targets_2) + (1.0 / targets_2.size(1))

                    self.optimizer.zero_grad()
                    head_loss = model.forward_head_and_loss(e1_idx, r_idx, targets)
                    loss_of_epoch += head_loss.item()
                    tail_loss = model.forward_tail_and_loss(r_idx_2, e2_idx, targets_2)
                    loss_of_epoch += tail_loss.item()
                    loss = head_loss + tail_loss
                    loss.backward()
                    self.optimizer.step()

            if self.decay_rate:
                self.scheduler.step()

            losses.append(loss_of_epoch)
        self.logger.info('Loss at {0}.th epoch:{1}'.format(it, loss_of_epoch))
        losses = np.array(losses)
        np.savetxt(self.storage_path + "/loss_per_epoch.csv", losses, delimiter=",")
        model.eval()
        return model

class Analyser:
    def __init__(self):
        self.train_loss = None

    def apply(self, exp):
        from sklearn.decomposition import PCA
        self.train_loss = np.loadtxt(fname=exp.storage_path + "/loss_per_epoch.csv", delimiter=",")
        plt.plot(self.train_loss)
        plt.show()

        entity_emb = pd.read_csv(exp.storage_path + "/{0}_entity_embeddings.csv".format(exp.model),
                                 index_col=0).to_numpy()
        rel_emb = pd.read_csv(exp.storage_path + "/{0}_relation_embeddings.csv".format(exp.model),
                              index_col=0).to_numpy()

        low_X = PCA(n_components=2).fit_transform(entity_emb)
        plt.scatter(low_X[:, 0], low_X[:, 1])
        plt.title('Entity emb')

        plt.show()

        low_X = PCA(n_components=2).fit_transform(rel_emb)
        plt.scatter(low_X[:, 0], low_X[:, 1])
        plt.title('Relation emb')
        plt.show()

        exit(1)
