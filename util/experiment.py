import json
from util.helper_funcs import *
from util.helper_classes import HeadAndRelationBatchLoader
from util.helper_classes import TriplesBatchLoader
from models.complex_models import *
from models.real_models import *
from torch.optim.lr_scheduler import ExponentialLR
from collections import defaultdict
from torch.utils.data import DataLoader
import pandas as pd
# Fixing the random seeds.
# seed = 1
# np.random.seed(seed)
# torch.manual_seed(seed)


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
        self.neg_ratio = parameters['neg_ratio']
        self.decay_rate = parameters['decay_rate']
        self.label_smoothing = parameters['label_smoothing']
        self.optim = parameters['optim']
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
    def get_ere_vocab(data):
        # head entity and relation and tail entity
        ere_vocab = defaultdict(list)
        i = 0
        for triple in data:
            if not ere_vocab[triple[0], triple[1],triple[2]]:
                ere_vocab[triple[0], triple[1],triple[2]].append(1)
                i = i + 1

        return ere_vocab

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

    def get_batch_1_to_1(self, ere_vocab, ere_vocab_pairs, idx):
        batch = ere_vocab_pairs[idx:idx + self.batch_size]
        targets = np.ones((len(batch))) * self.negative_label
        for idx, pair in enumerate(batch):
            targets[idx] = self.positive_label
        return np.array(batch), torch.FloatTensor(targets)

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

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
                   'MR': mean_rank, 'MRR': mean_reciprocal_rank}

        return results

    def eval(self, model):
        """
        trained model
        """
        if self.dataset.test_data:
            if model.name.__contains__("1to1"):
                results = self.evaluate_one_to_one(model, self.dataset.test_data,
                                                 'Standard Link Prediction evaluation on Testing Data')
            else:
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
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optim == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate)
        else:
            print(f'Please provide valid name for optimizer. Currently => {self.optim}')
            raise ValueError
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
        # We may implement the negative sampling technique.
        elif self.kwargs['scoring_technique'] == '1vs1':
            model = self.one_vs_one_training_schema(model)
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
        elif self.model == 'Distmult':
            model = Distmult(self.kwargs)
        elif self.model == 'Tucker':
            model = Tucker(self.kwargs)
        elif self.model == 'Complex':
            model = Complex(self.kwargs)
        elif self.model == 'Complex1to1':
            model = Complex1to1(self.kwargs)
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

    # This function takes the training data in 1-1 fashion and generate corresponding
    # negative examples depending upon neg ratio
    def one_vs_one_training_schema(self, model):
        self.logger.info('1_vs_1_training_schema starts')
        print(int(len(self.dataset.train_data) / 100))
        train_data_idxs = self.get_data_idxs(self.dataset.train_data)
        losses = []
        ere_vocab = self.get_ere_vocab(train_data_idxs)
        head_to_relation_batch = DataLoader(
            TriplesBatchLoader(ere_vocab=ere_vocab, num_e=len(self.dataset.entities)),
            batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)

        # To indicate that model is not trained if for if self.num_of_epochs=0
        loss_of_epoch, it = -1, -1

        # Added for neg sampling   neg_ratio set = 2 or any

        self.new_triples_indexes = np.empty((self.batch_size * (self.neg_ratio + 1), 3), dtype=np.int64)     # FB15k-237 16326,3
        self.new_triples_values = np.empty((self.batch_size * (self.neg_ratio + 1)), dtype=np.float32)      # FB15k-237 16326,0

        # =======
        print(self.batch_size)
        print(len(head_to_relation_batch.dataset))

        # exit(1)

        for it in range(1, self.num_of_epochs + 1):
            loss_of_epoch = 0.0
            # if neg_ratio > 0
            # given a triple (e_i,r_k,e_j), we generate two sets of corrupted triples
            # 1) (e_i,r_k,x) where x \in Entities AND (e_i,r_k,x) \not \in KG
            for head_batch in head_to_relation_batch:  # mini batches
                # added for neg sampling
                last_idx = self.batch_size              # 5442
                if self.batch_size != len(head_batch[0]):
                    continue
                if self.neg_ratio > 0:
                    self.new_triples_indexes[:self.batch_size,0] = np.array(head_batch[0])          # FB15k-237 16326
                    self.new_triples_indexes[:self.batch_size, 1] = np.array(head_batch[1])
                    self.new_triples_indexes[:self.batch_size, 2] = np.array(head_batch[2])
                    self.new_triples_values[:self.batch_size] = np.array(head_batch[3])


                    # print(np.random.randint(0, len(self.dataset.entities), last_idx * self.neg_ratio))


                    # Pre-sample everything, faster
                    rdm_entities = np.random.randint(0, len(self.dataset.entities), last_idx * self.neg_ratio)
                    rdm_choices = np.random.random(last_idx * self.neg_ratio) < 0.5

                    # print(rdm_entities.shape)
                    # print(len(rdm_choices))

                    # Pre copying everyting
                    # print(np.tile(self.new_triples_indexes[:last_idx, :], (self.neg_ratio, 1)).shape)
                    # print(self.new_triples_indexes.shape)
                    # print(last_idx)
                    # print(self.new_triples_indexes[last_idx:(last_idx * (self.neg_ratio + 1)), :].shape)

                    self.new_triples_indexes[last_idx:(last_idx * (self.neg_ratio + 1)), :] = np.tile(
                        self.new_triples_indexes[:last_idx, :], (self.neg_ratio, 1))
                    self.new_triples_values[last_idx:(last_idx * (self.neg_ratio + 1))] = np.tile(
                        self.new_triples_values[:last_idx], self.neg_ratio)


                    for i in range(last_idx):
                        for j in range(self.neg_ratio):
                            cur_idx = i * self.neg_ratio + j
                            # Sample a random subject or object
                            if rdm_choices[cur_idx]:
                                # head_batch[0].T[last_idx + cur_idx]  = rdm_entities[cur_idx]
                                self.new_triples_indexes[last_idx + cur_idx, 0] = rdm_entities[cur_idx]
                            else:
                                self.new_triples_indexes[last_idx + cur_idx, 2] = rdm_entities[cur_idx]

                            self.new_triples_values[last_idx + cur_idx] = 0

                    last_idx += cur_idx + 1
                    # print(self.new_triples_indexes)

                    e1_idx, r_idx, e2_idx, targets = torch.from_numpy(self.new_triples_indexes[:, 0]), torch.from_numpy(
                        self.new_triples_indexes[:, 1]), torch.from_numpy(
                        self.new_triples_indexes[:, 2]), torch.from_numpy(self.new_triples_values)
                else:
                    e1_idx, r_idx, e2_idx, targets = head_batch
                #     ========

                # print((e1_idx))
                # print((r_idx))
                # print((e2_idx))
                # print((targets.shape))

                if self.cuda:
                    e2_idx = e2_idx.cuda()
                    r_idx = r_idx.cuda()
                    e1_idx = e1_idx.cuda()


                # print(targets)
                # if self.label_smoothing:
                #     targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(0))

                # print(targets)


                self.optimizer.zero_grad()
                targets = targets.type(torch.LongTensor)
                # print(targets)

                loss = model.forward_head_and_loss(e1_idx, r_idx, e2_idx,targets)
                loss_of_epoch += loss.item()
                loss.backward()
                self.optimizer.step()
                print(loss_of_epoch)

            if self.decay_rate:
                self.scheduler.step()
            losses.append(loss_of_epoch)
            print(loss_of_epoch)
            # exit(1)
        self.logger.info('Loss at {0}.th epoch:{1}'.format(it, loss_of_epoch))
        # exit(1)
        np.savetxt(fname=self.storage_path + "/loss_per_epoch.csv", X=np.array(losses), delimiter=",")
        model.eval()
        return model


    def evaluate_one_to_one(self, model, data, log_info='Evaluate one to one.'):
        """
         Evaluate model
        """
        self.logger.info(log_info)
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])
        test_data_idxs = self.get_data_idxs(data)
        ere_vocab = self.get_ere_vocab(self.get_data_idxs(self.dataset.data))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch_1_to_1(ere_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward_head_batch(e1_idx=e1_idx, rel_idx=r_idx, e2_idx=e2_idx)
            for j in range(data_batch.shape[0]):
                filt = ere_vocab[(data_batch[j][0], data_batch[j][1],data_batch[j][2])]
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

        results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10,
                   'MR': mean_rank, 'MRR': mean_reciprocal_rank}

        return results
