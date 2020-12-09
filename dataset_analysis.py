from util.helper_classes import Data
import numpy as np


def describe_stats(d):
    nodes = dict()
    for i in d:
        h, r, t = i
        nodes.setdefault(h, []).append(t)

    num_of_edges = []
    for k, v in nodes.items():
        num_of_edges.append(len(v))

    num_of_edges = np.array(num_of_edges)

    print('Avg. number of edges per node /relations per entity: ', num_of_edges.mean())
    print('Std. number of edges per node /relations per entity: ', num_of_edges.std())


def sets_of_ent_and_rel(l):
    entities, relations, = set(), set()
    for t in l:
        head, rel, tail = t
        entities.add(head)
        entities.add(tail)
        relations.add(rel)

    return entities, relations


def describe(dataset_name, dataset):
    print('########## Dataset:{0} #########'.format(dataset_name))

    # (1) Set of entities and relations
    e_train, rel_train = sets_of_ent_and_rel(dataset.train_data)
    e_val, rel_val = sets_of_ent_and_rel(dataset.valid_data)
    e_test, rel_test = sets_of_ent_and_rel(dataset.test_data)

    print('# of unique entities on (train + valid + test) = {0}'.format(len(e_train.union(e_val).union(e_test))))
    print('# of unique relations on (train + valid + test) = {0}'.format(len(rel_train.union(rel_val).union(rel_test))))

    print('|Train|= {0}\t|Valid|= {1}\t|Test|= {2}'.format(len(dataset.train_data), len(dataset.valid_data),
                                                           len(dataset.test_data)))

    # (3) Set diff.
    e_out_of_vocab_valid = e_val.difference(e_train)
    e_out_of_vocab_test = e_test.difference(e_train)
    print('# of entities on Valid not occurring on Train={0}'.format(len(e_out_of_vocab_valid)))
    print('# of entities on Test not occurring on Train={0}'.format(len(e_out_of_vocab_test)))

    rel_out_of_vocab_valid = rel_val.difference(rel_train)
    rel_out_of_vocab_test = rel_test.difference(rel_train)
    print('# of relations on Valid not occurring on Train={0}'.format(len(rel_out_of_vocab_valid)))
    print('# of relations on Test not occurring on Train={0}'.format(len(rel_out_of_vocab_test)))

    # (4)
    triples_valid_out_of_vocab = [t for t in dataset.valid_data if
                                  t[0] in e_out_of_vocab_valid or t[2] in e_out_of_vocab_valid]
    triples_test_out_of_vocab = [t for t in dataset.test_data if
                                 t[0] in e_out_of_vocab_test or t[2] in e_out_of_vocab_test]

    print('# of triples on Valid containing out of vocab entities={0} \t percentage:{1}%'.format(
        len(triples_valid_out_of_vocab), round((len(triples_valid_out_of_vocab) / len(dataset.valid_data)) * 100, 3)))
    print('# of triples on Test containing out of vocab entities={0} \t percentage:{1}%'.format(
        len(triples_test_out_of_vocab), round((len(triples_test_out_of_vocab) / len(dataset.test_data)) * 100, 3)))

    describe_stats(dataset.train_data)
    print('###################\n'.format(dataset_name))


for d in ['WN18', 'FB15k', 'WN18RR', 'FB15k-237', 'YAGO3-10']:
    data = Data(data_dir='KGs/' + d + '/', reverse=False)
    describe(d, data)
