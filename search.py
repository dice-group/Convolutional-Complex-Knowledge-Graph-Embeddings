from util.experiment import Experiment
from util.data import Data
import traceback
from sklearn.model_selection import ParameterGrid

datasets = ['FB15k-237', 'WN18RR', 'YAGO3-10', 'FB15k', 'WN18']
models = ['ConExNeg'] # ConEx
for kg_root in datasets:
    for model_name in models:
        data_dir = 'KGs/' + kg_root + '/'
        config = {
            'num_of_epochs': [2000],  # no tuning
            'batch_size': [1024],  # no tuning.
            'learning_rate': [.001],  # no tuning.
            'label_smoothing': [0.1],  # no tuning.
            'decay_rate': [None],  # no tuning.
            'scoring_technique': ['KvsAll'],  # no tuning.
            'train_plus_valid': [True],
            'num_workers': [32],  # depends on the machine available.
        }
        if model_name in ['ConEx']:
            config.update({'embedding_dim': [200],
                           'input_dropout': [.3],
                           'hidden_dropout': [.3],
                           'feature_map_dropout': [.4],
                           'num_of_output_channels': [16], 'kernel_size': [3]})
        elif model_name in ['ConExNeg']:
            config.update({'embedding_dim': [200],
                           'input_dropout': [.3],
                           'hidden_dropout': [.3],
                           'feature_map_dropout': [.4],
                           'num_of_output_channels': [16], 'kernel_size': [3]})
        else:
            print(model_name)
            raise ValueError

        for th, setting in enumerate(ParameterGrid(config)):
            dataset = Data(data_dir=data_dir, train_plus_valid=setting['train_plus_valid'])
            experiment = Experiment(dataset=dataset,
                                    model=model_name,
                                    parameters=setting, ith_logger='_' + str(th) + '_' + kg_root,
                                    store_emb_dataframe=False)
            try:
                experiment.train_and_eval()
            except RuntimeError as re:
                traceback.print_exc()
                print('Exit.')
                exit(1)
