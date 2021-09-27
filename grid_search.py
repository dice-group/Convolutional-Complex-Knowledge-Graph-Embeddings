from util.experiment import Experiment
from util.data import Data
import traceback
from sklearn.model_selection import ParameterGrid

torch.backends.cudnn.deterministic = True
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

datasets = ['FB15k-237', 'WN18RR', 'YAGO3-10', 'FB15k', 'WN18']
models = ['ConEx']

num_runs = 2

for kg_root in datasets:
    for model_name in models:
        data_dir = 'KGs/' + kg_root + '/'
        config = {
            'num_of_epochs': [2000],
            'batch_size': [1024],
            'optim': ['Adam', 'RMSprop'],
            'learning_rate': [.001],
            'label_smoothing': [0.1],
            'decay_rate': [None],
            'scoring_technique': ['KvsAll'],
            'train_plus_valid': [False, True],
            'num_workers': [32],  # depends on the machine available.
        }
        if model_name in ['ConEx']:
            config.update({'embedding_dim': [100, 200],
                           'input_dropout': [.3, .4],
                           'hidden_dropout': [0],
                           'feature_map_dropout': [.3, .4],
                           'num_of_output_channels': [16, 32], 'kernel_size': [3]})
        else:
            print(model_name)
            raise ValueError

        for th, setting in enumerate(ParameterGrid(config)):
            for r in range(num_runs):
                dataset = Data(data_dir=data_dir, train_plus_valid=setting['train_plus_valid'])
                experiment = Experiment(dataset=dataset,
                                        model=model_name,
                                        parameters=setting, ith_logger='_' + str(th) + '_' + '_' + str(r) + '_' + kg_root,
                                        store_emb_dataframe=False)
                try:
                    experiment.train_and_eval()
                except RuntimeError as re:
                    traceback.print_exc()
                    print('Exit.')
                    continue
