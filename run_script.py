from util.experiment import Experiment
from util.data import Data
import traceback
import argparse


def start(args):
    dataset = Data(data_dir=args.path_dataset_folder, train_plus_valid=args.train_plus_valid)
    experiment = Experiment(dataset=dataset,
                            model=args.model_name,
                            parameters=vars(args), ith_logger='__' + args.path_dataset_folder,
                            store_emb_dataframe=False)
    try:
        experiment.train_and_eval()
    except RuntimeError as re:
        print(re)
        traceback.print_exc()
        print('Exit.')
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ConEx')
    parser.add_argument('--num_of_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--scoring_technique', type=str, default='KvsAll')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--optim', type=str, default='RMSprop', help='Choose optimizer: Adam or RMSprop')
    parser.add_argument('--decay_rate', default=None)
    parser.add_argument('--train_plus_valid', default=True)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--input_dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dropout', type=float, default=0.0)
    parser.add_argument('--feature_map_dropout', type=float, default=0.4)
    parser.add_argument('--num_of_output_channels', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/UMLS/')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of cpus used during batching')
    start(parser.parse_args())
