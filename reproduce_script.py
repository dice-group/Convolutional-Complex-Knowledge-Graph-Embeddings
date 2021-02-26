from util.helper_classes import Reproduce
import argparse

def start(args):
    Reproduce().reproduce(model_path=args.path_model_folder, data_path=args.path_dataset_folder+'/',
                          model_name=args.model_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset_folder", type=str, default='KGs/WN18RR')
    parser.add_argument("--path_model_folder", type=str, default='PretrainedModels/WN18RR/ConEx')
    parser.add_argument("--model_name", type=str, default='ConEx')
    start(parser.parse_args())
