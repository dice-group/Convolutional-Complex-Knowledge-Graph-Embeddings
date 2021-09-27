from util.helper_classes import Reproduce
from models.ensemble import Ensemble
import numpy as np
from util.helper_funcs import compute_confidence_interval

fb15k237_path = 'KGs/FB15k-237'

no_input_dropout = True

no_label_smoothing = True
no_dropout = True
use_rmsprop = True

lp_per_relation_results = False

if no_input_dropout:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    print('NO input')
    report1 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/NoDropout/1', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)
    report2 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/NoDropout/2', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)

    hit10 = np.array([report1['Hits@10'], report2['Hits@10']])
    hit3 = np.array([report1['Hits@3'], report2['Hits@3']])
    hit1 = np.array([report1['Hits@1'], report2['Hits@1']])
    mrr = np.array([report1['MRR'], report2['MRR']])
    results = {'MRR': mrr,
               'Hits@1': hit1,
               'Hits@3': hit3,
               'Hits@10': hit10}
    compute_confidence_interval(results)

    print('###########################################{0}##################################################'.format(
        fb15k237_path))
if no_label_smoothing:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    report1 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/NpLS/1', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)
    report2 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/NpLS/2', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)

    hit10 = np.array([report1['Hits@10'], report2['Hits@10']])
    hit3 = np.array([report1['Hits@3'], report2['Hits@3']])
    hit1 = np.array([report1['Hits@1'], report2['Hits@1']])
    mrr = np.array([report1['MRR'], report2['MRR']])
    results = {'MRR': mrr,
               'Hits@1': hit1,
               'Hits@3': hit3,
               'Hits@10': hit10}
    compute_confidence_interval(results)

    print('###########################################{0}##################################################'.format(
        fb15k237_path))
if no_dropout:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    report1 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/NoDropout/1', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)
    report2 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/NoDropout/2', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)

    hit10 = np.array([report1['Hits@10'], report2['Hits@10']])
    hit3 = np.array([report1['Hits@3'], report2['Hits@3']])
    hit1 = np.array([report1['Hits@1'], report2['Hits@1']])
    mrr = np.array([report1['MRR'], report2['MRR']])
    results = {'MRR': mrr,
               'Hits@1': hit1,
               'Hits@3': hit3,
               'Hits@10': hit10}
    compute_confidence_interval(results)

    print('###########################################{0}##################################################'.format(
        fb15k237_path))
if use_rmsprop:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    report1 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/Optim/1', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)
    report2 = Reproduce().reproduce(model_path='PretrainedModels/Ablation/Optim/2', data_path="%s/" % fb15k237_path,
                                    model_name='ConEx', per_rel_flag_=lp_per_relation_results)

    hit10 = np.array([report1['Hits@10'], report2['Hits@10']])
    hit3 = np.array([report1['Hits@3'], report2['Hits@3']])
    hit1 = np.array([report1['Hits@1'], report2['Hits@1']])
    mrr = np.array([report1['MRR'], report2['MRR']])
    results = {'MRR': mrr,
               'Hits@1': hit1,
               'Hits@3': hit3,
               'Hits@10': hit10}
    compute_confidence_interval(results)

    print('###########################################{0}##################################################'.format(
        fb15k237_path))
