from util.helper_classes import Reproduce
from models.ensemble import Ensemble

run_FB15K_237 = True
run_WN18RR = True
run_YAGO3_10 = True

fb15k237_path = 'KGs/FB15k-237'
yago3_10_path = 'KGs/YAGO3-10'
wn18rr_path = 'KGs/WN18RR'

if run_YAGO3_10:
    print('###########################################{0}##################################################'.format(
        yago3_10_path))
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/ConEx', data_path="%s/" % yago3_10_path,
                          model_name='ConEx',
                          out_of_vocab_flag=True)

if run_FB15K_237:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConEx', data_path="%s/" % fb15k237_path,
                          model_name='ConEx', out_of_vocab_flag=True)
    print('###########################################{0}##################################################'.format(
        fb15k237_path))

if run_WN18RR:
    print('###########################################{0}##################################################'.format(
        wn18rr_path))
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConEx', data_path="%s/" % wn18rr_path, model_name='ConEx',
                          out_of_vocab_flag=True)
