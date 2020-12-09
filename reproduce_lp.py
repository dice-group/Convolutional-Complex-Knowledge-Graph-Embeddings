from util.helper_classes import Reproduce
from models.ensemble import Ensemble

run_WN18 = True
run_FB15K_237 = True
run_YAGO3_10 = False
run_FB15K = True
run_WN18RR = True

wn18_path = 'KGs/WN18'
fb15k237_path = 'KGs/FB15k-237'
yago3_10_path = 'KGs/YAGO3-10'
fb15k_path = 'KGs/FB15k'
wn18rr_path = 'KGs/WN18RR'

if run_WN18:
    print('###########################################{0}##################################################'.format(
        wn18_path))
    Reproduce().reproduce(model_path='PretrainedModels/WN18/ConEx', data_path="%s/" % wn18_path, model_name='ConEx')

if run_FB15K:
    print('###########################################{0}##################################################'.format(
        fb15k_path))
    Reproduce().reproduce(model_path='PretrainedModels/FB15K/ConEx', data_path="%s/" % fb15k_path, model_name='ConEx')

    print('###########################################{0}##################################################'.format(
        fb15k_path))

if run_FB15K_237:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/ConEx', data_path="%s/" % fb15k237_path,
                          model_name='ConEx')
    print('###########################################{0}##################################################'.format(
        fb15k237_path))

if run_YAGO3_10:  # RAM must be greater than 16GPU
    print('###########################################{0}##################################################'.format(
        yago3_10_path))
    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/ConEx', data_path="%s/" % yago3_10_path,
                          model_name='ConEx')
    print('###########################################{0}##################################################'.format(
        yago3_10_path))

if run_WN18RR:
    print('###########################################{0}##################################################'.format(
        wn18rr_path))
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/ConEx', data_path="%s/" % wn18rr_path,model_name='ConEx')
