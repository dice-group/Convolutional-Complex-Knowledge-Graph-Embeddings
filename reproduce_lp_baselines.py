from util.helper_classes import Reproduce
from models.ensemble import Ensemble

run_FB15K_237 = True
run_WN18RR = True
run_YAGO3_10 = True

fb15k237_path = 'KGs/FB15k-237'
wn18rr_path = 'KGs/WN18RR'
yago_3_10 = 'KGs/YAGO3-10'

# Set lp_per_relation_results=True, if you wish to obtain link prediction per relation.
lp_per_relation_results = False

if run_WN18RR:
    print('###########################################{0}##################################################'.format(
        wn18rr_path))
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/Tucker', data_path="%s/" % wn18rr_path,
                          model_name='Tucker', per_rel_flag_=lp_per_relation_results)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/Distmult', data_path="%s/" % wn18rr_path,
                          model_name='Distmult', per_rel_flag_=lp_per_relation_results)
    Reproduce().reproduce(model_path='PretrainedModels/WN18RR/Complex', data_path="%s/" % wn18rr_path,
                          model_name='Complex', per_rel_flag_=lp_per_relation_results)

if run_FB15K_237:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/Distmult', data_path="%s/" % fb15k237_path,
                          model_name='Distmult', per_rel_flag_=lp_per_relation_results)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/Complex', data_path="%s/" % fb15k237_path,
                          model_name='Complex', per_rel_flag_=lp_per_relation_results)
    Reproduce().reproduce(model_path='PretrainedModels/FB15K-237/Tucker', data_path="%s/" % fb15k237_path,
                          model_name='Tucker', per_rel_flag_=lp_per_relation_results)
    print('###########################################{0}##################################################'.format(
        fb15k237_path))

if run_YAGO3_10:
    print('###########################################{0}##################################################'.format(
        yago_3_10))

    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/Complex', data_path="%s/" % yago_3_10,
                          model_name='Complex', per_rel_flag_=lp_per_relation_results)

    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/Distmult', data_path="%s/" % yago_3_10,
                          model_name='Distmult', per_rel_flag_=lp_per_relation_results)

    Reproduce().reproduce(model_path='PretrainedModels/YAGO3-10/Tucker', data_path="%s/" % yago_3_10,
                          model_name='Tucker', per_rel_flag_=lp_per_relation_results)

    print('###########################################{0}##################################################'.format(
        yago_3_10))
