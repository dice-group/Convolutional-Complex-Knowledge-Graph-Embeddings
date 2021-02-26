from util.helper_classes import Reproduce
from models.ensemble import Ensemble

run_FB15K_237 = True
run_WN18RR = True
fb15k237_path = 'KGs/FB15k-237'
fb15k_path = 'KGs/FB15k'
wn18rr_path = 'KGs/WN18RR'

if run_WN18RR:
    print('###########################################{0}##################################################'.format(
        wn18rr_path))
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/Distmult', model_name='Distmult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/Complex', model_name='Complex')),
        data_path="%s/" % wn18rr_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/Distmult', model_name='Distmult'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/Tucker', model_name='Tucker')),
        data_path="%s/" % wn18rr_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConEx', model_name='ConEx'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/Distmult', model_name='Distmult')),
        data_path="%s/" % wn18rr_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConEx', model_name='ConEx'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/Complex', model_name='Complex')),
        data_path="%s/" % wn18rr_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/WN18RR/ConEx', model_name='ConEx'),
                       Reproduce().load_model(model_path='PretrainedModels/WN18RR/Tucker', model_name='Tucker')),
        data_path="%s/" % wn18rr_path)

if run_FB15K_237:
    print('###########################################{0}##################################################'.format(
        fb15k237_path))
    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/Distmult', model_name='Distmult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/Complex', model_name='Complex')),
        data_path="%s/" % fb15k237_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/Distmult', model_name='Distmult'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/Tucker', model_name='Tucker')),
        data_path="%s/" % fb15k237_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConExStatistics/16Channel/ConEx', model_name='ConEx'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/Distmult', model_name='Distmult')),
        data_path="%s/" % fb15k237_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConExStatistics/16Channel/ConEx', model_name='ConEx'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/Complex', model_name='Complex')),
        data_path="%s/" % fb15k237_path)

    Reproduce().reproduce_ensemble(
        model=Ensemble(Reproduce().load_model(model_path='PretrainedModels/FB15K-237/ConExStatistics/16Channel/ConEx', model_name='ConEx'),
                       Reproduce().load_model(model_path='PretrainedModels/FB15K-237/Tucker', model_name='Tucker')),
        data_path="%s/" % fb15k237_path)

    print('###########################################{0}##################################################'.format(
        fb15k237_path))
