import pandas
import sklearn
import numpy
#import skopt

import sys,os
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from preprocess import split_data, image_processing
from analysis import feature_extractions
from utils import kaggle_reader


def tune(main, model, tuned_parameters, feature_method="pixelval", fraction=1, diversity_vars=None, iterations=50, bayes=False, verbose=0, n_jobs=1, random_state=1):
    """

    :param main: Classifier object
    :param model: Classifier
    :param tuned_parameters: Grid with hyperparameters
    :param feature_method: string describing method to extract features
    :param fraction: Fraction of data used to tune hyperparameters
    :param diversity_vars: Variable names of which diversity should be maintained
    :param iterations: Amount of iteration for Bayesian search
    :param bayes: boolean indicating to use Bayesian search (True) or normal search (False)
    :param verbose: How much to output (e.g. 0, 10, 50)
    :param n_jobs: Number of jobs to run in parallel (e.g. 1, 2, 4, 8)
    :param random_state: Basically the random seed (e.g. 1)
    :return: Classifier object and search object
    """

    data = split_data.slice_smaller_subset_of_data(main.data, fraction=fraction, diversity_vars=diversity_vars,
                                                   random_state=random_state)

    # preprocessing images
    if main.image_processing_options:
        print('Preprocessing images')
        image_df = kaggle_reader.load_all_raw_images(main.raw_image_path)
        processed_data = image_processing.process_roi_extractions(image_df, data,
                                                                  main.image_processing_options)
    else:
        processed_data = data
    # extract features from dataframes
    print('Extracting features using: '+main.feature_method)
    features = pandas.DataFrame( feature_extractions.feature_extraction(processed_data,
                                                                        input_parameters=main.image_col_name,
                                                                        method=main.feature_method,
                                                                        extraction_options=main.feature_options))

    # create target array which contains the correct answer
    target = pandas.DataFrame(numpy.repeat([True], len(features)))
    target[data[main.category_col_name].values == '0'] = False

    # take the sliced data to further split into a training and x-validation (for hyper-parameter tuning) sets
    print('Preparing training and cross-validation sets')
    train_indices = split_data.split_data_stratified(data = data, fraction = 0.8, diversity_vars = diversity_vars, random_state = random_state)
    train_indices = features.index.isin(features.index[train_indices])
    train_features = features[train_indices]
    train_target = target[train_indices].values.ravel()
    test_features = features[train_indices == False]
    test_target = target[train_indices == False].values.ravel()

    inner_cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=1)

    print('Begin training and tuning')
    #if bayes:
    #    opt = skopt.BayesSearchCV(model, tuned_parameters, n_iter=iterations, cv=inner_cv, verbose=True)
    #else:
    opt = sklearn.model_selection.GridSearchCV(model, tuned_parameters, cv=inner_cv, scoring= "neg_log_loss", verbose=verbose, n_jobs=n_jobs)
    opt.fit(train_features, train_target)
    print(opt.best_params_)
    print("Train score: %s" % opt.best_score_)
    print("Test score: %s" % opt.score(test_features, test_target))

    for name, value in opt.best_params_.items():
        setattr(main, name, value)
    return main, opt
