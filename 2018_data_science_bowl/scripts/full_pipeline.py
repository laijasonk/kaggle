
import sys, os
import random
import ast, configparser

sys.path.insert(0, os.path.join('..'))
sys.path.insert(0, os.path.join('..','src'))
from scripts import preprocessing_split_raw
from scripts import preprocessing_extract_windows_from_split
from scripts import train_nn_classifier
from scripts import test_nn_classifier
from scripts import test_nn_classifier
from scripts import evaluate_kaggle_performance_final


if __name__ == '__main__':
    
    ########################
    # 1) Download raw data
    #

    print( 'STEP 1: DOWNLOAD RAW DATA' )

    """ Assumes the data is downloaded and unzipped in /data/raw/ """





    ########################
    # 2) Split data
    #

    print( 'STEP 2: SPLIT DATA' )

    # set paths
    data_path = os.path.join( '..', 'data' )
    split_path = os.path.join( data_path, 'split' )

    # set raw data paths
    stage1_train_image_dir = os.path.join( data_path, 'raw', 'stage1_train' )
    stage1_train_zip_file = os.path.join( data_path, 'raw', 'stage1_train.zip' )
    stage1_train_csv_file = os.path.join( data_path, 'raw', 'stage1_train_labels.csv' )

    # name of the split directories
    split_dirnames = [ 'train', 'test' ]

    # ratio/fraction to split the data
    split_ratios = [ 0.7, 0.3 ]

    # call the script
    split_raw_main = preprocessing_split_raw.split_raw( 
            data_path=data_path, 
            split_path=split_path, 
            train_image_dir=stage1_train_image_dir, 
            split_dirnames=split_dirnames, 
            split_ratios=split_ratios )
    split_raw_main.run()





    ########################
    # 3b) Extract windows
    #

    print( 'STEP 3B: EXTRACT WINDOWS' )

    # define random seed for reproducibility
    random.seed( 1 )

    # choose paths
    split_path = os.path.join( '..', 'data', 'split' ) # 'split' or 'small_split'
    windows_path = os.path.join( '..', 'data', 'windows_20x20_center1px' ) # 'windows' or 'small_windows'

    # size of the window: [ R, C ]
    windows_size = [ 64, 64 ] 

    # number of windows per edge (e.g. 10 windows means 20x20 windows will be spliced from the parent image (i.e. 400 total)
    windows_per_edge = 20

    # pixels around the center point (e.g. 2px means all pixel from (center-2) to (center+2), which totals 5px around center at each direction)
    window_center_nuclei_distance = 0
    
    # call the script
    extract_split_windows_main = preprocessing_extract_windows_from_split.extract_split_windows( 
            split_path=split_path, 
            windows_path=windows_path, 
            windows_size=windows_size, 
            windows_per_edge=windows_per_edge, 
            window_center_nuclei_distance=window_center_nuclei_distance )
    extract_split_windows_main.run()





    ########################
    # 4) Modify config
    #
    
    print( 'STEP 4: MODIFY CONFIG' )

    """ As of 2018-Mar-29, the final cfg file is /config/nn_final.cfg """





    ########################
    # 5b) Train neural network
    #

    print( 'STEP 5B: TRAIN NEURAL NETWORK' )

    config = configparser.ConfigParser( allow_no_value=True )
    config.read( '../config/nn_final.cfg' )

    # no need to modify since it reads the config file
    train_classifier_main = train_nn_classifier.train_classifier(
            df_path=ast.literal_eval( config.get( 'paths', 'train_df_path' ) ),
            image_col_name=ast.literal_eval( config.get( 'columns', 'image_col_name' ) ),
            category_col_name=ast.literal_eval( config.get( 'columns', 'category_col_name' ) ),
            fixed_size=ast.literal_eval( config.get( 'specifications', 'fixed_size' ) ),
            classifier_pkl=ast.literal_eval( config.get( 'paths', 'classifier_pkl' ) ),
            feature_method=ast.literal_eval( config.get( 'features', 'method' ) ), 
            feature_options=ast.literal_eval( config.get( 'features', 'feature_options' ) ),
            image_processing_options=ast.literal_eval( config.get( 'features', 'image_processing_options' ) ),
            raw_image_path=ast.literal_eval( config.get( 'paths', 'raw_image_path' ) ),
            solver=ast.literal_eval( config.get( 'mlp', 'solver' ) ),
            alpha=ast.literal_eval( config.get( 'mlp', 'alpha' ) ),
            tol=ast.literal_eval( config.get( 'mlp', 'tol' ) ),
            max_iter=ast.literal_eval( config.get( 'mlp', 'max_iter' ) ),
            hidden_layer_sizes=ast.literal_eval( config.get( 'mlp', 'hidden_layer_sizes' ) ),
            random_state=ast.literal_eval( config.get( 'mlp', 'random_state' ) ) )
    train_classifier_main.run()





    ########################
    # 6b) Test neural network
    #

    print( 'STEP 6B: TEST NEURAL NETWORK' )

    config = configparser.ConfigParser( allow_no_value=True )
    config.read( '../config/nn_final.cfg' )

    # no need to modify since it reads the config file
    test_classifier_main = test_nn_classifier.test_classifier( 
            df_path=ast.literal_eval( config.get( 'paths', 'test_df_path' ) ),
            image_col_name=ast.literal_eval( config.get( 'columns', 'image_col_name' ) ),
            category_col_name=ast.literal_eval( config.get( 'columns', 'category_col_name' ) ),
            fixed_size=ast.literal_eval( config.get( 'specifications', 'fixed_size' ) ),
            classifier_pkl=ast.literal_eval( config.get( 'paths', 'classifier_pkl' ) ),
            feature_method=ast.literal_eval( config.get( 'features', 'method' ) ), 
            feature_options=ast.literal_eval( config.get( 'features', 'feature_options' ) ),
            image_processing_options=ast.literal_eval( config.get( 'features', 'image_processing_options' ) ),
            raw_image_path=ast.literal_eval( config.get( 'paths', 'raw_image_path' ) ),
            prediction_output_dir=ast.literal_eval( config.get( 'paths', 'prediction_output_dir' ) ) )
    test_classifier_main.run()

    



    ########################
    # 6) Evaluate performance
    #

    print( 'STEP 6: EVALUATE PEFORMANCE' )
    evaluate_kaggle_performance_final.main() 

