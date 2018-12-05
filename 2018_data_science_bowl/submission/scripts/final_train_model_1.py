
import os
import random
import ast, configparser

import final_functions


if __name__ == '__main__':

    ########################
    # 1) Download raw data
    #

    print( 'STEP 1: DOWNLOAD RAW DATA' )

    """ Assumes the data is downloaded and unzipped in /data/raw/train """

    ########################
    # 2) Extract windows
    #

    print( 'STEP 2: EXTRACT WINDOWS' )

    # define random seed for reproducibility
    random.seed( 1 )

    # choose paths
    split_path = os.path.join( '..', 'data', 'raw', ) # 'split' or 'small_split'
    windows_path = os.path.join( '..', 'data', 'windows_20x20_center1px' ) # 'windows' or 'small_windows'

    # size of the window: [ R, C ]
    windows_size = [ 64, 64 ]

    # number of windows per edge (e.g. 10 windows means 20x20 windows will be spliced from the parent image (i.e. 400 total)
    windows_per_edge = 20

    # pixels around the center point (e.g. 2px means all pixel from (center-2) to (center+2), which totals 5px around center at each direction)
    window_center_nuclei_distance = 0

    # call the script
    extract_split_windows_main = final_functions.extract_split_windows(
            split_path=split_path,
            windows_path=windows_path,
            windows_size=windows_size,
            windows_per_edge=windows_per_edge,
            window_center_nuclei_distance=window_center_nuclei_distance )
    extract_split_windows_main.run( [ "stage1_train" ] )


    ########################
    # 3) Train neural network
    #

    print( 'STEP 3: TRAIN NEURAL NETWORK' )

    config = configparser.ConfigParser( allow_no_value=True )
    config.read( '../config/submission_1.cfg' )

    # no need to modify since it reads the config file
    train_classifier_main = final_functions.train_classifier(
            df_path=ast.literal_eval( config.get( 'paths', 'train_df_path' ) ),
            image_col_name=ast.literal_eval( config.get( 'columns', 'image_col_name' ) ),
            category_col_name=ast.literal_eval( config.get( 'columns', 'category_col_name' ) ),
            fixed_size=ast.literal_eval( config.get( 'specifications', 'fixed_size' ) ),
            classifier_pkl=ast.literal_eval( config.get( 'paths', 'classifier_pkl' ) ),
            feature_method=ast.literal_eval( config.get( 'features', 'method' ) ),
            image_processing_options=ast.literal_eval( config.get( 'features', 'image_processing_options' ) ),
            raw_image_path=ast.literal_eval( config.get( 'paths', 'raw_image_path' ) ),
            solver=ast.literal_eval( config.get( 'mlp', 'solver' ) ),
            alpha=ast.literal_eval( config.get( 'mlp', 'alpha' ) ),
            tol=ast.literal_eval( config.get( 'mlp', 'tol' ) ),
            max_iter=ast.literal_eval( config.get( 'mlp', 'max_iter' ) ),
            hidden_layer_sizes=ast.literal_eval( config.get( 'mlp', 'hidden_layer_sizes' ) ),
            random_state=ast.literal_eval( config.get( 'mlp', 'random_state' ) ) )
    train_classifier_main.run()

