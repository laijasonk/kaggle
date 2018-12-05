
""" Train the neural network classifier

"""

import numpy
import pandas
import scipy
import sklearn
import sklearn.neural_network
import sklearn.externals
import sklearn.metrics
import ast, configparser

import sys,os
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from preprocess import split_data
from utils import kaggle_reader
from analysis import feature_extractions, analyze_predictions, tuning
from preprocess import image_processing


class train_classifier( object ):

    def __init__( self, df_path, image_col_name, category_col_name, fixed_size, classifier_pkl,
                  feature_method, feature_options, image_processing_options, raw_image_path,
                  solver='lbfgs', alpha=0.0001, tol=0.00001, max_iter=200, hidden_layer_sizes=(100,), random_state=1 ):

        """ The class constructor
        :param df_path: full path to the input roi DataFrame file (e.g. ../data/roi/roidf.h5)
        :param image_col_name: column name for the image data column
        :param category_col_name: column name for the image category (negative or positive) column
        :param fixed_size: 2x1 matrix containing the height and width of the images, format=[ height, width ]
        :param classifier_pkl: Path to the output classifier pickle file
        :param solver: Neural network solver (e.g. lbfgs, sgd, adam)
        :param alpha: L2 penalty, aka regularization (e.g. 0.0001, 0.00001, etc.)
        :param hidden_layer_sizes: number of neurons in the ith hidden layer, e.g. (100,) or () or ()
        :param random_state: Basically the random seed (e.g. 1)
        :return: None
        """

        # parameters that require input
        self.fixed_size = fixed_size
        self.classifier_pkl = classifier_pkl

        # parameters for feature extraction
        self.feature_method = feature_method
        self.feature_options = feature_options
        self.image_processing_options = image_processing_options
        self.raw_image_path = raw_image_path

        # neural network parameters
        self.solver = solver
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

        # data related parameters, needs to be read from a file
        self.data = None
        self.image_col_name = image_col_name
        self.category_col_name = category_col_name

        print( "Loading dataframe: " + df_path )
        self.data = self._load_data_from_dataframe( df_path )

        return


    def _load_data_from_dataframe(self, df_path):

        """ Loads data from a saved panda DataFrame file

        :param df_path: file path to the saved DataFrame file
        :return: none

        """

        try:
            df = pandas.read_pickle( df_path )
            # filter out the ROIs with a size not equal to the input fixed_size
            df = df[ df[ self.image_col_name ].apply( numpy.shape ) == self.fixed_size ]
        except IOError:
            print( 'Please check the training data file path and make sure the file is in pickle format.' )
            print( 'tried: ' + df_path )

        return df


    def run( self ):

        """ Run the main code to train classifier and save to file

        :return: None

        """

        # process images
        if self.image_processing_options:
            print( "Preprocessing images" )
            image_df = kaggle_reader.load_all_raw_images( self.raw_image_path )
            processed_data = image_processing.process_roi_extractions( image_df, self.data,
                                                                       self.image_processing_options )
        else:
            processed_data = self.data
        # extract features from dataframes
        print( "Extracting features" )
        features = feature_extractions.feature_extraction( processed_data,
                                                           input_parameters=self.image_col_name,
                                                           method=self.feature_method,
                                                           extraction_options=self.feature_options )

        # create target array which contains the correct answer
        target = pandas.DataFrame( numpy.repeat( [ True ], len( features ) ) )
        target[ self.data[ self.category_col_name ].values == '0' ] = False

        # initialize classifier
        print( "Initializing classifier" )
        classifier = sklearn.neural_network.MLPClassifier( solver=self.solver, alpha=self.alpha, tol=self.tol, max_iter=self.max_iter, hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state, verbose=True )
        print( classifier )
        print()

        # train the classifier
        print( "Training classifier" )
        classifier.fit( features, target.values.ravel() )
        print()

        # save the classifier to be loaded later
        print( "Saving classifier: " + str( self.classifier_pkl ) )
        sklearn.externals.joblib.dump( classifier, self.classifier_pkl )

        return

if __name__ == '__main__':

    config = configparser.ConfigParser( allow_no_value=True )
    #config.read( '../config/nn_adam_20_20_20_20_20_center3px_10x10win_graytrimnormsobel.cfg' )
    config.read( '../config/nn_test.cfg' )

    main = train_classifier(
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

    if ast.literal_eval( config.get( 'tune', 'tune_parameters' ) ) == True:
        model = sklearn.neural_network.MLPClassifier()
        main, clf = tuning.tune( 
                main=main, 
                model=model, 
                tuned_parameters=ast.literal_eval( config.get( 'tune', 'parameters' ) ), 
                fraction=ast.literal_eval( config.get( 'tune', 'fraction' ) ), 
                diversity_vars=ast.literal_eval( config.get( 'tune', 'diversity_vars' ) ), 
                bayes=ast.literal_eval( config.get( 'tune', 'bayes' ) ), 
                verbose=ast.literal_eval( config.get( 'tune', 'verbose' ) ), 
                n_jobs=ast.literal_eval( config.get( 'tune', 'n_jobs' ) ),
                random_state=ast.literal_eval( config.get( 'tune', 'random_state' ) ) )

    main.run()

