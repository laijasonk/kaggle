"""Training of a small subset

Simple attempt to train a classifier using a small subset of the data
via the HOG approach. Trained classifier is saved as a pkl file.
"""

import numpy
import pandas
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.metrics
import ast, configparser
    
import sys,os
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from utils import kaggle_reader
from preprocess import image_processing
from analysis import tuning, feature_extractions


class train_classifier( object ):

    def __init__( self, df_path, image_col_name, category_col_name, fixed_size, classifier_pkl,
                  feature_method, feature_options, image_processing_options, raw_image_path,
                  kernel='rbf', gamma='auto', C=1 ):

        """ The class constructor

        :param df_path: full path to the input roi DataFrame file (e.g. ../data/roi/roidf.h5)
        :param image_col_name: column name for the image data column
        :param category_col_name: column name for the image category (negative or positive) column
        :param fixed_size: 2x1 matrix containing the height and width of the images, format=[ height, width ]
        :param classifier_pkl: Path to the output classifier pickle file
        :param kernel: SVM kernel (e.g. rbf, linear, poly, sigmoid)
        :param gamma: kernel coefficient (e.g. 'auto', 0.001, etc.)
        :param C: penalty parameter (e.g. 1.0, 100)
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

        # parameters that has default values, also SVM parameters
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

        # data related parameters, needs to be read from a file
        self.data = None
        self.image_col_name = image_col_name
        self.category_col_name = category_col_name
        
        print( "Loading dataframe: " + df_path )
        self._load_data_from_dataframe_file( df_path )

        return


    def _load_data_from_dataframe_file(self, df_path):

        """ Loads data from a saved panda DataFrame file
        :param df_path: file path to the saved DataFrame file
        :return: none

        """

        try:
            df = pandas.read_pickle(df_path)
            # filter out the ROIs with a size not equal to the input fixed_size
            self.data = df[df[self.image_col_name].apply(numpy.shape) == self.fixed_size]
        except IOError:
            print('Please check the training data file path and make sure the file is in pickle format.')
            print('tried: '+df_path)
        return


    def run( self ):
 
        """ Run the main code to train classifier and save to file

        :return: None

        """

        # process images
        if self.image_processing_options:
            print('Preprocessing images')
            image_df = kaggle_reader.load_all_raw_images(self.raw_image_path)
            processed_data = image_processing.process_roi_extractions(image_df, self.data,
                                                                      self.image_processing_options)
        else:
            processed_data = self.data
        # extract features from dataframes
        print('Extracting features')
        features = feature_extractions.feature_extraction(processed_data,
                                                          input_parameters=self.image_col_name,
                                                          method=self.feature_method,
                                                          extraction_options=self.feature_options)

        # create target array which contains the correct answer
        target = pandas.DataFrame( numpy.repeat( [ True ], len( features ) ) )
        target[ self.data[ self.category_col_name ].values == '0' ] = False

        # initialize classifier
        print( "Initializing classifier" )
        classifier = sklearn.svm.SVC( kernel=self.kernel, gamma=self.gamma, C=self.C, verbose=True )

        # train the classifier
        print( "Training classifier" )
        classifier.fit( features, target.values.ravel() )
        print()

        # save the classifier to be loaded later
        print( "Saving classifier: " + str( self.classifier_pkl ) )
        sklearn.externals.joblib.dump( classifier, self.classifier_pkl )


if __name__ == '__main__':

    config = configparser.ConfigParser( allow_no_value=True )
    config.read( '../config/svm_gkhp_tuning_center3px_10x10win_graytrimnorm.cfg' )
        
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
            kernel=ast.literal_eval( config.get( 'svm', 'kernel' ) ),
            gamma=ast.literal_eval( config.get( 'svm', 'gamma' ) ), 
            C=ast.literal_eval( config.get( 'svm', 'C' ) ) )

    if ast.literal_eval( config.get( 'tune', 'tune_parameters' ) ):
        model = sklearn.svm.SVC(probability=True)
        tuned_parameters = ast.literal_eval(config.get('tune', 'parameters'))
        main, clf = tuning.tune( 
                main=main, 
                model=model, 
                tuned_parameters=ast.literal_eval(config.get('tune', 'parameters')),
                fraction=ast.literal_eval(config.get('tune', 'fraction')),
                diversity_vars=ast.literal_eval(config.get('tune', 'diversity_vars')),
                bayes=ast.literal_eval(config.get('tune', 'bayes')),
                verbose=ast.literal_eval(config.get('tune', 'verbose')),
                n_jobs=ast.literal_eval(config.get('tune', 'n_jobs')),
                random_state=ast.literal_eval(config.get('tune', 'random_state')) )

    main.run()

