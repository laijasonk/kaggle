
""" Test the neural network classifier

"""

import numpy
import pandas
import sklearn
import sklearn.externals
import sklearn.metrics
import ast, configparser
    
import sys,os
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from utils import kaggle_reader
from analysis import analyze_predictions, feature_extractions
from preprocess import image_processing


class test_classifier( object ):

    def __init__( self, df_path, image_col_name, category_col_name, fixed_size, classifier_pkl, 
                  feature_method, feature_options, image_processing_options, raw_image_path,
                  prediction_output_dir=None ):

        """ The class constructor

        :param df_path: Path to the directories containing test images
        :param classifier_pkl: Path to the trained pickled classifer
        :return: None

        """

        self.classifier_pkl = classifier_pkl
        self.image_col_name = image_col_name
        self.category_col_name = category_col_name
        self.fixed_size = fixed_size
        self.prediction_output_dir = prediction_output_dir

        # feature extraction properties
        self.feature_method = feature_method
        self.feature_options = feature_options
        self.image_processing_options = image_processing_options
        self.raw_image_path = raw_image_path

        print( "Loading dataframe: " + df_path )
        self.data = self._load_data_from_dataframe(df_path)
        
        return


    def run( self ):
 
        """ Run the main code to test classifier and output results

        :return: None

        """

        # loading pre-trained classifier
        print( "Loading external classifier: " + self.classifier_pkl )
        try:
            external_classifier = sklearn.externals.joblib.load( self.classifier_pkl )
        except OSError:
            print('Please check the file location for the trained classifier.')
            print('Tried: '+self.classifier_pkl)
            return

        # process images
        if self.image_processing_options:
            print('Preprocessing images')
            image_df = kaggle_reader.load_all_raw_images(self.raw_image_path)
            processed_data = image_processing.process_roi_extractions(image_df, self.data,
                                                                      self.image_processing_options)
        else:
            processed_data = self.data
        # extract features from dataframes
        print( 'Extracting features' )
        # features = pandas.DataFrame( processed_data.image_matrix.apply( numpy.ravel ).apply( pandas.Series ) )
        features = feature_extractions.feature_extraction(processed_data,
                                                          input_parameters=self.image_col_name,
                                                          method=self.feature_method,
                                                          extraction_options=self.feature_options)

        # run prediction
        print( 'Running prediction' )
        predicted = external_classifier.predict( features )

        # create array of expected results
        print( 'Organizing expected results' )
        expected = pandas.DataFrame(numpy.repeat([True], len(features)))
        expected[self.data[self.category_col_name].values == '0'] = False

        # preparing to analyze predictions
        analyze = analyze_predictions.AnalyzePredictions()  
        analyze.set_expected( pandas.DataFrame( expected ) )
        analyze.set_predicted( pandas.DataFrame( predicted ) )
        analyze.set_test_set( self.data )

        print()
        print( "########################" )
        print( "# CLASSIFIER" )
        print( "#" )
        print()
        print( external_classifier )
        print()
        analyze.print_summary_results()

        # save results to directory
        if not self.prediction_output_dir is None:
            print( 'Saving predictions to disk' )
            correct = analyze.get_correct_predictions()
            incorrect = analyze.get_incorrect_predictions()
            analyze.save_predictions_as_images( correct, os.path.join( self.prediction_output_dir, 'correct' ) )
            analyze.save_predictions_as_images( incorrect, os.path.join( self.prediction_output_dir, 'incorrect' ) )

        return


    def _load_data_from_dataframe( self, df_path ):

        """ load the index file into a dataframe

        :param index_path: Path to the index file
        :return: Dataframe containing indices of test files plus ImageMat

        """

        try:
            df = pandas.read_pickle(df_path)
            df = df[df[self.image_col_name].apply(numpy.shape) == self.fixed_size]
        except IOError:
            print("Please check the test data file path and make sure the file is in pickle format.")
            print('tried: ' + df_path)
            df = []

        return df


if __name__ == '__main__':

    config = configparser.ConfigParser( allow_no_value=True )
    #config.read( '../config/nn_adam_20_20_20_20_20_center3px_10x10win_graytrimnormsobel.cfg' )
    config.read( '../config/nn_test.cfg' )

    main = test_classifier( 
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

    main.run()

