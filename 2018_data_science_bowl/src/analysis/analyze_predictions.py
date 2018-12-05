
""" Take the lists of predicted and expected results and analyze predictions

Example: 
    sys.path.append( os.path.join( '.', 'src' ) ) # set path to project src/
    from analysis import analyze_predictions
    analyze = analyze_predictions.AnalyzePredictions()  
    analyze.set_expected( expected_values )
    analyze.set_predicted( predicted_values )
    analyze.print_summary_results()
    analyze.set_test_set( test_set )
    correct = analyze.get_correct_predictions()
    incorrect = analyze.get_incorrect_predictions()
    analyze.save_predictions_as_images( correct, /path/to/correct/dir )
    analyze.save_predictions_as_images( incorrect, /path/to/incorrect/dir )

Function list:
    set_expected( expected_values )
    set_predicted( predicted_values )
    set_test_set( test_values )
    print_summary_results()
    get_correct_predictions()
    get_incorrect_predictions()
    save_predictions_as_images( predictions, output_directory)

"""

import os
import pandas
import numpy
import scipy.misc
import sklearn
import sklearn.externals
import sklearn.metrics


class AnalyzePredictions( object ):
    
    def __init__( self, expected_values=None, predicted_values=None, test_values=None ):
        
        """ The class constructor.
        
        :param expected_values: List with expected values on each row
        :param predicted_values: List with predicted values on each row
        :param test_values: List with all test values (only required for some functions)
        :return: None

        """

        self.expected_values = None
        self.predicted_values = None
        self.test_values = None

        self.set_expected( expected_values )
        self.set_predicted( predicted_values )
        self.set_test_set( test_values )
        
        return


    def set_expected( self, expected_values ):

        """ Set the expected values
        
        :param expected_values: List with expected values on each row
        :return: None

        """

        self.expected_values = pandas.DataFrame( expected_values )
        #if not self.predicted_values is None:
        #    if not len( self.expected_values ) == len( self.predicted_values ):
        #        raise Exception( 'Expected values and predicted values must be of same length' )

        return


    def set_predicted( self, predicted_values ):

        """ Set the predicted values
        
        :param predicted_values: DataFrame with predicted values on each row
        :return: None

        """

        self.predicted_values = pandas.DataFrame( predicted_values )
        #if not self.expected_values is None:
        #    if not len( self.expected_values ) == len( self.predicted_values ):
        #        raise Exception( 'Expected values and predicted values must be of same length' )

        return


    def set_test_set( self, test_values ):

        """ Set the test values
        
        :param test_values: List with test values on each row (typically contains various information per row)
        :return: None

        """

        self.test_values = pandas.DataFrame( test_values )
        #if not self.expected_values is None and not self.predicted_values is None:
        #    if not len( self.test_values ) == len( self.expected_values ) or not len( self.test_values ) == len( self.predicted_values ):
        #        raise Exception( 'Number of test values must equal to the number of expected and the number of predicted values' )

        return


    def print_summary_results( self ):

        """ Print out the classification report and confusion matrix
        
        :return: None

        """

        if self.expected_values is None or self.predicted_values is None:
            raise Exception( 'The expected and predicted values must be set before summarizing reports' )

        print()
        print( '############################' )
        print( '# CLASSIFICATION REPORT' )
        print( '#' )
        print()
        print( sklearn.metrics.classification_report( self.expected_values, self.predicted_values ) )
        print()
        print( '############################' )
        print( '# CONFUSION MATRIX' )
        print( '#' )
        print()
        print( sklearn.metrics.confusion_matrix( self.expected_values, self.predicted_values ) )
        print()

        return


    def get_correct_predictions( self ):

        """ Get the test values where the prediction is correct
        
        :return: List containing the information extracted from the
            test_values that are correct predictions

        """

        if self.expected_values is None or self.predicted_values is None or self.test_values is None:
            raise Exception( 'The expected, predicted, and test values must be set before getting results' )

        correct_predictions = self.expected_values == self.predicted_values
        return self.test_values[ correct_predictions.values ]


    def get_incorrect_predictions( self ):

        """ Get the test values where the prediction is incorrect
        
        :return: List containing the information extracted from the
            test_values that are incorrect predictions

        """

        if self.expected_values is None or self.predicted_values is None or self.test_values is None:
            raise Exception( 'The expected, predicted, and test values must be set before getting results' )

        incorrect_predictions = self.expected_values != self.predicted_values
        return self.test_values[ incorrect_predictions.values ]
    

    def save_predictions_as_images( self, predictions, output_directory ):

        """ Save the input predictions into the specified directory
        
        :param predictions: Input dataframe of predictions (e.g.
            incorrect, correct)
        :param output_directory: Output directory where all images will
            be saved
        :return: List containing the information extracted from the
            test_values that are correct predictions

        """
        
        if not os.path.exists( output_directory ):
            os.makedirs( output_directory )

        for index, row in predictions.iterrows():
            output_filename = os.path.splitext( os.path.basename( row.filename ) )[0] + '.png'
            scipy.misc.imsave( os.path.join( output_directory, output_filename), row.image_matrix )

            # Uncomment if masks should be saved too
            #if output_filename[ 0:4 ] != 'neg_':
            #    mask_file = 'mask_' + os.path.splitext( os.path.basename( output_filename[4:] ) )[0]
            #    mask_matrix = numpy.load( os.path.join( '..', 'data', 'roi', 'test', mask_file + '.npy' ) )
            #    scipy.misc.imsave( os.path.join( output_directory, mask_file + '.png' ), mask_matrix )

        return


if __name__ == '__main__':
    pass

