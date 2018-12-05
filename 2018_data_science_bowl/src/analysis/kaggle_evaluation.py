
""" Evaluate the performance according to Kaggle's metrics

Example: 
    sys.path.append( os.path.join( '.', 'src' ) ) # set path to project src/
    from analysis import kaggle_evaluation
    evaluation = kaggle_evaluation.KaggleEvaluation()  
    evaluation.set_predicted_masks_from_directory( predicted_masks_directory )
    evaluation.set_expected_masks_from_directory( expected_masks_directory )
    evaluation.calculate_iou()
    evaluation.calculate_score_with_thresholds()
    iou = evaluation.get_iou()
    score = evaluation.get_score()
    evaluation.print_table()

Function list:
    set_predicted_masks_from_directory( predicted_masks_directory )
    set_expected_masks_from_directory( expected_masks_directory )
    set_predicted_masks( predicted_masks_directory )
    set_expected_masks( expected_masks_directory )
    calculate_iou()
    calculate_score_from_thresholds( thresholds=None )
    get_iou()
    get_score()
    get_table()
    print_table()

"""

import pandas
import numpy
import skimage.io
import skimage.segmentation

import sys, os
sys.path.append( os.path.join( '..', '..', 'src' ) ) # set path to project src/
from utils import kaggle_io, kaggle_reader

class KaggleEvaluation( object ):
    
    def __init__( self, predicted_masks_directory=None, expected_masks_directory=None, predicted_masks=None, expected_masks=None ):
        
        """ The class constructor.
        
        :param predicted_mask: The predicted mask (all nuclei) from prediction
        :param expected_masks_directory: Directory containing all the masks
        :param expected_mask: Directly input the expected mask
        :return: None

        """

        if not predicted_masks_directory is None:
            self.predicted_masks = self.set_predicted_masks_from_directory( predicted_masks_directory )
        else:
            self.predicted_masks = predicted_masks
        if not expected_masks_directory is None:
            self.expected_masks = self.set_expected_masks_from_directory( expected_masks_directory )
        else:
            self.expected_masks = expected_masks

        # global result variables
        self.IoU = None
        self.table = None
        self.score = None
        
        return


    def set_predicted_masks( self, predicted_masks ):

        """ Set the predicted masks

        :param predicted_masks: The input predicted mask containing all nuclei in separate entries
        :return: None

        """

        self.predicted_masks = predicted_masks
        return


    def set_expected_masks( self, expected_masks ):

        """ Set the expected masks

        :param expected_mask: The input expected mask containing all nuclei in separate entries
        :return: None

        """

        self.expected_masks = expected_masks
        return


    def set_expected_masks_from_directory( self, expected_masks_directory ):

        """ Set the expected masks from directory

        :param expected_masks_directory: Directory containing all the expected masks
        :return: None

        """

        self.expected_masks = kaggle_reader.get_masks_from_directory( expected_masks_directory )
        return

    
    def set_predicted_masks_from_directory( self, predicted_masks_directory ):

        """ Set the predicted masks from directory

        :param predicted_masks_directory: Directory containing all the predicted masks
        :return: None

        """

        self.predicted_masks = kaggle_reader.get_masks_from_directory( predicted_masks_directory )
        return


    def calculate_iou( self ):

        """ Calculate the iou 

        :return: None

        """

        if self.expected_masks is None:
            raise Exception( 'Expected mask has not yet been set' )
        if self.predicted_masks is None:
            raise Exception( 'Predicted mask has not yet been set' )

        # useful things to track
        number_of_expected = len( self.expected_masks )
        number_of_predicted = len( self.predicted_masks )

        # reset IoU just in case
        self.IoU = []

        # calculate IoU
        for predicted in range( number_of_predicted ):

            # track the best results
            # union count is initialized at 1 since the union is always larger than 0
            # if union count is initialized at 0 and no intersection is never larger than 0, a division by 0 occurs
            best_intersection = 0
            best_union = 1

            for expected in range( number_of_expected ):

                # points where BOTH expected AND predicted have nuclei
                intersection_points = self.predicted_masks[ predicted ] * self.expected_masks[ expected ] 

                # total number of points where BOTH expected AND predicted have nuclei
                intersection_count = numpy.sum( intersection_points )

                # points where EITHER expected OR predicted have nuclei
                union_points = numpy.maximum( self.predicted_masks[ predicted ], self.expected_masks[ expected ] )

                # total number of points where EITHER expected OR predicted have nuclei
                union_count = numpy.sum( union_points )

                if intersection_count > best_intersection:
                    best_intersection = intersection_count
                    best_union = union_count

            # IoU calculation
            self.IoU.append( best_intersection / best_union )

        return 


    def calculate_score_with_thresholds( self, thresholds=None ):

        """ Calculate the scores with thresholds
        
        :param thresholds: Optional argument to specific thresholds, otherwise Kaggle default
        :return: None

        """

        if self.IoU is None:
            raise Exception( 'The IoU has not been calculated yet: calculate_iou()' )

        if thresholds is None:
            thresholds = numpy.arange( 0.5, 1.0, 0.05 )

        # useful things to track
        number_of_expected = len( self.expected_masks )
        number_of_predicted = len( self.predicted_masks )

        # reset the variables just in case
        self.score = 0
        self.table = "Thres.\tTP\tFP\tFN\tPrecision"

        # consider all the thresholds
        for threshold in thresholds:

            # keep track of the matches over the threshold
            matches = self.IoU > threshold

            # calculate various counts
            true_positives = numpy.count_nonzero( matches )
            false_positives = number_of_predicted - true_positives
            false_negatives = number_of_expected - true_positives

            # determine the precision and use it to create a score
            precision = true_positives / ( true_positives + false_positives + false_negatives )
            self.score += precision

            # record results for every threshold
            self.table += "\n{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format( threshold, int(true_positives), int(false_positives), int(false_negatives), precision )

        # normalize by number of thresholds
        self.score = self.score / len( thresholds )
        
        # print out the final score
        self.table += "\nScore\t-\t-\t-\t{:1.3f}".format( self.score )

        return


    def get_iou( self ):

        """ Get the table with per threshold results

        :return: Array of IoU scores

        """

        if self.IoU is None:
            raise Exception( 'The IoU has not been calculated yet: calculate_iou()' )

        return self.IoU


    def get_score( self ):

        """ Get the single score according to Kaggle

        :return: Float score

        """

        if self.score is None:
            raise Exception( 'The score has not been calculated yet: calculate_score_with_thresholds()' )

        return self.score


    def get_table( self ):

        """ Get the table with per threshold results

        :return: String table with results

        """

        if self.table is None:
            raise Exception( 'The score table has not been calculated yet: calculate_score_with_thresholds()' )

        return self.table


    def print_table( self ):

        """ Print out the table after calculations 
        
        :return: None

        """

        if self.table is None:
            raise Exception( 'The score table has not been calculated yet: calculate_score_with_thresholds()' )

        print()
        print( self.table )
        print()


if __name__ == '__main__':

    # example run
    expected_masks_directory = '../../data/raw/stage1_train/2e2d29fc44444a85049b162eb359a523dec108ccd5bd75022b25547491abf0c7/masks'
    predicted_masks_directory = '../../data/raw/stage1_train/2e2d29fc44444a85049b162eb359a523dec108ccd5bd75022b25547491abf0c7/masks'

    evaluation = KaggleEvaluation()
    evaluation.set_predicted_masks_from_directory( predicted_masks_directory )
    evaluation.set_expected_masks_from_directory( expected_masks_directory )
    evaluation.calculate_iou()
    evaluation.calculate_score_with_thresholds()
    evaluation.print_table()

