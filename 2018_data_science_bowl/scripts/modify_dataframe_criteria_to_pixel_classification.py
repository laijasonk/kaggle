
""" Modify an input pickle dataframes to match criteria of center pixel being a nuclei

"""

import numpy
import pandas
import skimage.color
    
import sys,os
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from preprocess import image_processing
from utils import kaggle_reader


def main( input_pickle_path, output_pickle_path, masks_dict, window_center_nuclei_distance ):
    
    """ The main function

    :param input_pickle_path: The path to the pickled dataframe to be modified
    :param output_pickle_path: The path to the output pickle file where dataframe will be saved
    :return: None

    """

    print( 'Reading input pickle dataframe: ' + input_pickle_path )
    in_dataframe = pandas.read_pickle( input_pickle_path )
    out_dataframe = in_dataframe.copy()

    # save info to track progress
    count = 0
    total_rows = in_dataframe.shape[ 0 ]

    print( 'Iterating through all rows' )
    for idx, row in in_dataframe.iterrows():

        # output the progress of the script
        if count % 100 == 0:
            print( str( idx ) + " / " + str( total_rows ) + " rows altered" )
        count += 1

        # manipulate specific column of the dataframe at current index
        image_id = row.parent_image_id

        # whether center pixel is part of a nuclei
        is_nuclei = '0'
        height, width = masks_dict[ image_id ].shape
        for center_row in range(  row.index_row - window_center_nuclei_distance, row.index_row + window_center_nuclei_distance ):
            for center_col in range( row.index_col - window_center_nuclei_distance, row.index_col + window_center_nuclei_distance ):
                try: 
                    if masks_dict[ image_id ][ center_row ][ center_col ] == 1:
                        is_nuclei = '1'
                except:
                    print( 'Out-of-bounds: ' + image_id )
                    print( 'Center point (' + str( center_row ) + ', ' + str( center_col ) + ') is not within matrix size (' + str( row.image_height ) + ', ' + str( row.image_width ) + ')' )

        # apply changes to output dataframe on specific index and column
        out_dataframe.at[ idx, 'is_positive' ] = is_nuclei
        
    print( 'Saving new pickle dataframe to ' + output_pickle_path )
    out_dataframe.to_pickle( output_pickle_path )

    return


if __name__ == '__main__':
    
    ########################
    # UPDATE THESE INPUT
    # 

    input_train_pickle_path = os.path.join( '..', 'data', 'roi', 'train_dataframe.pkl' )
    output_train_pickle_path = os.path.join( '..', 'data', 'roi', 'train_dataframe_center_pixel.pkl' )

    input_test_pickle_path = os.path.join( '..', 'data', 'roi', 'test_dataframe.pkl' )
    output_test_pickle_path = os.path.join( '..', 'data', 'roi', 'test_dataframe_center_pixel.pkl' )

    raw_path = os.path.join( '..', 'data', 'raw', 'stage1_train' )
    
    # pixels around the center point (e.g. 2px means all pixel from (center-2) to (center+2), which totals 5px around center at each direction)
    window_center_nuclei_center = 1

    #
    # END USER INPUT
    ########################


    print( 'Loading masks' )

    # save info to track progress
    count = 0
    train_list = os.listdir( raw_path )
    train_size = len( train_list )

    masks_dict = {}
    for image_id in train_list:

        # print out progress
        if count % 10 == 0:
            print( str( count ) + ' / ' + str( train_size ) + ' mask directories read' )
        count += 1

        # read in masks directory and convert into a single binary array
        masks_directory = os.path.join( raw_path, image_id, 'masks' )
        masks_array = kaggle_reader.get_masks_from_directory( masks_directory )
        image_height, image_width = masks_array[0].shape
        mask_matrix = numpy.zeros( ( image_height, image_width ), numpy.uint16 )
        for current_mask in masks_array:
            mask_matrix = numpy.maximum( mask_matrix, current_mask )

        # save to array
        masks_dict[ image_id ] = mask_matrix

    # run the main function
    main( input_train_pickle_path, output_train_pickle_path, masks_dict, window_center_nuclei_center )
    main( input_test_pickle_path, output_test_pickle_path, masks_dict, window_center_nuclei_center )

