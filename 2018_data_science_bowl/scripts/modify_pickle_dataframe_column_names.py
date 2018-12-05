
""" Correct the column names of the main pickle dataframes """

import numpy
import pandas
import skimage.color
    
import sys,os
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from preprocess import image_processing


def main( input_pickle_path, output_pickle_path ):
    
    """ The main function

    :param input_pickle_path: The path to the pickled dataframe to be modified
    :param output_pickle_path: The path to the output pickle file where dataframe will be saved
    :return: None

    """

    in_dataframe = pandas.read_pickle( input_pickle_path )
    out_dataframe = in_dataframe.copy()

    # make changes to column names (all at once)
    out_dataframe.columns = [ 'filename', 'parent_image_id', 'extract_height', 'extract_width', 'image_height', 'image_width', 'index_row', 'index_col', 'is_positive', 'image_matrix' ]

    # alternatively... (one at a time)
    # out_dataframe.rename( { 'extract_width':'extract_height_new' }, in_place=True )
    # out_dataframe.rename( { 'extract_height':'extract_width_new' }, in_place=True )
    # out_dataframe.rename( { 'extract_width_new':'extract_width' }, in_place=True )
    # out_dataframe.rename( { 'extract_height_new':'extract_height' }, in_place=True )
    # etc.

    print( 'Saving new pickle dataframe to ' + output_pickle_path )
    out_dataframe.to_pickle( output_pickle_path )

    return


if __name__ == '__main__':

    input_pickle_path = os.path.join( '..', 'data', 'roi', 'test_dataframe.pkl' )
    output_pickle_path = os.path.join( '..', 'data', 'roi', 'test_dataframe_fixed.pkl' )
    main( input_pickle_path, output_pickle_path )

    input_pickle_path = os.path.join( '..', 'data', 'roi', 'train_dataframe.pkl' )
    output_pickle_path = os.path.join( '..', 'data', 'roi', 'train_dataframe_fixed.pkl' )
    main( input_pickle_path, output_pickle_path )

    input_pickle_path = os.path.join( '..', 'data', 'roi_32buffer', 'test_dataframe.pkl' )
    output_pickle_path = os.path.join( '..', 'data', 'roi_32buffer', 'test_dataframe_fixed.pkl' )
    main( input_pickle_path, output_pickle_path )

    input_pickle_path = os.path.join( '..', 'data', 'roi_32buffer', 'train_dataframe.pkl' )
    output_pickle_path = os.path.join( '..', 'data', 'roi_32buffer', 'train_dataframe_fixed.pkl' )
    main( input_pickle_path, output_pickle_path )

