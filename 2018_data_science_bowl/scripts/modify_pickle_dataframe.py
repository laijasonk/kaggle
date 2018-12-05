
""" Modify an input pickle dataframe and save it 

Requires modification to script where comments indicate

"""

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

    count = 0
    total_rows = in_dataframe.shape[ 0 ]

    for idx, row in in_dataframe.iterrows():

        # output the progress of the script
        if count % 100 == 0:
            print( str( idx ) + " / " + str( total_rows ) + " rows finished" )
        count += 1


        ########################
        # EDIT THIS SECTION
        #
        
        # manipulate specific column of the dataframe at current index
        in_image_matrix = row.image_matrix
        gray_image_matrix = skimage.color.rgb2gray( row.image_matrix )
        out_image_matrix = image_processing.trim_extreme_pixels_in_grayscale( gray_image_matrix, trim_left=1, trim_right=99)

        # apply changes to output dataframe on specific index and column
        out_dataframe.at[ idx, 'image_matrix' ] = out_image_matrix
        
        #
        # END EDITS
        ########################


    print( 'Saving new pickle dataframe to ' + output_pickle_path )
    out_dataframe.to_pickle( output_pickle_path )

    return


if __name__ == '__main__':

    ########################
    # UPDATE THESE PATHS
    # 

    input_pickle_path = os.path.join( '..', 'data', 'roi_32buffer', 'test_dataframe.pkl' )
    output_pickle_path = os.path.join( '..', 'data', 'roi_32buffer', 'test_dataframe_gray_trimmed.pkl' )

    # run the main function
    main( input_pickle_path, output_pickle_path )

