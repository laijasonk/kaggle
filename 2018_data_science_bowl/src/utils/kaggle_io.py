
"""Take input and convert it into various different output

Example: 
    sys.path.append( os.path.join( '.', 'src' ) ) # set path to project src/
    from utils import kaggle_io
    output = kaggle_io.png_to_pixel_matrix( '/path/to/image', mode=RGB )
    output = kaggle_io.mask_png_to_kaggle_format( 'path/to/mask', 'image_id' )

Function list:
    png_to_pixel_matrix( image_path, mode='RGB' )
    mask_png_to_mask_matrix( image_path )
    mask_png_to_mask_vector( image_path )
    mask_png_to_nuclei_list( image_path )
    mask_png_to_kaggle_format( image_path, image_id )

"""

import numpy as np
import pandas as pd
import scipy.misc


def png_to_pixel_matrix( image_path, mode='RGB' ):
 
    """ PNG to HxW matrix of image values for every pixel

    :param image_path: File path to the image file
    :return: HxW matrix of image values for every pixel

    """

    pixel_matrix = scipy.misc.imread( image_path, flatten=False, mode=mode )
    return pixel_matrix


def mask_png_to_mask_matrix( image_path ):

    """ Mask PNG to HxW matrix of non-black pixels (1=nuclei/non-black, 0=black)

    :param image_path: Path to the image file
    :return: HxW binary matrix where 1=nuclei/non-black and 0=black

    """
    
    pixel_matrix = png_to_pixel_matrix( image_path, mode='L' )
    pixel_matrix[pixel_matrix!=0] = 1
    """
    height, width = pixel_matrix.shape[ 0:2 ]

    mask_matrix = np.zeros( ( height, width ) )
    for row in range( height ):
        for col in range( width ):

            # any non-black pixel is marked with 1
            if not ( pixel_matrix[row][col] == [ 0, 0, 0 ] ).all():
                mask_matrix[row][col] = 1
    """

    return pixel_matrix


def pixel_matrix_to_mask_matrix( pixel_matrix ):

    """ HxW matrix of image to binary matrix with non-black pixels (1=nuclei/non-black, 0=black)

    :param pixel_matrix: Image matrix, typically in RGB
    :return: HxW binary matrix where 1=nuclei/non-black and 0=black

    """
    
    height, width = pixel_matrix.shape[ 0:2 ]

    mask_matrix = np.zeros( ( height, width ) )
    for row in range( height ):
        for col in range( width ):

            # any non-black pixel is marked with 1
            if not ( pixel_matrix[row][col] == [ 0, 0, 0 ] ).all():
                mask_matrix[row][col] = 1

    return mask_matrix


def mask_png_to_mask_vector( image_path ):
    
    """ Vectorized version of mask_matrix (matching Kaggle's top->bottom, left->right format)

    :param image_path: Path to the image file
    :return: Binary vector where 1=nuclei/non-black and 0=black

    """
    
    mask_matrix = mask_png_to_mask_matrix( image_path )
    height, width = mask_matrix.shape
    
    # transpose matrix so that reshape goes top to bottom, left to right
    mask_matrix_T = np.transpose( mask_matrix, axes=( 1, 0 ) )
    mask_vector = mask_matrix_T.reshape( ( height * width ) )

    return mask_vector


def mask_png_to_nuclei_list( image_path ):

    """ Mask PNG to nuclei 1D list containing indices of every nuclei (non-black) pixel

    :param image_path: Path to the image file
    :return: 1D list containing indices of every nuclei (non-black) pixel

    """

    mask_vector = mask_png_to_mask_vector( image_path )
    nuclei_list = []
    for pixel_idx in range( len( mask_vector ) ):
        
        # any non-black pixel is saved
        if mask_vector[ pixel_idx ] == 1:
            
            # index is corrected since Kaggle indices starts at 1 (not 0) 
            nuclei_list.append( pixel_idx + 1 )

    return nuclei_list


def mask_matrix_to_nuclei_list( mask_matrix ):
    
    """ Mask matrix to nuclei 1D list containing indices of every nuclei (non-black) pixel
    
    :param mask_matrix: HxW matrix (see other functions)
    :return: 1D list containing indices of every nuclei (non-black) pixel

    """

    height, width = mask_matrix.shape
    mask_matrix_T = np.transpose( mask_matrix, axes=( 1, 0 ) )
    mask_vector = mask_matrix_T.reshape( ( height * width ) )
    
    nuclei_list = []
    for pixel_idx in range( len( mask_vector ) ):
        
        # any non-black pixel is saved
        if mask_vector[ pixel_idx ] == 1:
            
            # index is corrected since Kaggle indices starts at 1 (not 0) 
            nuclei_list.append( pixel_idx + 1 )

    return nuclei_list


def mask_png_to_kaggle_format( image_path, image_id ):

    """ Mask PNG to formatted Kaggle submission string
    
    :param image_path: Path to image
    :return: String in the format of Kaggle's submission output

    """

    nuclei_list = mask_png_to_nuclei_list( image_path )
    return nuclei_list_to_kaggle_format( nuclei_list, image_id )


def mask_matrix_to_kaggle_format( mask_matrix, image_id):

    """ Mask matrix to formatted Kaggle submission string
    
    :param mask_matrix: HxW matrix (see other functions)
    :return: String in the format of Kaggle's submission output

    """

    nuclei_list = mask_matrix_to_nuclei_list( mask_matrix )
    return nuclei_list_to_kaggle_format( nuclei_list, image_id )


def nuclei_list_to_kaggle_format( nuclei_list, image_id ):
    
    """ Nuclei list to formatted Kaggle submit string
    
    :param nuclei_list: List of nuclei indices (see other functions)
    :return: String in the format of Kaggle's submission output

    """

    previous = -1
    kaggle_format = []
    for idx in range( len( nuclei_list ) ):

        current = nuclei_list[ idx ]

        # kaggle format has index followed by number of consecutive pixels
        if current - 1 == previous:
            kaggle_format[-1] += 1
        else:
            kaggle_format.append( current )
            kaggle_format.append( 1 )

        previous = nuclei_list[ idx ]

    kaggle_string = [ str( element ) for element in kaggle_format ]
    return [str( image_id ), str( ' '.join( kaggle_string ) )]


if __name__ == '__main__':

    # Below are example usages of utils

    image_path = '../../data/raw/stage1_train/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552/masks/0e548d0af63ab451616f082eb56bde13eb71f73dfda92a03fbe88ad42ebb4881.png'
    image_id = '00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552'

    # PNG to a HxW matrix of color values for each pixel
    pixel_matrix = png_to_pixel_matrix( image_path )
    
    # Mask PNG to a HxW matrix of nuclei/non-black pixels (1=nuclei/non-black, 0=black)
    mask_matrix = mask_png_to_mask_matrix( image_path )

    # Mask PNG to 1D list containing indices of all nuclei/non-black pixels
    nuclei_list = mask_png_to_nuclei_list( image_path )
    
    # Format mask PNG to a string matching the Kaggle submission format
    submit_string = mask_png_to_kaggle_format( image_path, image_id ) 

    # Format mask_matrix to a string matching the Kaggle submission format
    submit_string = mask_matrix_to_kaggle_format( mask_matrix, image_id )
    
    # Format nuclei_list to a string matching the Kaggle submission format
    submit_string = nuclei_list_to_kaggle_format( nuclei_list, image_id )

    # Sample submit_string
    print( submit_string )

