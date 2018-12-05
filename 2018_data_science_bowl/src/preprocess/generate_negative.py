
"""Functions to generate a negative image/box for a given image

Example: 
    sys.path.append( os.path.join( '.', 'src' ) ) # set path to project src/
    from preprocess import generate_negative
    negative = generate_negative.GenerateNegative()  
    negative.set_image_with_png( '/path/to/image.png' )
    negative.set_negative_box_size( [ heigth, width ] )
    negative.set_random_seed( seed )
    negative.run()
    output = negative.get_negative()
    negative.save_negative_as_image( '/path/to/output.png' )

Function list:
    set_image_with_png( image_path )
    set_image_with_matrix( image_matrix )
    set_negative_box_size( negative_size )
    set_random_seed( seed )
    run()
    get_negative()
    save_negative_as_image( output_path )

"""

import numpy as np
import scipy.misc
import random

import sys, os
sys.path.append( os.path.join( '..', '..', 'src' ) ) # point to src/
from utils import kaggle_io


class GenerateNegative( object ):
    
    def __init__( self, image_matrix=None, negative_box_size=None, negative_matrix=None, random_seed=1 ):
        
        """ The class constructor.
        
        :param image_matrix: Pixel matrix of the parent image
        :param negative_box_size: The height and width of the negative box, format=[height,width]
        :param negative_matrix: Predefine the pixel matrix of the negative image
        :param random_seed: Set the random seed for reproducibility
        :return: None
            
        """

        self.image = image_matrix
        self.size = negative_box_size
        self.negative = negative_matrix
        self.seed = random_seed

        self.negative_row = 0 # placeholder
        self.negative_col = 0 # placeholder
        
        return


    def set_image_with_png( self, image_path ):

        """ Set the image based on an input image path.
        
        :param image_path: Path to the image file
        :return:

        """

        try:
            self.image = kaggle_io.png_to_pixel_matrix( image_path )
        except:
            raise Exception( 'Image path not valid' )

        return
    

    def set_image_with_matrix( self, image_matrix ):

        """ Set the image input based on an input WxH matrix.
        
        :param image_matrix: WxH matrix of the image
        :return: None
        
        """

        if not len( image_matrix[0] ) > 1:
            raise Exception( 'Invalid input image matrix: must be WxH array' )
        self.image = image_matrix

        return


    def set_negative_box_size( self, negative_size ):

        """ Set the negative box size as [ height, width ].
        
        :param negative_size: The size of the negative box, format=[height,width]
        :return: None
        
        """

        if not len( negative_size ) == 2:
            raise Exception( 'Negative box size should be formated as [ height, width ]' )
        self.size = negative_size

        return


    def set_random_seed( self, random_seed ):

        """ Set the random seed.
        
        :param random_seed: Specify a constant random seed
        :return: None

        """

        try:
            self.seed = int( random_seed )
        except:
            raise Exception( 'Invalid random seed' )

        return

    def run( self, roi_center_list=None, buffer_size=0 ):
        
        """ Generate the negative box based on previously set arguments.

        :param roi_center_list: Path to file containing ROI centers
        :param buffer_size: Distance from ROI center
        :return: None
        
        """

        if self.image is None or self.size is None or self.seed is None:
            raise Exception( 'The image, size, and seed must be passed to class before running.' )

        # get the dimensions of the current image
        height, width = self._get_dimensions( self.image )

        # set the random seed
        random.seed( self.seed )

        # calculate half height and width of negative box
        row_min_from_center = int( np.floor( self.size[0] / 2 ) - 1 )
        row_max_from_center = self.size[0] - row_min_from_center
        col_min_from_center = int( np.floor( self.size[1] / 2 ) - 1 )
        col_max_from_center = self.size[1] - col_min_from_center

        # choose an index within the size of the current image minus the negative box size
        self.negative_row = random.randint( row_min_from_center, height - row_max_from_center  )
        self.negative_col = random.randint( col_min_from_center, width - col_max_from_center  )

        # if the image center is too close to a known positive example, skip and find a new location
        if roi_center_list is not None:

            while self._near_roi_center( self.negative_row, self.negative_col, roi_center_list, buffer_size ):

                # print('Never tell me the odds!')
                self.negative_row = random.randint( row_min_from_center, height - row_max_from_center  )
                self.negative_col = random.randint( col_min_from_center, width - col_max_from_center  )

        # define dimensions of the negative box (nbox)
        nbox_row_min, nbox_row_max, nbox_col_min, nbox_col_max = self._get_negative_box( self.negative_row, self.negative_col )

        # extract negative from image
        self.negative = self.image[nbox_row_min:nbox_row_max, nbox_col_min:nbox_col_max]

        return
    

    def get_negative( self ):

        """ Return a matrix containing the image information in the negative box.
        
        :return: Pixel matrix of the negative image

        """

        if self.negative is None:
            raise Exception( 'The GenerateNegative class has not been run() yet.' )
        
        return self.negative


    def save_negative_as_npy( self, output_npy_path ):

        """ Takes the negative box and saves it to a numpy array. Assumes image is RGB.
        
        :param output_npy_path: Path to the output numpy array
        :return: None

        """

        if self.negative is None:
            raise Exception( 'The negative box has not been generated yet.' )
        np.save( output_npy_path, self.negative )

        return


    def save_negative_as_image( self, output_image_path ):

        """ Takes the negative box and saves it to an image. Assumes image is RGB.

        :param output_image_path: Path to the output image
        :return: None
        
        """

        if self.negative is None:
            raise Exception( 'The negative box has not been generated yet.' )
        scipy.misc.imsave( output_image_path, self.negative )

        return


    def _near_roi_center( self, row, col, roi_center_list, buffer_size=0 ):

        """ Returns a bool indicating if the randomly chosen center is too close to a positive example.
        
        :param row: proposed row coordinate
        :param col: proposed col coordinate
        :param roi_center_list: Path to file containing roi centers
        :param buffer_size: Distance from roi center
        :return: True if near center, False if not
        
        """

        # calculates L1 distance between [row, col] and locations in roi_center_list
        try:
            distances = np.sum( np.abs( roi_center_list - np.array( [row, col] ) ), axis=1 )
        except:
            # special case where only one mask
            return np.sum( np.abs( roi_center_list - np.array( [row, col] ) ) ) <= buffer_size

        return min( distances ) <= buffer_size
    

    def _get_dimensions( self, array ):

        """ Simple function to get height and width dimensions of 2D matrix.
        
        :param array: Input image array
        :return: The height and width of the array

        """

        height = len( array )
        width = len( array[ 0 ] )

        return [ height, width ]


    def _get_negative_box( self, row_center, col_center ):

        """ Calculate negative box with a around nuclei center.
        
        :param row_center: The center in row coordinates
        :param col_center: The center in col coordinates
        :return: The four corners of the negative bounding box

        """

        # get dimensions of image (image should be same size as mask)
        height, width = self._get_dimensions( self.image )
        
        # calculate half height and width of the negative box
        fixed_row_min_from_center = int( np.floor( self.size[0] / 2 ) - 1 )
        fixed_row_max_from_center = self.size[0] - fixed_row_min_from_center
        fixed_col_min_from_center = int( np.floor( self.size[1] / 2 ) - 1)
        fixed_col_max_from_center = self.size[1] - fixed_col_min_from_center
        
        # create new bounding box depending on input arguments
        nbox_row_min = row_center - fixed_row_min_from_center
        nbox_row_max = row_center + fixed_row_max_from_center
        nbox_col_min = col_center - fixed_col_min_from_center
        nbox_col_max = col_center + fixed_col_max_from_center

        return nbox_row_min, nbox_row_max, nbox_col_min, nbox_col_max


if __name__ == '__main__':

    # Below are example usages of utils
    
    image_path = '../../data/raw/stage1_train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e/images/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png'
    negative = GenerateNegative()

    # Set the inputs according to paths (also possible to just give the class the actual matrices)
    negative.set_image_with_png( image_path )

    # Set the size of the negative box
    negative.set_negative_box_size( [ 64, 64 ] )

    # Set the random seed (not necessary)
    negative.set_random_seed( 1 )

    # Run the class to generate negative box
    negative.run()

    # Example of the resulting negative box matrix (pixel values of the negative box)
    #print( negative.get_negative() )
    
    # Save the negative as an image
    output_path = './negative.png'
    negative.save_negative_as_image( output_path )
    print( 'Negative image saved as ' + str( output_path ) )

