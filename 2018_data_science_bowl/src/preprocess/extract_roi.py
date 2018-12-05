
"""Extract the ROI given an image and mask PNG/matrix/etc

Example: 
    sys.path.append( os.path.join( '.', 'src' ) ) # set path to project src/
    from preprocess import extract_roi
    extractor = extract_roi.ExtractROI()  
    extractor.set_image_with_png( '/path/to/image.png' )
    extractor.set_mask_with_png( '/path/to/mask.png' )
    extractor.extract_roi( padding=10 )
    extractor.save_roi_as_image( '/path/to/output.png' )

Function list:
    set_image_with_png( image_path )
    set_image_with_matrix( image_matrix )
    set_mask_with_png( mask_path )
    set_mask_with_matrix( mask_matrix )
    extract_roi( padding=0, fixed_size=None )
    get_roi()
    save_roi_as_image( output_image_path )

"""

import numpy as np
import scipy.misc

import sys, os
sys.path.append( os.path.join( '..', '..', 'src' ) )
from utils import kaggle_io


class ExtractROI( object ):
    
    def __init__( self, image_matrix=None, mask_matrix=None, roi_matrix=None, roi_mask=None, center=None ):
        
        """ The class constructor.
        
        :param image_matrix: Pixel matrix of the parent image
        :param mask_matrix: Pixel matrix of the mask image
        :param roi_matrix: Pixel matrix of the ROI image
        :param roi_mask: Binary matrix of the ROI mask image
        :param center: Center of the ROI
        :return: None

        """

        self.image = image_matrix
        self.mask = mask_matrix
        self.roi = roi_matrix
        self.roi_mask = roi_mask
        self.roi_center = center
        
        return


    def set_image_with_png( self, image_path ):

        """ Set the image based on an input image path.
        
        :param image_path: Path to the parent image
        :return: None

        """

        try:
            self.image = kaggle_io.png_to_pixel_matrix( image_path )
        except:
            raise Exception( 'Image path not valid.' )

        return
    

    def set_image_with_matrix( self, image_matrix ):

        """ Set the image input based on an input HxW matrix.
        
        :param image_matrix: Pixel matrix of the parent image
        :return: None
        
        """

        if not len( image_matrix[0] ) > 1:
            raise Exception( 'Invalid input image matrix: must be HxW array' )
        self.image = image_matrix

        return


    def set_mask_with_png( self, mask_image_path ):

        """ Set the mask based on an input image path.
        
        :param mask_image_path: Path to the mask image
        :return: None
        
        """

        try:
            self.mask = kaggle_io.mask_png_to_mask_matrix( mask_image_path )
        except:
            raise Exception( 'Mask image path not valid.' )

        return


    def set_mask_with_matrix( self, mask_matrix ):

        """ Set the mask based on an input matrix of 0s and 1s.
        
        :param mask_matrix: Pixel matrix of the mask image
        :return: None
        
        """

        if not len( mask_matrix[0] ) > 1:
            raise Exception( 'Invalid input mask matrix: must be HxW array' )
        self.mask = mask_matrix

        return


    def extract_roi( self, padding=0, fixed_size=None, fixed_1x1_ratio=False ):

        """ Extract the ROI from the image based on the previously set.
        
        :param padding: Padding around the nuclei in pixels
        :param fixed_size: Dimensions of the bounding box, format=[height,width]
        :param fixed_1x1_ratio: Minimal box around nuclei with 1:1 aspect ratio
        :return: None
        
        """

        if self.image is None or self.mask is None:
            raise Exception( 'The image and mask must be set before extracting ROI' )
        
        if ( not padding == 0 and not fixed_size == None ) or ( not padding == 0 and not fixed_1x1_ratio == False ) or ( not fixed_size == None and not fixed_1x1_ratio == False ):
            raise Exception( 'Only one type of criteria for extracting ROI can be passed as argument simultaneously.' )

        image_height, image_width = self._get_dimensions( self.image )
        mask_height, mask_width = self._get_dimensions( self.mask )

        if not image_height == mask_height or not image_width == mask_width:
            raise Exception( 'The input image and mask do not match in size' )

        # create a list of nuclei indices
        nuclei_indices = np.array( kaggle_io.mask_matrix_to_nuclei_list( self.mask ) )

        # get the upper and lower dimensions of bounding box after applying padding
        bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max = self._get_bounding_box( nuclei_indices, padding ) 

        # calculate the center point of the nucleus, also the center of thr ROI and ROI mask
        nuclei_center = [ int( np.floor( ( bbox_row_max + bbox_row_min ) / 2 ) ),
                          int( np.floor( ( bbox_col_max + bbox_col_min ) / 2 ) ) ]
        # force bounding box to fixed dimensions if specified
        if not fixed_size is None:
            bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max = self._get_fixed_bounding_box ( nuclei_center, fixed_size )
    
        # force bounding box to 1x1 ratio (overrides other options)
        if fixed_1x1_ratio == True:
            length = max( bbox_row_max - bbox_row_min + 1, bbox_col_max - bbox_col_min + 1 )
            bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max = self._get_fixed_bounding_box ( nuclei_center, [ length, length ] )

        # store nucleus center location
        self.roi_center = nuclei_center

        # extract ROI from image based on mask
        self.roi = self.image[bbox_row_min:bbox_row_max, bbox_col_min:bbox_col_max]

        # extract mask for the ROI
        self.roi_mask = self.mask[bbox_row_min:bbox_row_max, bbox_col_min:bbox_col_max]

        return


    def get_roi( self ):

        """ Return a matrix containing the image information in the ROI.
        
        :return: The HxW pixel matrix of the ROI
        
        """

        if self.roi is None:
            raise Exception( 'The ROI has not been extracted yet.' )
        
        return self.roi


    def save_roi_mask_as_npy( self, output_npy_path ):

        """ Takes a the ROI and saves it to a numpy array. Assumes image is RGB.
        
        :param output_npy_path: Path to the output image to be saved
        :return: None
        
        """

        if self.roi_mask is None:
            raise Exception( 'The ROI has not been extracted yet.' )
        np.save( output_npy_path, self.roi_mask )

        return


    def save_roi_mask_as_image( self, output_image_path ):

        """ Takes a the ROI and saves it to an image. Assumes image is RGB.
        
        :param output_image_path: Path to the output image to be saved
        :return: None
        
        """

        if self.roi_mask is None:
            raise Exception( 'The ROI has not been extracted yet.' )
        scipy.misc.imsave( output_image_path, self.roi_mask )

        return


    def save_roi_as_npy( self, output_npy_path ):

        """ Takes a the ROI and saves it to a numpy array. Assumes image is RGB.
        
        :param output_image_path: Path to the output npy to be saved
        :return: None
        
        """

        if self.roi is None:
            raise Exception( 'The ROI has not been extracted yet.' )
        np.save( output_npy_path, self.roi )

        return


    def save_roi_as_image( self, output_image_path ):

        """ Takes a the ROI and saves it to an image. Assumes image is RGB.
       
        :param output_image_path: Path to the output image to be saved
        :return: None
        
        """
        
        if self.roi is None:
            raise Exception( 'The ROI has not been extracted yet.' )
        scipy.misc.imsave( output_image_path + '.png', self.roi )

        return
    

    def _get_dimensions( self, array ):

        """ Simple function to get height and width dimensions of 2D matrix.

        :param array: Image pixel array
        :return: Dimensions of the image array
        
        """

        height = len( array )
        width = len( array[ 0 ] )

        return [ height, width ]


    def _get_bounding_box( self, nuclei_indices, padding ):
        
        """ Determine the upper and lower values for the bounding box along row and col.
        
        :param nuclei_indices: List of nuclei indices
        :param padding: Padding around the nuclei
        :return: Four corners of the bounding box
        
        """

        # get dimensions of image (image should be same size as mask)
        height, width = self._get_dimensions( self.image )
        
        # convert 1D list of nuclei_indices into columns and rows
        nuclei_indices = np.array( nuclei_indices ) - 1
        cols = np.ceil( nuclei_indices / height ).astype( int )
        rows = np.mod( nuclei_indices, height ).astype( int )

        # defining the bounding box (bbox)
        bbox_row_min = min( rows )
        bbox_row_max = max( rows )
        bbox_col_min = min( cols )
        bbox_col_max = max( cols )
        
        # update bbox with paddings (only if within the dimensions of the original image)
        bbox_row_min = max( 0, bbox_row_min - padding )
        bbox_row_max = min( height, bbox_row_max + padding )
        bbox_col_min = max( 0, bbox_col_min - padding )
        bbox_col_max = min( width, bbox_col_max + padding )

        return bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max

 
    def _get_fixed_bounding_box( self, nuclei_center, fixed_size ):

        """ Calculate bounding box with a fixed size around nuclei center.
        
        :param nuclei_center: row/col coordinates of the nuclei center
        :param fixed_size: Size of the bounding box, format=[height,width]
        :return: Four corners of the bounding box
        
        """

        if len( nuclei_center ) < 2:
            raise Exception( 'The nuclei_center must be in the form [ row, col ].' )
        if len( fixed_size ) < 2:
            raise Exception( 'The fixed_size must be in the form [ height, width ].' )

        # get dimensions of image (image should be same size as mask)
        height, width = self._get_dimensions( self.image )
        
        # calculate half height and width of fixed bounding box
        fixed_row_min_from_center = int( np.floor( fixed_size[0] / 2 ) - 1 )
        fixed_row_max_from_center = fixed_size[0] - fixed_row_min_from_center
        fixed_col_min_from_center = int( np.floor( fixed_size[1] / 2 ) - 1 )
        fixed_col_max_from_center = fixed_size[1] - fixed_col_min_from_center
        
        # create new bounding box depending on input arguments
        bbox_row_min = nuclei_center[0] - fixed_row_min_from_center
        bbox_row_max = nuclei_center[0] + fixed_row_max_from_center
        bbox_col_min = nuclei_center[1] - fixed_col_min_from_center
        bbox_col_max = nuclei_center[1] + fixed_col_max_from_center

        # limit the box to within the original image (see commented code below for shifting the box instead)
        bbox_row_min = max( 0, bbox_row_min )
        bbox_row_max = min( height, bbox_row_max )
        bbox_col_min = max( 0, bbox_col_min )
        bbox_col_max = min( width, bbox_col_max )

        ## shift fixed bounding box if exceeds dimensions of the original image
        #if bbox_row_min < 0:
        #    bbox_row_min = 0
        #    bbox_row_max = bbox_row_min + fixed_size[0]
        #if bbox_row_max >= height:
        #    bbox_row_max = height - 1
        #    bbox_row_min = bbox_row_max - fixed_size[0]
        #if bbox_col_min < 0:
        #    bbox_col_min = 0
        #    bbox_col_max = bbox_col_min + fixed_size[1]
        #if bbox_col_max >= width:
        #    bbox_col_max = width - 1
        #    bbox_col_min = bbox_col_max - fixed_size[1]
        #if bbox_row_min >= height or bbox_row_max < 0 or bbox_col_min >= width or bbox_col_max < 0:
        #    raise Exception( 'Invalid fixed_size dimensions.' )

        return bbox_row_min, bbox_row_max, bbox_col_min, bbox_col_max


if __name__ == '__main__':

    # Below are example usages of utils
    
    image_path = '../../data/raw/stage1_train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e/images/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png'
    mask_path = '../../data/raw/stage1_train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e/masks/0fe691c27c3dcf767bc22539e10c840f894270c82fc60c8f0d81ee9e7c5b9509.png'
    extractor = ExtractROI()

    # Set the inputs according to paths (also possible to just give the class the actual matrices)
    extractor.set_image_with_png( image_path )
    extractor.set_mask_with_png( mask_path )
    
    # Extract the ROI with padding around nuclei pixels
    extractor.extract_roi( padding=3 )
    
    # Extract the ROI on a fixed size box around nuclei center
    #extractor.extract_roi( fixed_size=[ 40, 40 ] )

    # Example of the resulting ROI matrix (pixel values of bounding box around nuclei)
    #print( extractor.get_roi() )

    # Save the ROI as an image
    output_path = './example.png'
    extractor.save_roi_as_image( output_path )
    print( 'ROI image saved as ' + str( output_path ) )

