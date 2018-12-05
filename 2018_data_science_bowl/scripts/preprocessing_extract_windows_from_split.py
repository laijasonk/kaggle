
""" Batch script to extract windows from split directories

NOTE: Run after a split script (e.g. preprocessing_script_split_raw.py)
NOTE: Change the variables on bottom of script before running

""" 

import sys
import os
import random
import numpy
import pandas
import scipy.misc

sys.path.append( os.path.join( '..', 'src' ) ) # point to project src/
from utils import kaggle_io, kaggle_reader


class extract_split_windows( object ):

    def __init__( self, split_path, windows_path, windows_size=[ 64, 64 ], windows_per_edge=20, window_center_nuclei_distance=2 ):

        """ The class constructor

        :return: None

        """

        self.split_path = split_path
        self.windows_path = windows_path

        self.windows_size = windows_size
        self.windows_per_edge = windows_per_edge
        
        self.window_center_nuclei_distance = window_center_nuclei_distance

        return


    def run( self, dirs=["train", "test"] ):
        
        """ Run the main code to extract windows from each type of directory
        in the split path

        :return: None

        """

        # call the functions to create windows from split directories
        for i in dirs:
            print('Extracting windows from ' + i)
            self.create_windows_from_split(i)
        
        return


    def create_windows_from_split( self, dirname ):

        """ Extracts the windows from the split directory and stores it
        to the specified in the windows path

        :param dirname: Name of the directory (e.g 'train', 'validate', 'test')
        :return: None

        """
        
        # make sure directories exist
        if not os.path.exists( os.path.join( self.windows_path, dirname ) ):
            os.makedirs( os.path.join( self.windows_path, dirname ) )

        # information to track progress
        image_counter = 0
        total_images = len( os.listdir( os.path.join( self.split_path, dirname ) ) )

        # prepare extractor with mask path and corresponding image
        out_table = []
        out_dataframe = []

        for image_id in os.listdir( os.path.join( self.split_path, dirname ) ):

            # table to save roi centers for the image
            table_roi_center = []

            # show the current image_id to track progress
            image_counter += 1
            print( str( image_counter ) + '/' + str( total_images ) + ' : ' + image_id )

            # set the image, this is the common image for all its masks
            image_png = os.path.join( self.split_path, dirname, image_id, 'images', str( image_id + '.png' ) )

            # load the image into a numpy array
            image_matrix = kaggle_io.png_to_pixel_matrix(image_png)

            # get some basic information
            image_height, image_width, _ = image_matrix.shape

            # set the mask directory containing all the masks
            masks_directory = os.path.join( self.split_path, dirname, image_id, 'masks' )

            # load the masks into a combined numpy array
            masks_array = kaggle_reader.get_masks_from_directory( masks_directory )
            mask_matrix = numpy.zeros( ( image_height, image_width ), numpy.uint16 )
            for current_mask in masks_array:
                mask_matrix = numpy.maximum( mask_matrix, current_mask )

            ## preprocess the image according to options passed in
            #image_matrix = image_processing.process_image( image_matrix, self.preprocess_image_options )

            # calculate the step sizes
            row_step_size = int( ( image_height - self.windows_size[ 0 ] ) / self.windows_per_edge )
            col_step_size = int( ( image_width - self.windows_size[ 1 ] ) / self.windows_per_edge )

            # consider every window
            for row_min in range( 0, image_height - self.windows_size[0], row_step_size ):
                for col_min in range( 0, image_width - self.windows_size[1], col_step_size ):

                    # define bounding box
                    bbox_row_min = max( row_min, 0 )
                    bbox_row_max = min( row_min + self.windows_size[ 0 ], image_height )
                    bbox_col_min = max( col_min, 0 )
                    bbox_col_max = min( col_min + self.windows_size[ 1 ], image_width )
                    
                    # extract ROI from image based on mask
                    window = image_matrix[ bbox_row_min:bbox_row_max, bbox_col_min:bbox_col_max ]
                    
                    # general information
                    window_height, window_width = window.shape[ 0:2 ]
                    window_center = [ int( numpy.floor( ( bbox_row_max + bbox_row_min ) / 2 ) ), int( numpy.floor( ( bbox_col_max + bbox_col_min ) / 2 ) ) ]

                    # whether center pixel is part of a nuclei
                    is_nuclei = '0'
                    suffix = 'neg_'
                    row_lower_range = int( window_center[0] - self.window_center_nuclei_distance )
                    row_upper_range = int( window_center[0] + self.window_center_nuclei_distance ) + 1
                    col_lower_range = int( window_center[1] - self.window_center_nuclei_distance )
                    col_upper_range = int( window_center[1] + self.window_center_nuclei_distance ) + 1
                    for center_row in range( row_lower_range, row_upper_range ):
                        for center_col in range( col_lower_range, col_upper_range ):
                            try:
                                if mask_matrix[ center_row ][ center_col ] == 1:
                                    is_nuclei = '1'
                                    suffix = 'pos_'
                            except:
                                print( 'Out-of-bounds: ' + row.image_id )
                                print( 'Center point (' + str( center_row ) + ', ' + str( center_col ) + ') is not within matrix size (' + str( row.image_height ) + ', ' + str( row.image_width ) + ')' )

                    # filename for window
                    window_name = str( suffix + image_id + '_' + str( row_min ) + '_' + str( col_min ) ) 

                    ## prepare output directory 
                    #if not os.path.isdir( os.path.join( self.windows_path, dirname ) ):
                    #    os.makedirs( os.path.join( self.windows_path, dirname ) )
                    ## save as npy
                    #numpy.save( os.path.join( self.windows_path, dirname, window_name + '.npy' ), window )
                    ## save as png
                    #scipy.misc.imsave( os.path.join( self.windows_path, dirname, window_name + '.png' ), window )

                    # store the window info to table and dataframe
                    out_table.append( [ window_name, image_id, window_height, window_width, image_height, image_width, window_center[0], window_center[1], is_nuclei ] )
                    out_dataframe.append( [ window_name, image_id, window_height, window_width, image_height, image_width, window_center[0], window_center[1], is_nuclei, window ] )

        # table information
        table_header = 'filename, parent_image_id, extract_height, extract_width, image_height, image_width, index_row, index_col, is_positive'
        table_path = os.path.join( self.windows_path, dirname + '_table.csv' )

        # dataframe information
        dataframe_header = [ 'filename', 'parent_image_id', 'extract_height', 'extract_width', 'image_height', 'image_width', 'index_row', 'index_col', 'is_positive', 'image_matrix' ]
        #dataframe_path = os.path.join( self.windows_path, dirname + '_dataframe.h5' )

        # save table
        print( 'Saving table as CSV' )
        numpy.savetxt( table_path, out_table, fmt='%s', delimiter=',', header=table_header )

        # save dataframe
        print( 'Saving dataframe as PKL' )
        dataframe_path = os.path.join( self.windows_path, dirname + '_dataframe.pkl' )
        save_dataframe = pandas.DataFrame( out_dataframe, columns=dataframe_header )
        save_dataframe.to_pickle( dataframe_path )
        
        ## save dataframe (HDF version)
        #print( 'Saving dataframe as H5' )
        #save_dataframe = pandas.DataFrame( out_dataframe, columns=dataframe_header )
        #with pandas.HDFStore( dataframe_path ) as store:
        #    store[ 'data' ] = save_dataframe

        return


if __name__ == '__main__':

    # define random seed for reproducibility
    random.seed( 1 )

    # choose paths
    split_path = os.path.join( '..', 'data', 'split' ) # 'split' or 'small_split'
    windows_path = os.path.join( '..', 'data', 'windows_20x20_center1px' ) # 'windows' or 'small_windows'

    # size of the window: [ R, C ]
    windows_size = [ 64, 64 ] 

    # number of windows per edge (e.g. 10 windows means 20x20 windows will be spliced from the parent image (i.e. 400 total)
    windows_per_edge = 20

    # pixels around the center point (e.g. 2px means all pixel from (center-2) to (center+2), which totals 5px around center at each direction)
    window_center_nuclei_distance = 0

    # run script
    main = extract_split_windows( 
            split_path=split_path, 
            windows_path=windows_path, 
            windows_size=windows_size, 
            windows_per_edge=windows_per_edge,
            window_center_nuclei_distance=window_center_nuclei_distance )
            #preprocess_image_options={'rgb2gray': None, 'trim': [1, 99], 'norm': None} )

    main.run()

