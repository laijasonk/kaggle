
""" Batch script to extract and save ROI images from split directories

NOTE: Run after a split script (e.g. preprocessing_script_split_raw.py)
NOTE: Change the variables on bottom of script before running

""" 

import sys
import os
import random
import numpy
import pandas

sys.path.append( os.path.join( '..', 'src' ) ) # point to project src/
from preprocess import extract_roi, generate_negative, image_processing
from utils import kaggle_io


class extract_split_roi( object ):

    def __init__( self, split_path, roi_path, padding=0, fixed_size=None, fixed_1x1_ratio=False, negative_size=None,
                  number_of_neg_per_pos=1, check_neg_against_pos=False, neg_buffer_check_size=0,
                  preprocess_image_options={} ):

        """ The class constructor

        :param split_path: Path to the input split directory (e.g ../data/split)
        :param roi_path: Path to the output roi directory (e.g. ../data/roi)
        :param padding: Extra padding space in pixels around the region of interest 
        :param fixed_size: The fixed size of the ROI, format=[ height, width ]
        :param fixed_1x1_ratio: Boolean defining whether to fix the aspect ratio to 1:1
        :param number_of_neg_per_pos: The number of negative images to generate for every found positive image
        :param check_neg_against_pos: bool, indicates whether to read a list of no-go locations for negative samples
        :param neg_buffer_check_size: size of minimum offset between the center of negative samples and known positive examples (contained in roi_center_list)
        :param preprocess_image_options: dict, options for preprocessing, all possible options as key ('rgb2gray',
                                        'trim', 'norm') and parameters for the processing options are passed as values:
                                        'rgb2gray': None (no parameters to pass)
                                        'trim': [left_trim, right_rim], left_trim and right_trim are percentiles to trim
                                        'norm': None (no parameters) rescale intensity to fill the full possible range
        :return: None

        """

        # check if only one criterion is specified
        if ( ( not padding == 0 and not fixed_size == None ) or 
                ( not padding == 0 and not fixed_1x1_ratio == False ) or 
                ( not fixed_size == None and not fixed_1x1_ratio == False ) ):
            raise Exception( 'Only one type of criteria for extracting ROI can be passed as an argument simultaneously.' )

        self.padding = padding
        self.fixed_size = fixed_size
        self.fixed_1x1_ratio = fixed_1x1_ratio
        self.split_path = split_path
        self.roi_path = roi_path

        self.negative_size = negative_size
        self.number_of_neg_per_pos = number_of_neg_per_pos
        self.check_neg_against_pos = check_neg_against_pos
        self.neg_buffer_check_size = neg_buffer_check_size

        self.preprocess_image_options = preprocess_image_options

        return


    def run( self ):
        
        """ Run the main code to extract ROI from each type of directory
        in the split path

        :return: None

        """

        # call the functions to create ROI from split directories
        print( 'Extracting ROI from train' )
        self.create_roi_from_split( 'train' )

        print( 'Extracting ROI from test' )
        self.create_roi_from_split( 'test' )

        #print( 'Extracting ROI from validate' )
        #self.create_roi_from_split( 'validate' )
        
        return


    def create_roi_from_split( self, dirname ):

        """ Extracts the regions of interest from the split directory
        and stores it to the specified in the ROI path

        :param dirname: Name of the directory (e.g 'train', 'validate', 'test')
        :return: None

        """
        
        # make sure directories exist
        if not os.path.exists( os.path.join( self.roi_path, dirname ) ):
            os.makedirs( os.path.join( self.roi_path, dirname ) )

        # information to track progress
        image_counter = 0
        total_images = len( os.listdir( os.path.join( self.split_path, dirname ) ) )

        # prepare extractor with mask path and corresponding image
        out_table = []
        out_dataframe = []
        positive = extract_roi.ExtractROI()
        negative = generate_negative.GenerateNegative()

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
            # preprocess the image according to options passed in
            image_matrix = image_processing.process_image(image_matrix, self.preprocess_image_options)
            # set the image to both extractors
            positive.set_image_with_matrix( image_matrix )
            negative.set_image_with_matrix( image_matrix )

            # store some general information
            image_height, image_width = positive.image.shape[ 0:2 ][ 0:2 ]

            # identify all masks for this specific image
            for mask_file in os.listdir( os.path.join( self.split_path, dirname, image_id, 'masks' ) ):

                # extract the mask_id from the path (no extension)
                mask_id = os.path.basename( os.path.splitext( mask_file )[0] )

                # run the extractor
                pos_name, pos_mask_name = self.positive_handling( positive, dirname, image_id, mask_id )

                # extract some information from positive extractor
                pos_matrix = positive.get_roi()
                pos_height, pos_width = pos_matrix.shape[ 0:2 ]
                pos_center = positive.roi_center
                 
                # store the positive info to table and dataframe
                out_table.append( [ pos_name, image_id, pos_height, pos_width, image_height, image_width, pos_center[0], pos_center[1], '1' ] )
                out_dataframe.append( [ pos_name, image_id, pos_height, pos_width, image_height, image_width, pos_center[0], pos_center[1], '1', pos_matrix ] )

                # store positive centers
                table_roi_center.append( pos_center )

            # creation of "roi center" list
            center_header = 'center_col  center_row'
            table_path = os.path.join( self.roi_path, dirname, 'roi_center_list_for_' + image_id + '.txt' )
            numpy.savetxt( table_path, table_roi_center, fmt='%s', header=center_header )
            roi_center_list = numpy.loadtxt( table_path, skiprows=1 )

            # identify all masks for this specific image
            count = 0
            for mask_file in os.listdir( os.path.join( self.split_path, dirname, image_id, 'masks' ) ):

                # repeat until reached desirable number of negatives per positive
                for repeat in range( self.number_of_neg_per_pos ):

                    # run the negative generator
                    count += 1
                    neg_name = self.negative_handling( negative, dirname, image_id, mask_id, '_' + str( count ).zfill( 5 ), roi_center_list )

                    # extract information from negative generator
                    neg_matrix = negative.get_negative()
                    neg_height, neg_width = neg_matrix.shape[ 0:2 ]
                    neg_center = [ negative.negative_row, negative.negative_col ]

                    # store the negative info to table and dataframe
                    out_table.append( [ neg_name, image_id, neg_height, neg_width, image_height, image_width, neg_center[0], neg_center[1], '0' ] )
                    out_dataframe.append( [ neg_name, image_id, neg_height, neg_width, image_height, image_width, neg_center[0], neg_center[1], '0', neg_matrix ] )

        # table information
        table_header = 'filename, parent_image_id, extract_height, extract_width, image_height, image_width, index_row, index_col, is_positive'
        table_path = os.path.join( self.roi_path, dirname + '_table.csv' )

        # dataframe information
        dataframe_header = [ 'filename', 'parent_image_id', 'extract_height', 'extract_width', 'image_height', 'image_width', 'index_row', 'index_col', 'is_positive', 'image_matrix' ]
        dataframe_path = os.path.join( self.roi_path, dirname + '_dataframe.pkl' )

        # save table
        print( 'Saving table as CSV' )
        numpy.savetxt( table_path, out_table, fmt='%s', delimiter=',', header=table_header )

        # save dataframe
        print( 'Saving dataframe as PKL' )
        save_dataframe = pandas.DataFrame( out_dataframe, columns=dataframe_header )
        save_dataframe.to_pickle( dataframe_path )

        return


    def positive_handling( self, positive_obj, dirname, image_id, mask_id ):

        """ Save the positive roi set to file and return the path of
        these new images

        :param positive_obj: The call to the ExtractROI class
        :param dirname: Name of the directory (e.g 'train', 'test')
        :param image_id: The hash-like id of the parent image
        :param mask_id: The hash-like id of the mask image
        :return: None

        """
        
        # set the mask before extraction
        mask_png = os.path.join( self.split_path, dirname, image_id, 'masks', mask_id + '.png' )
        positive_obj.set_mask_with_png( mask_png )

        # run the actual extraction
        positive_obj.extract_roi( 
                padding=self.padding,
                fixed_size=self.fixed_size, 
                fixed_1x1_ratio=self.fixed_1x1_ratio )

        # define the names of the output files
        out_roi_name = str( 'pos_' + image_id + '_' + mask_id + '.npy' )
        out_mask_name = str( 'mask_' + image_id + '_' + mask_id + '.npy' )
        pos_path = os.path.join( self.roi_path, dirname, out_roi_name )
        pos_mask_path = os.path.join( self.roi_path, dirname, out_mask_name )

        # save roi and roi_mask to appropriate directory
        positive_obj.save_roi_as_npy( pos_path )
        positive_obj.save_roi_mask_as_npy( pos_mask_path )

        return os.path.basename( pos_path ), os.path.basename( pos_mask_path )


    def negative_handling( self, negative_obj, dirname, image_id, mask_id, suffix='', roi_center_list=None ):

        """ Save the negative roi set to file and return the path of
        these new images

        :param positive_obj: The call to the ExtractROI class
        :param dirname: Name of the directory (e.g 'train', 'test')
        :param image_id: The hash-like id of the parent image
        :param mask_id: The hash-like id of the mask image
        :param roi_center_list: A list of roi centers
        :return: None

        """

        # set the images before running
        image_png = os.path.join( self.split_path, dirname, image_id, 'images', str( image_id ) + '.png' )

        # run the actual negative generation
        negative_obj.set_negative_box_size( negative_size )
        negative_obj.set_random_seed( random.getrandbits( 100 ) )
        negative_obj.run( roi_center_list, self.neg_buffer_check_size )

        # save negative to appropriate directory
        out_neg_name = 'neg_' + image_id + str( suffix ) + '.npy'
        neg_path = os.path.join( self.roi_path, dirname, out_neg_name )
        negative_obj.save_negative_as_npy( neg_path )

        return os.path.basename( neg_path )


if __name__ == '__main__':

    # define random seed for reproducibility
    random.seed( 1 )

    # choose paths
    split_path = os.path.join( '..', 'data', 'small_split' ) # 'split' or 'small_split'
    roi_path = os.path.join( '..', 'data', 'small_roi' ) # 'roi' or 'small_roi'

    # only change one criterion for ROI generation
    padding = 0 # default=0, change to non-zero to add padding to extracted ROI 
    fixed_size = [ 64, 64 ] # default=None, change to [W,H] to extract specific sized ROI
    fixed_1x1_ratio = False # default=False, change to True to extract 1x1 aspect ratio ROI
    
    # negative information
    negative_size = fixed_size # change to [W,H] to generate example negative
    number_of_neg_per_pos = 1
    check_neg_against_pos = True
    neg_buffer_check_size = 0

    # run script
    main = extract_split_roi( 
            split_path=split_path, 
            roi_path=roi_path, 
            padding=padding, 
            fixed_size=fixed_size, 
            fixed_1x1_ratio=fixed_1x1_ratio, 
            negative_size=negative_size, 
            number_of_neg_per_pos=number_of_neg_per_pos,
            check_neg_against_pos=check_neg_against_pos,
            neg_buffer_check_size=neg_buffer_check_size,
            preprocess_image_options={} )
            #preprocess_image_options={'rgb2gray': None, 'trim': [1, 99], 'norm': None} )
    main.run()

