import os
import time
import ast
import csv
import configparser
import gc
import shutil
import sklearn
import numpy
import scipy.ndimage
import skimage
import skimage.morphology
import skimage.measure
import matplotlib.pyplot as plt

import final_functions

if __name__ == '__main__':

    ########################
    # general options
    #

    # adjustable parameters
    pixel_clf_pkl = '../config/submission_1.cfg' # config file
    segmentation_method = 'watershed'

    # paths to input and output directories
    input_test_images = '../data/raw/stage2_test3'

    ########################
    # technical options
    #

    # padding for moving window
    padding_type = 'constant' # 'reflect', 'constant', 'symmetric' or other inputs for the mode to numpy.pad

    # calculation of markers for watershed
    # (set any of these factors to None to skip step)
    marker_erosion_type = 'normal' # either None or 'normal'

    # settings for normal erosion of markers
    erosion_factor = 9 # 9 default value describing extent of erosion
    variable_erosion_factor = True # whether to use a different erosion factor depending on nuclei size
    medium_erosion_factor = 5 # 5 erosion factor when the nuclei size is smaller than the medium threshold
    nucleus_size_threshold_for_medium_erosion = 800 # 800 use medium_erosion_factor when the nucleus is smaller than this value
    small_erosion_factor = 1 # 1 erosion factor when the nuclei size is smaller than the small threshold
    nucleus_size_threshold_for_small_erosion = 200 # 200 use small_erosion_factor when the nucleus is smaller than this value

    # fill holes in nuclei_mask
    fill_mask_holes = True

    force_predicted_nuclei_mask = True # whether to force final prediction to match nuclei prediction

    # lower and upper bounds of the size for nuclei predictions
    lower_bound_pixels = 50 # i.e. predicted nucleus must have more than N pixels
    upper_bound_fraction = 1.0/10.0 # i.e. predicted nucleus must be smaller than X fraction of the whole image

    output_figure_dir = '../data/submission_1.3_001'
    start_from = 1





    ########################
    # evaluation code below
    #

    # consider all images in input directory
    imdirs = [ d for d in os.listdir( input_test_images ) if os.path.isdir( os.path.join( input_test_images, d ) ) ]

    # removes old figure directory in preparation of new figures
    if os.path.exists( output_figure_dir ):
        shutil.rmtree( output_figure_dir )
    os.makedirs( output_figure_dir )

    # initialize various objects
    config = configparser.ConfigParser( allow_no_value=True )
    config.read( pixel_clf_pkl )

    # track progress
    i_image = 0
    n_images = len( imdirs )

    # create csv file
    output_csv_path = output_figure_dir + '/' + 'prediction.csv'
    with open( output_csv_path, "w" ) as csvfile:
        csv_writer = csv.writer( csvfile, delimiter=',', quotechar='"' )
        csv_writer.writerow( [ "ImageId", "EncodedPixels" ] )

    # consider every image in the directory
    for image_id in imdirs:

        # print progress
        i_image += 1
        print( 'Image ' + str( i_image ) + '/'+ str( n_images ) + ' : ' + image_id )
        if i_image < start_from:
            continue

        try:
            # read files
            image_array = scipy.misc.imread( os.path.join( input_test_images, image_id, 'images', image_id + '.png') )

            # store height and width of images
            height = len( image_array )
            width = len( image_array[0] )

            # miscellaneous options
            lim_segment_size = [ lower_bound_pixels, image_array.size * upper_bound_fraction ]

            # extracting pixels in nuclei
            classifier = sklearn.externals.joblib.load( ast.literal_eval( config.get( 'paths', 'classifier_pkl' ) ) )
            image_options = ast.literal_eval( config.get( 'features', 'image_processing_options' ) )
            window_sizes = ast.literal_eval( config.get( 'moving_window', 'window_size' ) )
            step_sizes = ast.literal_eval( config.get( 'moving_window', 'step_size' ) )
            feature_method = ast.literal_eval( config.get( 'features', 'method' ) )
            feature_options = ast.literal_eval( config.get( 'features', 'feature_options' ) )
            boxes = final_functions.run_moving_window( classifier, image_array, image_options, window_sizes, step_sizes, padding=padding_type, feature_method=feature_method, feature_options=feature_options )
            start = time.time()
            replacement = [1, 1]
            nuclei_mask = final_functions.boxes_to_mask( boxes, image_array, replacement )
            print("boxes_to_mask time =" + str(time.time()-start))
            start = time.time()
            del boxes
            gc.collect()
            print("garbage collect time =" + str(time.time()-start))
            start = time.time()

            # process the nuclei prediction
            if fill_mask_holes:
                nuclei_mask = scipy.ndimage.binary_fill_holes( nuclei_mask )
            print("fill_mask_holes time =" + str(time.time()-start))
            start = time.time()

            # determine which type of marker info to give to the segmentation code
            # normal erosion of markers
            if marker_erosion_type == 'normal':

                # create a segmentation based on connected algorithm
                labels = skimage.measure.label( nuclei_mask )
                connected_segments = [ 1*( labels == value ) for value in range( 1, numpy.amax( labels )+1, 1 ) ]
                centers = numpy.zeros( ( height, width ) )
                del labels

                if variable_erosion_factor:

                    # consider every segment separately
                    for segment in connected_segments:

                        # set a different erosion factor depending on the size of the nuclei in the segment
                        segment_size = numpy.sum( numpy.sum( segment ) )
                        factor = erosion_factor
                        if segment_size <= nucleus_size_threshold_for_small_erosion:
                            factor = small_erosion_factor
                        elif segment_size <= nucleus_size_threshold_for_medium_erosion:
                            factor = medium_erosion_factor

                        # keep decreasing erosion factor until a nucleus is present (erosion_nucleus_size is not 0)
                        erosion_nucleus_size = 0
                        adjustment = 0
                        while erosion_nucleus_size == 0 and adjustment <= factor:
                            eroded_segment = skimage.morphology.erosion( segment, skimage.morphology.disk( factor - adjustment ) )
                            erosion_nucleus_size = numpy.sum( numpy.sum( eroded_segment ) )
                            adjustment += 1
                        centers = numpy.maximum( centers, eroded_segment )

                # run erosion on the entire mask
                else:
                    centers = skimage.morphology.erosion( nuclei_mask, skimage.morphology.disk( erosion_factor ) )

            # do not run erosion on the centers
            else:
                centers = nuclei_mask

            # actually run the segmentation code
            print("erosion time =" + str(time.time()-start))
            start = time.time()
            segmentation = final_functions.apply_segmentation(
                    image=nuclei_mask,
                    type=segmentation_method,
                    centers=centers,
                    lim_segment_size=lim_segment_size)
            print("segmentation time =" + str(time.time()-start))
            del centers

            # debug segmentation
            print( "Number of segmentations: " + str( len( segmentation ) ) )

            # save results to csv
            try:
                output_csv = [ final_functions.mask_matrix_to_kaggle_format( x, image_id ) for x in segmentation ]
            except:
                output_csv = []
            with open( output_csv_path, "a" ) as csvfile:
                csv_writer = csv.writer( csvfile, delimiter=',', quotechar="'" )
                csv_writer.writerows( output_csv )
            print( 'Predictions saved to ' + output_csv_path )

            # cleanup for memory purposes
            plt.close('all')
            del image_array
            del nuclei_mask
            del segmentation
            del height
            del width
            gc.collect()
            print()
        except:
            print( 'IMAGE FAILED' )
            gc.collect()
