
import sys, os
import random
import numpy
import scipy
import sklearn
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import ast, configparser
import gc, shutil
import time

sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from analysis import moving_window, image_segmentation, kaggle_evaluation, kaggle_evaluation_multi, hog_classifier
from preprocess import image_processing, generate_centers
from utils import kaggle_reader, kaggle_io


def main():
    

    ########################
    # general options
    #

    # adjustable parameters
    pixel_clf_pkl = '../config/submission_2.cfg' # config file
    #pixel_clf_pkl = '../config/nn_final.cfg' # config file
    #pixel_clf_pkl = '../config/nn_test.cfg'
    #pixel_clf_pkl = '../config/nn_adam_20_20_20_20_20_center3px_10x10win_graytrimnorm.cfg'
    #pixel_clf_pkl = '../config/svm_gkhp_tuning_center3px_10x10win_graytrimnorm.cfg'
    segmentation_method = 'watershed' # connected, watershed, felzenszwalb, randomwalker

    # paths to input and output directories
    input_test_images = '../data/split/test'

    # test only one image
    only_evaluate_this_image = None # set to None to test all images, otherwise set an image index number (e.g. 1, 2, or 201) to test only that one image
    #only_evaluate_this_image = 12


    ########################
    # technical options
    #

    # padding for moving window
    padding_type = 'constant' # 'reflect', 'constant', 'symmetric' or other inputs for the mode to numpy.pad

    # calculation of markers for watershed and randomwalker
    # (set any of these factors to None to skip step)
    marker_erosion_type = 'normal' # either None, 'ultimate', or 'normal'

    # settings for ultimate erosion of markers
    ultimate_dilation = 4 # default dilation
    variable_dilation = False # whether to use a different dilation depending on nuclei size
    larger_dilation = 5 # larger dilation to prevent oversplitting
    nucleus_size_threshold_for_larger_dilation = 600 # nuclei size threshold on when to switch to the larger dilation
    largest_dilation = 7 # largest dilation to prevent oversplitting
    nucleus_size_threshold_for_largest_dilation = 1000 # nuclei size threshold on when to switch to the largest dilation

    # settings for normal erosion of markers
    erosion_factor = 9 # 9 default value describing extent of erosion
    variable_erosion_factor = True # whether to use a different erosion factor depending on nuclei size
    medium_erosion_factor = 5 # 5 erosion factor when the nuclei size is smaller than the medium threshold
    nucleus_size_threshold_for_medium_erosion = 800 # 800 use medium_erosion_factor when the nucleus is smaller than this value
    small_erosion_factor = 1 # 1 erosion factor when the nuclei size is smaller than the small threshold
    nucleus_size_threshold_for_small_erosion = 200 # 200 use small_erosion_factor when the nucleus is smaller than this value

    # fill holes in nuclei_mask
    fill_mask_holes = True

    # preprocessing before segmentation
    preprocess = None # None, 'closing_opening', 'dilation_erosion', 'opening_closing', 'erosion_dilation'
    preprocess_options = None # e.g. [ 3, 3 ]
    force_predicted_nuclei_mask = True # whether to force final prediction to match nuclei prediction

    # lower and upper bounds of the size for nuclei predictions
    lower_bound_pixels = 50 # i.e. predicted nucleus must have more than N pixels
    upper_bound_fraction = 1.0/10.0 # i.e. predicted nucleus must be smaller than X fraction of the whole image
    
    # miscellaneous options
    thresholds = numpy.arange( 0.5, 1.0, 0.05 )
    nms_threshold = 1

    pixel_clf = 'nn_20x20_center1px_' + segmentation_method + "padding_" + padding_type + '_erosion=' + marker_erosion_type + str(ultimate_dilation) # this is just for naming the output directory
    #output_figure_dir = '../data/figures_' + pixel_clf + '_' + segmentation_method
    output_figure_dir = '../data/submission_2'


    ########################
    # technical options
    #

    # adjustable parameters for postprocessing (set post_iter=0 to turn off)
    post_iter = 0 # set to 0 to turn off
    min_distance = 1 # 5
    post_threshold = 0.1 # 1.05
    max_angle = -1.5 # -1.5
    calc_distance_convex_points = 3 # 3
    pen_ratio_circularity = 1 # 1
    scoring = 'distance' # convexity, distance, intensity, point_angle, circularity
    




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
    prediction_counts = numpy.zeros( ( len( thresholds ), 4 ) )
    del thresholds

    # track progress
    i_image = 0
    n_images = len( imdirs )
    removed_images = 0

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

        # only consider
        if not only_evaluate_this_image is None:
            if not i_image == only_evaluate_this_image:
                continue

        # read files
        image_array = scipy.misc.imread( os.path.join( input_test_images, image_id, 'images', image_id + '.png') )
        expected_masks_directory = os.path.join( input_test_images, image_id, 'masks' )

        # store height and width of images
        height = len( image_array )
        width = len( image_array[0] )
       
        # miscellaneous options
        lim_segment_size = [ lower_bound_pixels, image_array.size * upper_bound_fraction ]

        # extracting pixels in nuclei
        classifier = sklearn.externals.joblib.load( ast.literal_eval( config.get( 'paths', 'classifier_pkl' ) ) )
        feature_method = ast.literal_eval( config.get( 'features', 'method' ) )
        feature_options = ast.literal_eval( config.get( 'features', 'feature_options' ) )
        image_options = ast.literal_eval( config.get( 'features', 'image_processing_options' ) )
        window_sizes = ast.literal_eval( config.get( 'moving_window', 'window_size' ) )
        step_sizes = ast.literal_eval( config.get( 'moving_window', 'step_size' ) )
        boxes, _ = moving_window.run_moving_window(classifier, image_array, feature_method, feature_options, image_options, window_sizes, step_sizes, 2, padding=padding_type )
        start = time.time()
        replacement = [1, 1]
        nuclei_mask = moving_window.boxes_to_mask( boxes, image_array, replacement )
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

        # run segmentation on the nuclei_mask
        print( 'calculating segmentation' )

        # determine whether to give the segmentation code the actual image or the binary mask
        if segmentation_method == 'randomwalker' or segmentation_method == 'felzenszwalb':
            image = image_array
        else:
            image = nuclei_mask

        # miscellaneous segmentation options
        center_image = nuclei_mask
        border_mask = None
        segment_array = nuclei_mask

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

        # ultimate erosion of markers
        elif marker_erosion_type == 'ultimate':

            # create a segmentation based on connected algorithm
            labels = skimage.measure.label( nuclei_mask )
            connected_segments = [ 1*( labels == value ) for value in range( 1, numpy.amax( labels )+1, 1 ) ]
            centers = numpy.zeros( ( height, width ) )
            del labels
        
            # consider every segment separately if dilation should be variable
            if variable_dilation:

                for segment in connected_segments:

                    # set a different dilation depending on the size of the nuclei in the segment
                    dilation = ultimate_dilation
                    segment_size = numpy.sum( numpy.sum( segment ) )
                    if segment_size <= nucleus_size_threshold_for_larger_dilation:
                        dilation = larger_dilation
                    elif segment_size <= nucleus_size_threshold_for_largest_dilation:
                        dilation = largest_dilation
            
                    eroded_segment = image_segmentation.ultimate_erosion( nuclei_mask, dilation )
                    centers = numpy.maximum( centers, eroded_segment )
            
            # run erosion on the entire mask
            else:
                centers = generate_centers.ultimate_erosion( nuclei_mask, ultimate_dilation )
        
        # do not run erosion on the centers
        else:
            centers = nuclei_mask
        
        # actually run the segmentation code
        print("ultimate_erosion time =" + str(time.time()-start))
        start = time.time()
        segmentation = image_segmentation.apply_segmentation( 
                image=image,
                center_image=center_image,
                border_mask=border_mask,
                image_array=segment_array,
                type=segmentation_method,
                preprocess=preprocess,
                min_distance=min_distance,
                centers=centers,
                preprocess_options=preprocess_options,
                lim_segment_size=lim_segment_size,
                output_plot=False,
                post_iter=post_iter, 
                post_threshold=post_threshold, 
                max_angle=max_angle, 
                calc_distance_convex_points=calc_distance_convex_points,
                pen_ratio_circularity=pen_ratio_circularity,
                scoring=scoring )
        print("segmentation time =" + str(time.time()-start))
        del image
        del center_image
        del border_mask
        del segment_array
        del centers

        # debug segmentation
        print( "Number of segmentations: " + str( len( segmentation ) ) )

        # option to force the final prediction to match the nuclei mask
        if force_predicted_nuclei_mask and not numpy.sum( numpy.sum( nuclei_mask ) ) == 0 and not len( segmentation ) == 0: 
            final_prediction = segmentation * nuclei_mask
        elif len( segmentation ) == 0 or numpy.sum( numpy.sum( nuclei_mask ) ) == 0:
            final_prediction = []
        else: 
            final_prediction = segmentation

        # calculate scores (single)
        header_single = 'Single evaluation : ' + image_id
        evaluation_single = kaggle_evaluation.KaggleEvaluation()
        evaluation_single.set_predicted_masks( final_prediction )
        evaluation_single.set_expected_masks_from_directory( expected_masks_directory )
        evaluation_single.calculate_iou()
        evaluation_single.calculate_score_with_thresholds()
        table_single = evaluation_single.get_table()
        masks = evaluation_single.expected_masks
        score = evaluation_single.get_score()
        del evaluation_single
        gc.collect()

        # calculate scores (multi)
        header_cumulative = 'Cumulative evaluation : ' + str(i_image) + '/' + str(n_images)
        evaluation = kaggle_evaluation_multi.KaggleEvaluation()
        evaluation.prediction_counts = prediction_counts # save memory by only storing prediction_counts between image_id loops
        evaluation.set_predicted_masks( final_prediction )
        evaluation.set_expected_masks_from_directory( expected_masks_directory )
        masks_array = evaluation.expected_masks
        evaluation.calculate_iou()
        evaluation.calculate_score_with_thresholds()
        table_cumulative = evaluation.get_table()
        del evaluation # save memory by only storing prediction_counts between image_id loops
        gc.collect()

        # results
        output_text = header_single + '\n' + table_single + '\n\n' + header_cumulative + '\n' + table_cumulative + '\n'
        print( output_text ) 

        # prepare to plot
        fig = plt.figure()

        # prepare colors of masks
        col_list = numpy.linspace( 0.001, 1, 1000 )
        random.shuffle( col_list )
        
        # plot top-left, original image
        plt.subplot( 231 )
        plt.imshow( image_array )
        plt.xticks( [] )
        plt.yticks( [] )
        plt.title( 'Original Image\n(' + str(width) + 'x' + str(height) + ')' )
              
        # plot top-middle, true pixels
        plt.subplot( 232 )
        mask_matrix = numpy.zeros( ( height, width ), numpy.uint16 )
        for current_mask in masks_array:
            mask_matrix = numpy.maximum( mask_matrix, current_mask )
        plt.imshow( mask_matrix )
        plt.xticks( [] )
        plt.yticks( [] )
        plt.title( 'True Pixels' )

        # plot top-right, predicted nuclei pixels
        plt.subplot( 233 )
        plt.imshow( nuclei_mask )
        plt.xticks( [] )
        plt.yticks( [] )
        plt.title( 'Predicted Pixels' )

        # plot bottom-left, true mask segmentation
        plt.subplot( 234 )
        #plt.imshow( image_array )
        for i in range( len(masks) ):
            colormap = plt.cm.colors.ListedColormap( plt.cm.nipy_spectral( col_list[ i:i+1 ] ) )
            colormap.set_under( 'k', alpha=0 )
            plt.imshow( masks[i], cmap=colormap, alpha=1.0, clim=[0.5, 1] )
        plt.xticks( [] )
        plt.yticks( [] )
        plt.title( 'True Segment\n(n=' + str(len(masks)) + ')' )
 
        # plot bottom-middle, segmentation
        plt.subplot( 235 )
        #plt.imshow( image_array )
        try:
            for i in range( len(segmentation) ):
                colormap = plt.cm.colors.ListedColormap( plt.cm.nipy_spectral( col_list[ i:i+1 ] ) )
                colormap.set_under( 'k', alpha=0 )
                plt.imshow( segmentation[i], cmap=colormap, alpha=1.0, clim=[0.5, 1] )
        except:
            pass
        plt.xticks( [] )
        plt.yticks( [] )
        plt.title( 'Predicted Segment\n(n=' + str(len(segmentation)) + ')' )
        # place info on bottom of figure (as x-axis of bottom-middle)
        title = 'Classifier = ' + pixel_clf + ' ; Segmentation = ' + segmentation_method
        plt.xlabel( title )

        # plot bottom-right, final prediction
        plt.subplot( 236 )
        plt.imshow( image_array )
        try:
            for i in range( len(final_prediction) ):
                colormap = plt.cm.colors.ListedColormap( plt.cm.nipy_spectral( col_list[ i:i+1 ] ) )
                colormap.set_under( 'k', alpha=0 )
                plt.imshow( final_prediction[i], cmap=colormap, alpha=1.0, clim=[0.5, 1] )
        except:
            pass
        plt.xticks( [] )
        plt.yticks( [] )
        plt.title( 'Final Prediction \n(score: %.2f)' % score )

        # save image
        output_image_path = output_figure_dir + '/' + '%04d_' % i_image + image_id + '.png'
        plt.savefig( output_image_path )
        print( 'Image saved to ' + output_image_path )

        # save results to file
        output_text_path = output_figure_dir + '/' + '%04d_' % i_image + image_id + '.txt'
        output_text_file = open( output_text_path , 'w' )
        output_text_file.write( output_text )
        output_text_file.close()
        print( 'Results saved to ' + output_text_path )

        # save results to csv
        try:
            output_csv = [ kaggle_io.mask_matrix_to_kaggle_format( x, image_id ) for x in final_prediction ]
        except:
            output_csv = []
        with open( output_csv_path, "a" ) as csvfile:
            csv_writer = csv.writer( csvfile, delimiter=',', quotechar="'" )
            csv_writer.writerows( output_csv )
        print( 'Predictions saved to ' + output_csv_path )

        # cleanup for memory purposes
        plt.close('all')
        del image_array
        del expected_masks_directory
        del nuclei_mask
        del segmentation
        del final_prediction
        del fig
        del col_list
        del masks
        del height
        del width
        del score
        del colormap
        del masks_array
        del mask_matrix
        del output_text
        del header_single
        del table_single
        del header_cumulative
        del table_cumulative
        del output_text_path
        del output_text_file
        gc.collect()
        print()

if __name__ == '__main__':
    full_start = time.time()
    main()
    print("Total time elapsed: " + str(time.time()- full_start))

    # import profile
    # import pstats
    # profile.run("main()", "profstats")
    # p = pstats.Stats('profstats')
    # p.sort_stats('cumtime')
    # p.print_stats()
