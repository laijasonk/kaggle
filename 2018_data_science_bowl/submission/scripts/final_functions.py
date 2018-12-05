import os
import time
import math
import numpy
import pandas
import scipy
import scipy.misc
import scipy.ndimage.filters
import sklearn
import sklearn.neural_network
import skimage.feature
import skimage.color
import skimage.io

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
            image_matrix = png_to_pixel_matrix(image_png)

            # get some basic information
            image_height, image_width, _ = image_matrix.shape

            # set the mask directory containing all the masks
            masks_directory = os.path.join( self.split_path, dirname, image_id, 'masks' )

            # load the masks into a combined numpy array
            masks_array = get_masks_from_directory( masks_directory )
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

                    # store the window info to table and dataframe
                    out_table.append( [ window_name, image_id, window_height, window_width, image_height, image_width, window_center[0], window_center[1], is_nuclei ] )
                    out_dataframe.append( [ window_name, image_id, window_height, window_width, image_height, image_width, window_center[0], window_center[1], is_nuclei, window ] )

        # table information
        table_header = 'filename, parent_image_id, extract_height, extract_width, image_height, image_width, index_row, index_col, is_positive'
        table_path = os.path.join( self.windows_path, dirname + '_table.csv' )

        # dataframe information
        dataframe_header = [ 'filename', 'parent_image_id', 'extract_height', 'extract_width', 'image_height', 'image_width', 'index_row', 'index_col', 'is_positive', 'image_matrix' ]

        # save table
        print( 'Saving table as CSV' )
        numpy.savetxt( table_path, out_table, fmt='%s', delimiter=',', header=table_header )

        # save dataframe
        print( 'Saving dataframe as PKL' )
        dataframe_path = os.path.join( self.windows_path, dirname + '_dataframe.pkl' )
        save_dataframe = pandas.DataFrame( out_dataframe, columns=dataframe_header )
        save_dataframe.to_pickle( dataframe_path )

        return


class train_classifier( object ):

    def __init__( self, df_path, image_col_name, category_col_name, fixed_size, classifier_pkl,
                  feature_method, image_processing_options, raw_image_path,
                  solver='lbfgs', alpha=0.0001, tol=0.00001, max_iter=200, hidden_layer_sizes=(100,), random_state=1 ):

        """ The class constructor
        :param df_path: full path to the input roi DataFrame file (e.g. ../data/roi/roidf.h5)
        :param image_col_name: column name for the image data column
        :param category_col_name: column name for the image category (negative or positive) column
        :param fixed_size: 2x1 matrix containing the height and width of the images, format=[ height, width ]
        :param classifier_pkl: Path to the output classifier pickle file
        :param feature_method: current available:
                   'pixelval': extracting pixel values as features, no options needed (use None as input)
        :param image_processing_options: dict, options for preprocessing the image before roi extraction
        :param raw_image_path: Path to raw images
        :param solver: Neural network solver (e.g. lbfgs, sgd, adam)
        :param alpha: L2 penalty, aka regularization (e.g. 0.0001, 0.00001, etc.)
        :param hidden_layer_sizes: number of neurons in the ith hidden layer, e.g. (100,) or () or ()
        :param random_state: Basically the random seed (e.g. 1)
        :return: None
        """

        # parameters that require input
        self.fixed_size = fixed_size
        self.classifier_pkl = classifier_pkl

        # parameters for feature extraction
        self.feature_method = feature_method
        self.image_processing_options = image_processing_options
        self.raw_image_path = raw_image_path

        # neural network parameters
        self.solver = solver
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

        # data related parameters, needs to be read from a file
        self.data = None
        self.image_col_name = image_col_name
        self.category_col_name = category_col_name

        print( "Loading dataframe: " + df_path )
        self.data = self._load_data_from_dataframe( df_path )

        return


    def _load_data_from_dataframe(self, df_path):

        """ Loads data from a saved panda DataFrame file

        :param df_path: file path to the saved DataFrame file
        :return: none

        """

        try:
            df = pandas.read_pickle( df_path )
            # filter out the ROIs with a size not equal to the input fixed_size
            df = df[ df[ self.image_col_name ].apply( numpy.shape ) == self.fixed_size ]
        except IOError:
            print( 'Please check the training data file path and make sure the file is in pickle format.' )
            print( 'tried: ' + df_path )

        return df


    def run( self ):

        """ Run the main code to train classifier and save to file

        :return: None

        """

        # process images
        if self.image_processing_options:
            print( "Preprocessing images" )
            image_df = load_all_raw_images( self.raw_image_path )
            processed_data = process_roi_extractions( image_df, self.data,
                                                      self.image_processing_options )
        else:
            processed_data = self.data

        # extract features from dataframes
        print( "Extracting features" )
        features = feature_extraction( processed_data,
                                       input_parameters=self.image_col_name,
                                       method=self.feature_method)

        # create target array which contains the correct answer
        target = pandas.DataFrame( numpy.repeat( [ True ], len( features ) ) )
        target[ self.data[ self.category_col_name ].values == '0' ] = False

        # initialize classifier
        print( "Initializing classifier" )
        classifier = sklearn.neural_network.MLPClassifier( solver=self.solver,
                                                           alpha=self.alpha,
                                                           tol=self.tol,
                                                           max_iter=self.max_iter,
                                                           hidden_layer_sizes=self.hidden_layer_sizes,
                                                           random_state=self.random_state,
                                                           verbose=True )
        print( classifier )
        print()

        # train the classifier
        print( "Training classifier" )
        classifier.fit( features, target.values.ravel() )
        print()

        # save the classifier to be loaded later
        print( "Saving classifier: " + str( self.classifier_pkl ) )
        sklearn.externals.joblib.dump( classifier, self.classifier_pkl )

        return


class train_svm_classifier( object ):

    def __init__( self, df_path, image_col_name, category_col_name, fixed_size, classifier_pkl,
                  feature_method, feature_options, image_processing_options, raw_image_path,
                  kernel='rbf', gamma='auto', C=1 ):

        """ The class constructor

        :param df_path: full path to the input roi DataFrame file (e.g. ../data/roi/roidf.h5)
        :param image_col_name: column name for the image data column
        :param category_col_name: column name for the image category (negative or positive) column
        :param fixed_size: 2x1 matrix containing the height and width of the images, format=[ height, width ]
        :param classifier_pkl: Path to the output classifier pickle file
        :param kernel: SVM kernel (e.g. rbf, linear, poly, sigmoid)
        :param gamma: kernel coefficient (e.g. 'auto', 0.001, etc.)
        :param C: penalty parameter (e.g. 1.0, 100)
        :return: None

        """

        # parameters that require input
        self.fixed_size = fixed_size
        self.classifier_pkl = classifier_pkl

        # parameters for feature extraction
        self.feature_method = feature_method
        self.feature_options = feature_options
        self.image_processing_options = image_processing_options
        self.raw_image_path = raw_image_path

        # parameters that has default values, also SVM parameters
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

        # data related parameters, needs to be read from a file
        self.data = None
        self.image_col_name = image_col_name
        self.category_col_name = category_col_name
        
        print( "Loading dataframe: " + df_path )
        self._load_data_from_dataframe_file( df_path )

        return


    def _load_data_from_dataframe_file(self, df_path):

        """ Loads data from a saved panda DataFrame file
        :param df_path: file path to the saved DataFrame file
        :return: none

        """

        try:
            df = pandas.read_pickle(df_path)
            # filter out the ROIs with a size not equal to the input fixed_size
            self.data = df[df[self.image_col_name].apply(numpy.shape) == self.fixed_size]
        except IOError:
            print('Please check the training data file path and make sure the file is in pickle format.')
            print('tried: '+df_path)
        return


    def run( self ):
 
        """ Run the main code to train classifier and save to file

        :return: None

        """

        # process images
        if self.image_processing_options:
            print('Preprocessing images')
            image_df = load_all_raw_images(self.raw_image_path)
            processed_data = process_roi_extractions(image_df, self.data,
                    self.image_processing_options)
        else:
            processed_data = self.data
        # extract features from dataframes
        print('Extracting features')
        features = feature_extraction(processed_data,
                input_parameters=self.image_col_name,
                method=self.feature_method,
                extraction_options=self.feature_options)

        # create target array which contains the correct answer
        target = pandas.DataFrame( numpy.repeat( [ True ], len( features ) ) )
        target[ self.data[ self.category_col_name ].values == '0' ] = False

        # initialize classifier
        print( "Initializing classifier" )
        classifier = sklearn.svm.SVC( kernel=self.kernel, gamma=self.gamma, C=self.C, verbose=True )
        print( classifier )
        print()

        # train the classifier
        print( "Training classifier" )
        classifier.fit( features, target.values.ravel() )
        print()

        # save the classifier to be loaded later
        print( "Saving classifier: " + str( self.classifier_pkl ) )
        sklearn.externals.joblib.dump( classifier, self.classifier_pkl )


def png_to_pixel_matrix( image_path, mode='RGB' ):

    """ PNG to HxW matrix of image values for every pixel

    :param image_path: File path to the image file
    :return: HxW matrix of image values for every pixel

    """

    pixel_matrix = scipy.misc.imread( image_path, flatten=False, mode=mode )
    return pixel_matrix


def get_masks_from_directory( masks_directory ):

    """ Read in the directory of masks and get mask matrix

    :param masks_directory: The directory containing masks in black/white format
    :return: List of masks in matrix format: [n, H, W]

    """

    if not os.path.isdir( masks_directory ):
        raise Exception( 'Invalid directory containing masks' )

    raw_masks = skimage.io.imread_collection( os.path.join( masks_directory, '*.png' ), conserve_memory=True ).concatenate()
    number_of_masks = raw_masks.shape[ 0 ]
    height, width = raw_masks[ 0 ].shape

    # convert arrays that go from 0 or 255 into 0 or 1
    output_masks = numpy.zeros( ( number_of_masks, height, width ), numpy.uint16 )
    output_masks[:,:,:] = raw_masks[:,:,:] / 255

    return output_masks


def load_all_raw_images( imgspath, mode='RGB' ):

    """ Function that reads all the images in the training set provided
    by Kaggle and put them into a dataframe with two columns: ImageId
    and ImageMat

    :param imgspath: Path to the image folder
    :return: A dataframe that contains ImageId and ImageMat as columns.
        ImageMat is the array representation of the image in grayscale

    """

    # list all the image IDs, which is also the subfolder names
    imdirs = [d for d in os.listdir(imgspath)
              if os.path.isdir(os.path.join(imgspath, d))]
    # read the images with scipy.misc.imread function
    images = [png_to_pixel_matrix(os.path.join(imgspath, im, 'images', im + '.png'), mode=mode)
            for im in imdirs]
    return pandas.DataFrame(list(zip(imdirs, images)), columns=['ImageId', 'ImageMat'])


def process_roi_extractions(image_df, roi_df, image_options, image_col_name='ImageMat', image_id_col_name='ImageId'):
    """
    Change the roi images with the image process options applied to its parent image
    :param image_df: dataframe containing the parent images
    :param image_col_name: string, name for the parent image data column in image_df
    :param image_id_col_name: string, name for the parent image id column in image_df
    :param roi_df: dataframe containing roi images, must be those generated by preprocessing_extract_roi_from_split.py
                   or have the same columns
    :param image_options: dict, options for preprocessing the image before roi extraction
    :return:
    """
    output = roi_df.copy()
    # calculate the extraction range in the parent image
    output = output.assign(row1=numpy.maximum(0, roi_df.index_row-numpy.floor(roi_df.extract_height/2))+1)
    output = output.assign(row2=numpy.maximum(0, roi_df.extract_height+output.row1))
    output = output.assign(col1=numpy.maximum(0, roi_df.index_col-numpy.floor(roi_df.extract_width/2))+1)
    output = output.assign(col2=numpy.maximum(0, roi_df.extract_width+output.col1))
    # loop through all parent images
    for i in range(len(image_df)):
        # find the ID for each image
        image_id = image_df[image_id_col_name][i]
        # process the image according to options
        processed_image = process_image(image_df[image_col_name][i], image_options)
        # collect all the rois associated with this parent image
        rois = output[output.parent_image_id == image_id]
        # loop through each roi extraction
        for index, roi in rois.iterrows():
            # set the roi data with the processed image (replacing the raw extraction)
            output.set_value(index, 'image_matrix',
                             processed_image[int(roi.row1):int(roi.row2),int(roi.col1):int(roi.col2)])

    return output


def process_image(image_array, options):
    """
    preprocesses image with all options available
    :param image_array: image data
    :param options: option dictionary, available option keys and values are:
                    'rgb2gray': None -- converting rgb image to gray scale, no option values
                    'trim': [int1, int2] -- trims the pixel values lower than int1 percentile and higher than int2
                            percentile
                    'norm': 'equal_hist' or 'clahe', or default (all other option values -- normalizes the image with:
                            'equal_hist': global histogram equalization, 'clahe': contrast limited adaptive histogram
                            equalization, default: stretching the min and max of input image pixel values to the min
                            and max of the respective data type (e.g., min to 0 and max to 255 for uint8)
    :return: processed image
    """
    if 'rgb2gray' in options.keys():
        image_array = skimage.color.rgb2gray(image_array)
    if 'trim' in options.keys():
        if len(options['trim']) < 2:
            print('Must provide a list with two number for trimming. Received: ', options['trim'])
        else:
            image_array = trim_extreme_pixels_in_grayscale(image_array,
                                                           trim_left=options['trim'][0],
                                                           trim_right=options['trim'][1])
    if 'norm' in options.keys():
        if options['norm'] == 'equal_hist':
            image_array = skimage.exposure.equalize_hist(image_array)
        elif options['norm'] == 'clahe':
            image_array = skimage.exposure.equalize_adapthist(image_array)
        else:
            image_array = skimage.exposure.rescale_intensity(image_array)

    return image_array


def trim_extreme_pixels_in_grayscale(image_array, trim_left=1, trim_right=99):
    """
    change the pixel values less than the trim_left percentile and larger than the trim_right percentile to the
    value of trim_left percentile and trim_right percentile respectively
    :param image_array: numpy array, input image
    :param trim_left: lowest percentile value to keep, any pixel value < the value for this percentile is converted up.
                      Use None if wish to not trim the lower end
    :param trim_right: highest percentile value to keep, any pixel value > the value for this percentile is converted
                       down. Use None if wish to not trim the upper end
    :return: trimmed image array
    """
    if trim_left and trim_right:
        min_val, max_val = numpy.percentile(image_array.ravel(), [trim_left, trim_right])
        image_array = numpy.maximum(min_val, numpy.minimum(max_val, image_array))
    elif trim_left and not trim_right:
        min_val = numpy.percentile(image_array.ravel(), trim_left)
        image_array = numpy.maximum(min_val, image_array)
    elif trim_right and not trim_left:
        max_val = numpy.percentile(image_array.ravel(), trim_right)
        image_array = numpy.minimum(max_val, image_array)
    else:
        pass

    return image_array


def feature_extraction(input_data, input_parameters, method='pixelval', extraction_options={}):
    """
    extracting features from input data, with specified method, and options
    :param input_data: input data for extraction
    :param input_parameters: parameters to accompany input data, e.g., image data column name if input_data is an image
                             dataframe
    :param method: current available:
                   'pixelval': extracting pixel values as features, no options needed (use None as input)
    :param image_process_options: dict, options to preprocess the image before feature extraction, e.g.:
                                  {'rgb2gray': None}: converting rgb images into grayscale images
                                  {'trim': [1, 99]}: trimming off the top and bottom 1% pixel values
    :return: list, features for all inputs (n_input, n_features_per_input)
    """

    if method == 'gkhp':
        features = extract_gkhp_features(input_data, input_parameters, extraction_options)
    elif method == 'pixelval':
        features = extract_pixel_val_as_features(input_data, input_parameters)
    else:
        raise ValueError('Invalid feature extraction method: '+method+'!\n '
                        'Please use pixeval instead.')

    return features


def extract_pixel_val_as_features(image_dataframe, image_col_name):
    """
    extracting pixel values, post processing as specified in image_process_options, as features
    :param image_dataframe: dataframe containing all image data
    :param image_col_name: string, name for the column that contains image data
    :return: a list of pixel values, each entry correspond to each image in the dataframe
    """
    # print("extracting pixel value as features", flush=True)
    features = [0]*len(image_dataframe)

    for index, img_df in image_dataframe.iterrows():
        features[index] = img_df[image_col_name].ravel()

    return features


def gaussian_kernel_hadamard_product_feature(image_array, extraction_options):
    """
    Each feature is the total sum of the Hadamard product between a Gaussian kernel with a sigma specified by
    extraction_options and the image_array. Each sigma in extraction_options returns one feature.
    :param image_array: input image data
    :param extraction_options: list, containing all sigmas to use for Gaussian kernel
    :param image_process_options: dict, options for preprocessing image before extraction
    :return: numpy array, features of the input image, size len(extraction_options)
    """
    if not extraction_options:
        raise ValueError('Must provide options for Gaussian Kernel sigmas for Gaussian Kernel Hadamard product '
                         'feature extraction!')
    # preprocess image
    features = []
    for kernel in extraction_options:
        # calculating the feature value, the total sum of the Hadamard product between a kernel and the image
        features.append((kernel * image_array).sum())

    return numpy.array(features)


def extract_gkhp_features(image_dataframe, image_col_name, extraction_options):
    """
    extracting Gaussian kernel Hadamard product features from a dataframe of images
    :param image_dataframe: input image dataframe
    :param image_col_name: string, name for the image data column
    :param extraction_options: dict, options for Gaussian kernel generation, default as follows:
                                {'sigmas': [1, 2, 4, 8, 16, 32, 64, 128]}
    :param image_process_options: dict, options for preprocessing image before extraction
    :return: a list of all features for all images (n_images, n_features)
    """
    # print("extracting Gaussian kernel Hadamart product features", flush=True)
    features = []

    # default options if one or more options are missing
    default_options = {'sigmas': [1, 2, 4, 8, 16, 32, 64, 128]}
    # update default options with input options
    if extraction_options:
        options = {**default_options, **extraction_options}
    else:
        options = default_options

    # generate Gaussian kernels for all images
    kernels = []
    # get the size of processed image for Gaussian kernel generation
    image_array = image_dataframe.iloc[0][image_col_name]
    # image_array = image_processing.process_image(image_array, {'rgb2gray': None})
    # generate a central peak to be used for generating Gaussian kernel
    central_peak = numpy.zeros(image_array.shape)
    central_peak[int(image_array.shape[0] / 2)][int(image_array.shape[1] / 2)] = 1.0
    # generate all Gaussian for each kernel parameters
    for sigma in options['sigmas']:
        # append the kernel to the list of all kernels
        kernels.append(scipy.ndimage.filters.gaussian_filter(central_peak, sigma))

    # for each image, calculate the GKHP feature for all kernels generated above
    for index, img_df in image_dataframe.iterrows():
        # input_image = image_processing.process_image(img_df[image_col_name], {'rgb2gray': None})
        input_image = img_df[image_col_name]
        features.append(gaussian_kernel_hadamard_product_feature(input_image,
                                                                 kernels))
    return features




def mask_matrix_to_kaggle_format( mask_matrix, image_id):

    """ Mask matrix to formatted Kaggle submission string

    :param mask_matrix: HxW matrix (see other functions)
    :param image_id: id of image
    :return: String in the format of Kaggle's submission output

    """

    nuclei_list = mask_matrix_to_nuclei_list( mask_matrix )
    return nuclei_list_to_kaggle_format( nuclei_list, image_id )


def mask_matrix_to_nuclei_list( mask_matrix ):

    """ Mask matrix to nuclei 1D list containing indices of every nuclei (non-black) pixel

    :param mask_matrix: HxW matrix (see other functions)
    :return: 1D list containing indices of every nuclei (non-black) pixel

    """

    height, width = mask_matrix.shape
    mask_matrix_T = numpy.transpose( mask_matrix, axes=( 1, 0 ) )
    mask_vector = mask_matrix_T.reshape( ( height * width ) )

    nuclei_list = []
    for pixel_idx in range( len( mask_vector ) ):

        # any non-black pixel is saved
        if mask_vector[ pixel_idx ] == 1:

            # index is corrected since Kaggle indices starts at 1 (not 0)
            nuclei_list.append( pixel_idx + 1 )

    return nuclei_list


def nuclei_list_to_kaggle_format( nuclei_list, image_id ):

    """ Nuclei list to formatted Kaggle submit string

    :param nuclei_list: List of nuclei indices (see other functions)
    :param image_id: id of image
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


def run_moving_window(classifier, image_array, image_options,
                      window_sizes, step_sizes, padding="constant",
                      feature_method='pixelval', feature_options={}):

    """
    Run the moving window approach

    :param classifier: classifier to use for prediction
    :param image_array: input image data
    :param image_options: options for preprocessing images before feature extractions, e.g., {'rgb2gray': None} to
                          convert RGB image into grayscale. If no option required, use {} (enpty dict)
    :param window_sizes: sliding window size (nrow, ncol)
    :param step_sizes: step size for the sliding window (nrow, ncol)
    :param paddding: string specifying padding option
    :return: None
    """

    # preprocess the image
    print("Preprocess image")
    processed_image = process_image(image_array, image_options)

    # extract features from dataframes
    print("Extracting sub images", end=" ")
    start_time = time.time()
    boxes = extract_windowed_subimages_from_image(processed_image, window_sizes, step_sizes, padding=padding)
    print(time.time()-start_time)

    # extraction features according the feature_method parameter and feature_options, images could
    # be processed according to image_options if needed
    print("Extracting features", end=" ")
    start_time = time.time()
    features = feature_extraction(boxes, 'ImageMat', feature_method, feature_options)
    print(time.time()-start_time)

    # run SVM prediction
    print("Predicting", end=" ")
    boxes["classification"] = classifier.predict(features)
    boxes = boxes[ boxes.classification == True ]
    print(time.time()-start_time)

    return boxes


def extract_windowed_subimages_from_image(image_array, window_sizes, step_sizes, pad_image=True, padding="constant"):

    """
    function to extract sub images with a moving window
    :param image_array: numpy array like object containing image data, the first two dimensions needs to be
                        image sizes
    :param window_sizes: list object, size of the extraction window, in (height, width) format
    :param step_sizes: list object, specifying how far to move the window. It is in (horizontal step, vertical step).
    :param pad_image: boolean whether to pad image
    :param paddding: string specifying padding option
    :return: dataframe, each row contains ImageMat: numpy array of subimage data, and SubImageAnchor: list [starting
             row number, starting column number]
    """

    # image size is in (height, width, color channels) format. We only use height and width here
    image_sizes = image_array.shape
    ndim = len(image_sizes)

    # padding image
    anchor_correction = (0, 0)
    if pad_image:
        image_array = pad_images(image_array, window_sizes, padding=padding)
        image_sizes = image_array.shape
        anchor_correction = (int(window_sizes[0]/2), int(window_sizes[1]/2))
    # populate the window starting points
    window_col_start = list(range(0, image_sizes[1] + 1 - window_sizes[1], step_sizes[0]))
    window_row_start = list(range(0, image_sizes[0] + 1 - window_sizes[0], step_sizes[1]))
    r = image_sizes[1] % step_sizes[0]
    if r > 0:
        window_col_start.append(image_sizes[1] - window_sizes[1])
        window_row_start.append(image_sizes[0] - window_sizes[0])

    sub_images = []
    sub_images_anchors = []
    for c in window_col_start:
        for r in window_row_start:
            # each window starts on the left upper conner at (r, c) and has a height of window_sizes[0]
            # and width of window_sizes[1]
            if ndim > 2:
                sub_images.append(image_array[r:r + window_sizes[0], c:c + window_sizes[1], :])
            else:
                sub_images.append(image_array[r:r + window_sizes[0], c:c + window_sizes[1]])
            sub_images_anchors.append([r-anchor_correction[0], c-anchor_correction[1]])
    df = pandas.DataFrame(list(zip(sub_images, sub_images_anchors)), columns=['ImageMat', 'SubImageAnchor'])
    df["window_size"] = [[window_sizes[0], window_sizes[1]]]*len(df)
    del image_array
    del sub_images
    del sub_images_anchors
    return df


def pad_images(image_array, window_sizes, padding="constant"):

    """
    pad an image in each direction by half the window size in the corresponding direction
    :param image_array: numpy array, input image
    :param window_sizes: list of two int, correspond to size of the windows in vertical and horizontal directions
    :param paddding: string specifying padding option
    :param step_sizes:
    :return:
    """

    image_sizes = image_array.shape
    ndim = len(image_sizes)
    if ndim > 2:
        # pad images with more than one channel
        padded_image = numpy.zeros((image_sizes[0]+window_sizes[0],
                                image_sizes[1]+window_sizes[1],
                                image_sizes[2]))
        for channel in range(image_sizes[2]):
            if padding=="constant":
                padded_image[:,:,channel] = skimage.util.pad(image_array[:,:,channel],
                                                             (int(window_sizes[0]/2), int(window_sizes[1]/2)),
                                                             'constant',
                                                             constant_values=numpy.median(image_array[:,:,channel]))
            else:
                try:
                    padded_image[:,:,channel] = skimage.util.pad(image_array[:,:,channel],
                                                                 (int(window_sizes[0]/2), int(window_sizes[1]/2)),
                                                                 padding)
                except:
                    raise("Pad option not recognized.")
    else:
        # pad single channel images
        if padding=="constant":
            padded_image = skimage.util.pad(image_array,
                                            (int(window_sizes[0]/2), int(window_sizes[1]/2)),
                                            'constant',
                                            constant_values=numpy.median(image_array))
        else:
            try:
                padded_image = skimage.util.pad(image_array,
                                                (int(window_sizes[0]/2), int(window_sizes[1]/2)),
                                                padding)
            except:
                raise("Pad option not recognized.")

    return padded_image


def boxes_to_mask(boxes, image_array, replacement=[1, 1]):

    """
    Convert positively identified subimages to a mask

    :param boxes: Dataframe describing positively identified subimages
    :param image_array: image array of original image
    :param replacement: array with describing the size of the replacement for the positively identified pixel
    :return: array with mask
    """

    mask = numpy.zeros((image_array.shape[0], image_array.shape[1]))

    if replacement[0]%2 == 0:
        raise("Amount of rows should be an uneven number.")

    if replacement[1]%2 == 0:
        raise("Amount of columns should be an uneven number.")

    for i in boxes.index:
        row_start = max(0, boxes.SubImageAnchor[i][0]+int(boxes.window_size[i][0]/2)-1)
        row_end = min(image_array.shape[0]-1, row_start + replacement[0])
        column_start = max(0, boxes.SubImageAnchor[i][1]+int(boxes.window_size[i][1]/2)-1)
        column_end = min(image_array.shape[1]-1, column_start + replacement[1])

        if row_end>row_start and column_end>column_start:
            mask[row_start:row_end, column_start:column_end] = 1
        elif row_end==row_start:
            mask[row_start, column_start:column_end] = 1
            if column_end==column_start:
                mask[row_start, column_start] = 1
        elif column_end==column_start:
            mask[row_start:row_end, column_start] = 1
        else:
            raise Exception('Trying to edit mask outside of image array.')

    return mask


def apply_segmentation(image, type, centers=None, lim_segment_size=[1, math.inf]):

    """
    Segments an image using segmentation

    :param image: boolean array is_part_of_nucleus
    :param type: string specifying type of segmentation
    :param centers: array with booleans where entries with seed indices are True
    :param lim_segment_size: array specifying limits of segment size [min, max]
    :return: list with arrays describing masks
    """

    distance_transform = scipy.ndimage.distance_transform_edt(image)

    lbl = skimage.measure.label(centers)
    if type == "watershed":
        ids = skimage.morphology.watershed(-distance_transform, lbl, mask=image)
    else:
        raise("Type of segmentation not recognized.")


    segmentation = [1*(ids == value) for value in range(1, numpy.amax(ids)+1, 1)]
    filtered_segmentation = [x for x in segmentation if lim_segment_size[0]<numpy.sum(x)<lim_segment_size[1]]

    return filtered_segmentation


def tune(main, model, tuned_parameters, feature_method="pixelval", fraction=1, diversity_vars=None, iterations=50, bayes=False, verbose=0, n_jobs=1, random_state=1):
    """

    :param main: Classifier object
    :param model: Classifier
    :param tuned_parameters: Grid with hyperparameters
    :param feature_method: string describing method to extract features
    :param fraction: Fraction of data used to tune hyperparameters
    :param diversity_vars: Variable names of which diversity should be maintained
    :param iterations: Amount of iteration for Bayesian search
    :param bayes: boolean indicating to use Bayesian search (True) or normal search (False)
    :param verbose: How much to output (e.g. 0, 10, 50)
    :param n_jobs: Number of jobs to run in parallel (e.g. 1, 2, 4, 8)
    :param random_state: Basically the random seed (e.g. 1)
    :return: Classifier object and search object
    """

    data = slice_smaller_subset_of_data(main.data, fraction=fraction, diversity_vars=diversity_vars,
            random_state=random_state)

    # preprocessing images
    if main.image_processing_options:
        print('Preprocessing images')
        image_df = load_all_raw_images(main.raw_image_path)
        processed_data = process_roi_extractions(image_df, data, main.image_processing_options)
    else:
        processed_data = data
    # extract features from dataframes
    print('Extracting features using: '+feature_method)
    features = pandas.DataFrame( feature_extraction( processed_data,
            input_parameters=main.image_col_name,
            method=feature_method,
            extraction_options=main.feature_options ) )

    # create target array which contains the correct answer
    target = pandas.DataFrame(numpy.repeat([True], len(features)))
    target[data[main.category_col_name].values == '0'] = False

    # take the sliced data to further split into a training and x-validation (for hyper-parameter tuning) sets
    print('Preparing training and cross-validation sets')
    train_indices = split_data_stratified(data = data, fraction = 0.8, diversity_vars = diversity_vars, random_state = random_state)
    train_indices = features.index.isin(features.index[train_indices])
    train_features = features[train_indices]
    train_target = target[train_indices].values.ravel()
    test_features = features[train_indices == False]
    test_target = target[train_indices == False].values.ravel()

    inner_cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=1)

    print('Begin training and tuning')
    #if bayes:
    #    opt = skopt.BayesSearchCV(model, tuned_parameters, n_iter=iterations, cv=inner_cv, verbose=True)
    #else:
    opt = sklearn.model_selection.GridSearchCV(model, tuned_parameters, cv=inner_cv, scoring= "neg_log_loss", verbose=verbose, n_jobs=n_jobs)
    opt.fit(train_features, train_target)
    print(opt.best_params_)
    print("Train score: %s" % opt.best_score_)
    print("Test score: %s" % opt.score(test_features, test_target))

    for name, value in opt.best_params_.items():
        setattr(main, name, value)
    return main, opt


def split_data_stratified(data, fraction, diversity_vars, random_state):
    if diversity_vars == None:
        diversity_df = numpy.zeros(len(data.index))
    else:
        diversity_df = data[diversity_vars]
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, train_size=fraction,
                                                         test_size=1 - fraction, random_state=random_state)
    split_indices = sss.split(numpy.zeros(len(data.index)), diversity_df)
    return next(split_indices)[0]


def slice_smaller_subset_of_data(data, fraction, diversity_vars, random_state):

    """ Create a subset of the data while maintaining diversity
    :param fraction: Fraction of data to keep
    :param diversity_vars: Variable names of which diversity should be maintained
    :param random_state: Basically the random seed (e.g. 1)
    :return: None
    """

    selected_indices = split_data_stratified(data = data, fraction=fraction, diversity_vars=diversity_vars, random_state=random_state)
    print("Slicing data")
    data = data[data.index.isin(data.index[selected_indices])]

    return data


