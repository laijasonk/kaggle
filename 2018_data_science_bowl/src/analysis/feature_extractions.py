#!/usr/bin/env python3

"""
This module contains all the functions that will be used for extracting different types of features
Call extracting_feature with the feature name to activate the individual feature extraction functions
"""

import scipy.ndimage.filters
import numpy
import skimage.feature
import skimage.color
import sklearn
from preprocess import image_processing


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


def extract_hog_features(image_dataframe, image_col_name, extraction_options):
    """
     extracts hog features from images stored in a dataframe
     :param image_dataframe: input image dataframe, must have a column containing the image data
     :param image_col_name: string, containing the column name for the image data
     :param extraction_options: dict, options for HOG features, default as follows:
                       {'orientation': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}
     :param image_process_options: options for processing the image before feature extraction, e.g., converting image
                                   from RGB into gray scale with {'rgb2gray': None}
     :return: a list of HOG features, each entry correspond to each image in the input dataframe
     """
    # print("extracting HOG features", flush=True)
    features = []

    # default options if one or more options are missing
    default_options = {'orientation': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}
    # update default options with input options
    if extraction_options:
        options = {**default_options, **extraction_options}
    else:
        options = default_options
    # extract hog features for each image, with the given options
    for index, img_df in image_dataframe.iterrows():
        features.append(skimage.feature.hog(img_df[image_col_name], orientations=options['orientation'],
                                            pixels_per_cell=options['pixels_per_cell'],
                                            cells_per_block=options['cells_per_block'],
                                            block_norm=options['block_norm']))

    return features


def extract_pixel_val_as_features(image_dataframe, image_col_name):
    """
    extracting pixel values, post processing as specified in image_process_options, as features
    :param image_dataframe: dataframe containing all image data
    :param image_col_name: string, name for the column that contains image data
    :param image_process_options: dict, options for processing image before extraction
    :return: a list of pixel values, each entry correspond to each image in the dataframe
    """
    # print("extracting pixel value as features", flush=True)
    features = [0]*len(image_dataframe)

    for index, img_df in image_dataframe.iterrows():
        features[index] = img_df[image_col_name].ravel()

    return features


def feature_extraction(input_data, input_parameters, method='pixelval', extraction_options={}):
    """
    extracting features from input data, with specified method, and options
    :param input_data: input data for extraction
    :param input_parameters: parameters to accompany input data, e.g., image data column name if input_data is an image
                             dataframe
    :param method: current available:
                   'gkhp': Gaussian kernel Hadamart product feature, requires a list of Gaussian kernel sigmas as
                           extraction options
                   'hog': Histogram of gradients, requires options in a dict if none-default options is desired
                   'pixelval': extracting pixel values as features, no options needed (use None as input)
    :param extraction_options: dict, depending on the method, default ones shown below:
                               'gkhp': {'sigmas': [1, 2, 4, 8, 16, 32, 64, 128]}
                               'hog': {'orientation': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2),
                                       'block_norm': 'L2-Hys'}
                               'pixelval': {}, i.e., no options needed
                               if use {}, empty list, default options will be used
    :param image_process_options: dict, options to preprocess the image before feature extraction, e.g.:
                                  {'rgb2gray': None}: converting rgb images into grayscale images
                                  {'trim': [1, 99]}: trimming off the top and bottom 1% pixel values
    :return: list, features for all inputs (n_input, n_features_per_input)
    """

    if method == 'gkhp':
        features = extract_gkhp_features(input_data, input_parameters, extraction_options)
    elif method == 'hog':
        features = extract_hog_features(input_data, input_parameters, extraction_options)
    elif method == 'pixelval':
        features = extract_pixel_val_as_features(input_data, input_parameters)
    else:
        raise ValueError('Invalid feature extraction method: '+method+'!\n '
                        'Please use \'gkhp\', \'hog\', or \'pixeval\' instead.')

    return features


class feature_extraction_class(sklearn.base.TransformerMixin):

    def __init__(self, input_parameters=None, method=None, extraction_options=None):
        self.input_parameters=input_parameters
        self.method=method
        self.extraction_options=extraction_options

    def transform(self, X, **kwargs):
        features = feature_extraction(X, self.input_parameters, method=self.method,
                                      extraction_options=self.extraction_options)
        return features

    def fit(self, X, y=None, **kwargs):
        return self

    def get_params(self, **kwargs):
        return {"input_parameters": self.input_parameters,"method": self.method, "extraction_options":self.extraction_options}

    def set_params(self, input_parameters=None, method=None, extraction_options=None):
        self.input_parameters=input_parameters
        self.method=method
        self.extraction_options=extraction_options
