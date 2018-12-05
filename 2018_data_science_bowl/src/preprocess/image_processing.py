#!/usr/bin/env python3

"""
Contains functions to preprocess the raw input images, before extracting ROI or running through other analysis
"""

import numpy as np
import skimage.color
import skimage.transform
import skimage.exposure
import skimage.filters


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
        min_val, max_val = np.percentile(image_array.ravel(), [trim_left, trim_right])
        image_array = np.maximum(min_val, np.minimum(max_val, image_array))
    elif trim_left and not trim_right:
        min_val = np.percentile(image_array.ravel(), trim_left)
        image_array = np.maximum(min_val, image_array)
    elif trim_right and not trim_left:
        max_val = np.percentile(image_array.ravel(), trim_right)
        image_array = np.minimum(max_val, image_array)
    else:
        pass

    return image_array


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
        # image_array = (image_array*255).astype(np.uint8)
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
        # image_array = skimage.exposure.rescale_intensity(image_array)
    if 'sobel' in options.keys():
        image_array = skimage.filters.sobel(image_array)

    return image_array


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
    output = output.assign(row1=np.maximum(0, roi_df.index_row-np.floor(roi_df.extract_height/2))+1)
    output = output.assign(row2=np.maximum(0, roi_df.extract_height+output.row1))
    output = output.assign(col1=np.maximum(0, roi_df.index_col-np.floor(roi_df.extract_width/2))+1)
    output = output.assign(col2=np.maximum(0, roi_df.extract_width+output.col1))
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
