#!/usr/bin/env python3

"""
Containing functions for moving window classifier
"""

import pandas as pd
import numpy
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from analysis import feature_extractions, NSM
from preprocess import image_processing
import time
import skimage.util


def pad_images(image_array, window_sizes, padding="constant"):

    """
    pad an image in each direction by half the window size in the corresponding direction
    :param image_array: numpy array, input image
    :param window_sizes: list of two int, correspond to size of the windows in vertical and horizontal directions
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


def extract_windowed_subimages_from_image(image_array, window_sizes, step_sizes, pad_image=True, padding="constant"):

    """
    function to extract sub images with a moving window
    :param image_array: numpy array like object containing image data, the first two dimensions needs to be
                        image sizes
    :param window_sizes: list object, size of the extraction window, in (height, width) format
    :param step_sizes: list object, specifying how far to move the window. It is in (horizontal step, vertical step).
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
    df = pd.DataFrame(list(zip(sub_images, sub_images_anchors)), columns=['ImageMat', 'SubImageAnchor'])
    df["window_size"] = [[window_sizes[0], window_sizes[1]]]*len(df)
    del image_array
    del sub_images
    del sub_images_anchors
    return df


def classify_boxes(features, model):

    """
    Return classification category probabilities

    :param features: features to use in the model
    :param model: the classifier
    :return:
    """

    # initialize variables
    predicted = numpy.zeros((len(features), 2))
    curr_fraction = 0
    fraction = 25000 / len( features ) # only predict 25000 at a time
    prev_idx = 0
    curr_idx = 0

    # predict 50000 boxes at a time to prevent memory overflow
    while curr_idx < len( features ):
        curr_fraction += fraction # keep track of the current slice
        curr_idx = min( int( len(features) * curr_fraction ), len(features) ) # don't go over size of the feature matrix
        curr_features = features[ prev_idx:curr_idx ] # slice up the boxes
        prev_idx = curr_idx

        # the actual prediction
        if type(model).__name__ == "SVC":
            curr_predicted = model.decision_function( curr_features )
        else:
            curr_predicted = model.predict_proba( curr_features )

        predicted[curr_idx-curr_predicted.shape[0]:curr_idx, :] = curr_predicted

    if type(model).__name__ == "SVC":
        return predicted[:]
    else:
        return predicted[:,1]
 
    # old version
    ## run prediction
    #if type(model).__name__ == "SVC":
    #    predicted = model.decision_function( features )
    #    return predicted[:]
    #else:
    #    predicted = model.predict_proba( features )
    #    return predicted[:, 1]


def display_boxes(image_array, reduced_boxes, boxes):

    """ Display the subimages in a figure

    :param image_array: numpy array of the image
    :param reduced_boxes: Dataframe with a subimage data
    :param boxes: Dataframe with data of a filtered set subimages
    :return:
    """

    fig = plt.figure(1, figsize=(15, 45))
    ax = fig.add_subplot(131)
    ax.imshow(image_array)
    ax = fig.add_subplot(132)
    ax.imshow(image_array)
    for i in boxes.ImageMat.index:
        left = boxes.SubImageAnchor[i][1]
        bottom = boxes.SubImageAnchor[i][0]
        circ = patches.Circle((left+boxes.ImageMat[i].shape[1]/2, bottom+boxes.ImageMat[i].shape[0]/2),
                              radius=2, facecolor='r', alpha=0.5)
        # rect = patches.Rectangle((left, bottom),boxes.ImageMat[i].shape[1],boxes.ImageMat[i].shape[0],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(circ)
    ax = fig.add_subplot(133)
    ax.imshow(image_array)
    for i in reduced_boxes.ImageMat.index:
        left = boxes.SubImageAnchor[i][1]
        bottom = boxes.SubImageAnchor[i][0]
        circ = patches.Circle((left+boxes.ImageMat[i].shape[1]/2, bottom+boxes.ImageMat[i].shape[0]/2),
                              radius=2, facecolor='g')
        rect = patches.Rectangle((left, bottom),boxes.ImageMat[i].shape[1],boxes.ImageMat[i].shape[0],linewidth=1,
                                 edgecolor='g',facecolor='none', alpha=0.5)
        ax.add_patch(circ)
        ax.add_patch(rect)
    plt.show(block=False)
    return


def run_moving_window(classifier, image_array, feature_method, feature_options, image_options,
                      window_sizes, step_sizes, nms_threshold, plot=False, padding="constant"):

    """
    Run the moving window approach

    :param classifier: classifier to use for prediction
    :param image_array: input image data
    :param feature_method: the feature to extract
    :param feature_options: options for feature extraction, e.g., for Gaussian kernel this is the kernel sigmas.
                            Current options include:
                            'gkhp': Gaussian kernel Hadamart product feature, requires a list of Gaussian kernel sigmas
                            as extraction options
                            'hog': Histogram of gradients, requires options in a dict if none-default options is desired
                            'pixelval': extracting pixel values as features, no options needed (use None as input)
    :param image_options: options for preprocessing images before feature extractions, e.g., {'rgb2gray': None} to
                          convert RGB image into grayscale. If no option required, use {} (enpty dict)
    :param window_sizes: sliding window size (nrow, ncol)
    :param step_sizes: step size for the sliding window (nrow, ncol)
    :param nms_threshold: Threshold parameter for the non-maximum suppression
    algorithm specifying the maximum allowable overlap
    :return: None
    """

    # preprocess the image
    print("Preprocess image")
    processed_image = image_processing.process_image(image_array, image_options)


    # extract features from dataframes
    print("Extracting sub images", end=" ")
    start_time = time.time()
    boxes = extract_windowed_subimages_from_image(processed_image, window_sizes, step_sizes, padding=padding)
    print(time.time()-start_time)

    if not(type(classifier).__name__ == "SVC") and feature_method == "pixelval":
        print("Using fast method")
        boxes = boxes[[classifier.predict_proba([x.ravel()])[0][1]>0.5 for x in boxes.ImageMat]]
        return boxes, boxes

    # extraction features according the feature_method parameter and feature_options, images could
    # be processed according to image_options if needed
    print("Extracting features", end=" ")
    start_time = time.time()
    features = feature_extractions.feature_extraction(boxes, 'ImageMat', feature_method, feature_options)
    print(time.time()-start_time)

    # run prediction
    print("Running prediction", end=" ")
    start_time = time.time()
    if type(classifier).__name__ == "SVC":
        boxes["classification"] = classifier.predict(features)
        positive_boxes = boxes[boxes.classification == True]
    else:
        boxes["prediction"] = classify_boxes(features, classifier)
        boxes = boxes[boxes.prediction > 0.5]
    print(time.time()-start_time)

    start_time = time.time()
    del features
    del boxes
    print("Removing variables time" + str(time.time()-start_time))

    print("eliminating overlap", end=" ")
    start_time = time.time()
    if nms_threshold < 1:
        reduced_positive_boxes = NSM.remove_boxes_with_NSM(positive_boxes, nms_threshold)
    else:
        reduced_positive_boxes = positive_boxes
    print(time.time()-start_time)
    if plot:
        display_boxes(image_array, reduced_positive_boxes, positive_boxes)

    return positive_boxes, reduced_positive_boxes


def boxes_to_mask(boxes, image_array, replacement=[1, 1], value=False):

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

        if value:
            val = boxes.prediction[i]
        else:
            val = 1

        if row_end>row_start and column_end>column_start:
            mask[row_start:row_end, column_start:column_end] = val
        elif row_end==row_start:
            mask[row_start, column_start:column_end] = val
            if column_end==column_start:
                mask[row_start, column_start] = val
        elif column_end==column_start:
            mask[row_start:row_end, column_start] = val
        else:
            raise Exception('Trying to edit mask outside of image array.')

    return mask
