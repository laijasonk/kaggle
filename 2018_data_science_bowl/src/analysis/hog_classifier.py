#!/usr/bin/env python3

"""
Histogram of Orientated Gradient method for feature extraction from images
containing functions to calculate gradient orientation and strength for each pixel; normalizing oritentation
and strength with a given sized cell; averaging for a block (sliding window)
"""

import numpy as np
from scipy import signal
from skimage.feature import hog
from skimage.color import rgb2gray


def extract_hog_features_from_image(img, orientation_bins, is_unsigned=True, cell_size=8,
                                    epsilon=1, block_size=2, step_size=1):
    """
    returns histogram of orientated gradient features for an input image
    :param img: 2D numpy array like, image data
    :param orientation_bins: bins for the orientation histogram
    :param is_unsigned: bool, whether the gradient is signed or not
    :param cell_size: cell size (square box) to investigate (total pixels to investigate = cell_size^2)
    :param epsilon: float, a constant for norm calculation to avoid division by 0
    :param block_size: the size of each block, overlap is half of the block size
    :param step_size: the step size of the blocks (i.e., block overlap area is block_size - step_size
    :return: numpy 3D array, block normalized cell gradient orientations
    """
    gradx, grady = derivative_mask_1d_centered(img)
    orientation, strength = find_grad_orientation_strength(gradx, grady, is_unsigned)
    cell_features = grad_histogram_by_cell(orientation, strength, orientation_bins, cell_size)
    return block_norm_of_cell_histogram(cell_features, epsilon, block_size, step_size)


def derivative_mask_1d_centered(img):
    """
    Compute the 1d centered derivative mask of an image see doc hog_rafael_tao.pdf
    img is a 2 dimensional image dataframe
    Need to be careful since images have RGB so they are actually 3 dimensional
    """
    simple_mask = np.array([[ -1, 0,  1]]) # Gx + j*Gy
    grad = signal.convolve2d(img, simple_mask, boundary='symm', mode='same')
    gradT = signal.convolve2d(img, np.transpose(simple_mask), boundary='symm', mode='same')
    return grad, gradT


def find_grad_orientation_strength(gradx, grady, is_unsigned=True):
    """
    calculates the gradient orientations and strength for each point based on gradient in the x (horizontal) and y
    (vertical) direction, where orientation (in rad) is arctan(grady/gradx) and strength is sqrt(gradx^2 + grady^2)
    :param gradx: numpy array like, gradient in the x direction (horizontal)
    :param grady: numpy array like, gradient in the y direction (vertical), must be same size of gradx
    :param is_unsigned: bool, whether the gradient is signed or not
    :return: two arrays, same sizes as gradx, first returns the gradient orientation and the second returns strength
    """

    orientation = np.arctan2(grady, gradx)
    strength = np.sqrt(np.power(gradx, 2) + np.power(grady, 2))
    if is_unsigned:
        orientation[grady < 0] = orientation[grady < 0] + np.pi

    return orientation, strength


def grad_histogram_by_cell(orientation, strength, orientation_bins, cell_size=8):
    """
    calculate the histogram of the gradient orientation, using strength as weight, in a given cell size.
    :param orientation: numpy array like, same size of strength, orientation of gradient at each pixel
    :param strength: numpy array like, strength of orientation at each pixel
    :param cell_size: cell size (square box) to investigate (total pixels to investigate = cell_size^2)
    :param orientation_bins: bins for the orientation histogram
    :return: numpy array like, for each pixel, contains bin_size features
    """

    image_size = orientation.shape
    # check if the image (a sub image, with its size = sliding window size) can be covered by non-overlapping cells
    if (image_size[0] % cell_size > 0 ) or (image_size[1] % cell_size > 0 ):
        print('sub-image size not divisible by cell size, HOG will not work properly')

    # calculate how many cells are there in each direction
    n_cells_x = int(image_size[1] / cell_size)
    n_cells_y = int(image_size[0] / cell_size)

    # setup empty feature array
    bin_size = len(orientation_bins) - 1
    cell_features = np.zeros((n_cells_y, n_cells_x, bin_size))
    # iterate through the two directions to calculate the features for each cell: histogram of thr gradient orientation
    # with strength as weight in each cell_size x cell_size cells
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # first extract the orientation and strength in the cell
            cell_orientation = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_strength = strength[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            # calculate histogram for the cell
            cell_features[i, j, :] = np.histogram(cell_orientation,
                                                  bins=orientation_bins,
                                                  weights=cell_strength)[0]

    return cell_features


def l2norm(k, epsilon):
    """
    calculates the L2 norm of matrix k with a constant error epsilon
    :param k: numpy array
    :param epsilon: float
    :return: 1D numpy array of normalized k
    """
    k = k.ravel()
    return k / np.sqrt(k ** 2 + epsilon ** 2)


def block_norm_of_cell_histogram(cell_histograms, epsilon=1, block_size=2, step_size=1):
    """
    calculates the block norms for each cell
    :param cell_histograms: numpy 3D array, histogram of gradient orientations in each cell
    :param epsilon: float, a constant for norm calculation to avoid division by 0
    :param block_size: the size of each block, overlap is half of the block size
    :param step_size: the step size of the blocks (i.e., block overlap area is block_size - step_size
    :return: numpy 3D array, block normalized cell gradient orientations
    """

    # find the number of cells in each direction
    cell_sizes = cell_histograms.shape
    # initialize the block normalized gradient orientation histogram...
    block_norm_histogram = np.zeros(shape=(int(cell_sizes[0]/step_size) - 1, int(cell_sizes[1]/step_size) - 1,
                                           cell_sizes[2] * block_size * block_size))
    # outer loop in the y direction, i.e., along a column
    for i in range(0, cell_sizes[0] - block_size, step_size):
        # inner loop in the x direction, i.e., along a row
        for j in range(0, cell_sizes[1] - block_size, step_size):
            block_norm_histogram[i, j, :] = l2norm(cell_histograms[i*step_size:i*step_size+block_size,
                                                   j*step_size:j*step_size+block_size], epsilon)

    return block_norm_histogram


def feature_extraction(img_dataframes, image_col_name='ImageMat', orientation=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys'):
    """
    extracts hog features from images stored in a dataframe
    :param img_dataframes: input image dataframe, must have a column containing the image data
    :param image_col_name: string, containing the column name for the image data
    :param orientation: int, number of orientation bins
    :param pixels_per_cell: list of two int, represents the cell size in row and column pixel count
    :param cells_per_block: list of two int, represents the block size in row and column cell count
    :param block_norm: string, method for block norm, check skimage.feature.hog for list of options
    :return: a list of HOG features, each entry correspond to each image in the input dataframe
    """

    features = []

    for index, img_df in img_dataframes.iterrows():
        input_image = img_df[image_col_name]
        # check if input image is in gray scale, if not (assuming it is in RGB) convert into grayscale
        if len(input_image.shape) > 2:
            input_image = rgb2gray(input_image)
        features.append(hog(input_image, orientations=orientation, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, block_norm=block_norm))

    return features

