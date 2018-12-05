import math
import numpy
import scipy
import random
import matplotlib
import matplotlib.pyplot as plt
import skimage
import itertools
import copy

from analysis import postprocess_segmentation

def apply_segmentation(image, center_image, border_mask, image_array, type, preprocess=None, min_distance=1, centers=None,
                       preprocess_options=[1, 1], lim_segment_size=[1, math.inf], output_plot=False, post_iter=0,
                       post_threshold=1, max_angle=-1.5, calc_distance_convex_points=3, pen_ratio_circularity=1, scoring="circularity"):

    """
    Segments an image using segmentation

    :param image: boolean array is_part_of_nucleus
    :param center_image: boolean array is_center
    :param border_mask: boolean array is_border
    :param image_array: numpy array of original image
    :param type: string specifying type of segmentation
    :param preprocess: string specifying type of preprocessing
    :param min_distance: minimum distance between detected peaks and edges
    :param centers: array with booleans where entries with seed indices are True
    :param preprocess_options: array specifying parameters for preprocessing
    :param lim_segment_size: array specifying limits of segment size [min, max]
    :param output_plot: boolean indicating whether to plot result
    :param post_iter: integer specifying amount of postprocessing iterations
    :param post_threshold: float specifying postprocessing threshold
    :return: list with arrays describing masks
    """


    if not preprocess is None:
        if preprocess=="closing_opening":
            image = skimage.morphology.binary_closing(image, skimage.morphology.disk(preprocess_options[0]))
            image = skimage.morphology.binary_opening(image, skimage.morphology.disk(preprocess_options[1]))
        elif preprocess=="dilation_erosion":
            image = skimage.morphology.dilation(image, skimage.morphology.disk(preprocess_options[0]))
            image = skimage.morphology.erosion(image, skimage.morphology.disk(preprocess_options[1]))
        elif preprocess=="opening_closing":
            image = skimage.morphology.binary_opening(image, skimage.morphology.disk(preprocess_options[0]))
            image = skimage.morphology.binary_closing(image, skimage.morphology.disk(preprocess_options[1]))
        elif preprocess=="erosion_dilation":
            image = skimage.morphology.erosion(image, skimage.morphology.disk(preprocess_options[0]))
            image = skimage.morphology.dilation(image, skimage.morphology.disk(preprocess_options[1]))
        else:
            raise("Preprocessing method not recognized")

    distance_transform = scipy.ndimage.distance_transform_edt(center_image)

    if centers is None:
        maxima = skimage.feature.peak_local_max(distance_transform, min_distance=min_distance, indices=False)
    else:
        maxima = centers

    lbl = skimage.measure.label(maxima)
    if type == "watershed":
        ids = skimage.morphology.watershed(-distance_transform, lbl, mask=image)
    elif type == "randomwalker":
        ids = skimage.segmentation.random_walker(image, lbl*1, multichannel=True)
    elif type == "felzenszwalb":
        ids = skimage.segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50) # better defaults
        #ids = skimage.segmentation.felzenszwalb(image, scale=0.0001, sigma=0.8, min_size=10)
    elif type == "connected":
        ids = skimage.measure.label(image)
    else:
        raise("Type of segmentation not recognized.")

    if output_plot:
        col_list = numpy.linspace(0.001, 1, 1000)
        random.seed = 1
        random.shuffle(col_list)
        col_list = numpy.insert(col_list, obj=0, values=0)
        colormap = plt.cm.colors.ListedColormap(plt.cm.nipy_spectral(col_list))
        plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap="nipy_spectral")
        plt.axis('off')
        plt.title('Image')
        plt.subplot(122)
        plt.imshow(ids, cmap=colormap)
        plt.axis('off')
        plt.title('Segmentation')
        plt.show(block=False)

    segmentation = [1*(ids == value) for value in range(1, numpy.amax(ids)+1, 1)]
    if type == "connected" and post_iter>0:
        print( 'before post-processing: ' + str( len(segmentation) ) + ' segments' )
        for i in range(post_iter):
            segmentation = [postprocess_segmentation.postprocess_segmentation(x, image_array, image,
                                                                              threshold=post_threshold,
                                                                              distance=min_distance,
                                                                              calc_distance_convex_points=calc_distance_convex_points,
                                                                              max_angle=max_angle, scoring=scoring,
                                                                              pen_ratio_circularity=pen_ratio_circularity) for x in segmentation]
            image = sum(segmentation)
            ids = skimage.measure.label(image, neighbors=4)
            segmentation = [1*(ids == value) for value in range(1, numpy.amax(ids)+1, 1)]
        print( 'after post-processing: ' + str( len(segmentation) ) + ' segments' )
    filtered_segmentation = [x for x in segmentation if lim_segment_size[0]<numpy.sum(x)<lim_segment_size[1]]

    return filtered_segmentation


def plot_segmentation(image_array, segmentation):

    """
    Plot the segmentation over the original image

    :param image_array: image array of original image
    :param segmentation: list with arrays describing masks
    """

    col_list = numpy.linspace(0.001, 1, 1000)
    random.seed = 1
    random.shuffle(col_list)
    plt.figure()
    plt.imshow(image_array)
    plt.axis('off')
    for i in range(len(segmentation)):
        colormap = plt.cm.colors.ListedColormap(plt.cm.nipy_spectral(col_list[i:i+1]))
        colormap.set_under('k', alpha=0)
        plt.imshow(segmentation[i], cmap=colormap, alpha=0.5, clim=[0.5, 1])
    plt.show(block=False)
