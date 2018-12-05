import numpy
import scipy
import skimage
import matplotlib.pyplot as plt
import copy

def generate_centers_by_preprocessing(image):
    """
    Use preprocessing to find expected centers of nuclei

    :param image: image array
    :return: boolean mask is_center
    """
    modified_image = skimage.morphology.opening(image, skimage.morphology.disk(10))

    distance_transform = scipy.ndimage.distance_transform_edt(modified_image)
    maxima1 = skimage.feature.peak_local_max(distance_transform, min_distance=3, indices=False)

    hat = skimage.morphology.white_tophat(image, skimage.morphology.disk(10))
    hat = skimage.morphology.opening(hat, skimage.morphology.disk(5))

    distance_transform = scipy.ndimage.distance_transform_edt(hat)
    maxima2 = skimage.feature.peak_local_max(distance_transform, min_distance=3, indices=False)

    maxima = maxima1 + maxima2

    plt.imshow(image)
    plt.imshow(maxima, alpha=0.3)

    return maxima

def ultimate_erosion(image, dilation=3):
    """
    Perform ultimate erosion over an image

    :param image: image array
    :return: ultimately eroded image array
    """

    centers = numpy.zeros((image.shape))

    labels = skimage.measure.label(image)
    connected_segments = [1*(labels==value) for value in range(1, numpy.amax(labels)+1, 1)]
    new_connected_segments = [0]*len(connected_segments)

    while not (numpy.sum(connected_segments) == numpy.sum(new_connected_segments) and len(connected_segments) == len(new_connected_segments)):
        new_connected_segments = copy.copy(connected_segments)
        connected_segments = erode_segment(new_connected_segments)

    centers = numpy.max(connected_segments, axis=0)
    centers = skimage.morphology.dilation(centers, skimage.morphology.disk(dilation))

    return centers


def erode_segment(connected_segments):
    """
    Erodes all segments in a list under specific conditions.
    In case the segment disappears, no erosion is applied.
    In case the segment is split into 2 segments by the erosion, the newly created segment is appended to the list.

    :param connected_segments: list with segments
    :return: list with eroded segments
    """

    next_connected_segments = [0]*len(connected_segments)

    for i in range(len(connected_segments)):
        new_segment = skimage.morphology.erosion(connected_segments[i], skimage.morphology.disk(1))
        new_labels = skimage.measure.label(new_segment)
        new_connected_segments = [1*(new_labels==value) for value in range(1, numpy.amax(new_labels)+1, 1)]
        if len(new_connected_segments)<1:
            next_connected_segments[i] = connected_segments[i]
        elif len(new_connected_segments)>1:
            next_connected_segments[i] = new_connected_segments[0]
            for j in range(1, len(new_connected_segments), 1):
                next_connected_segments.append(new_connected_segments[j])
        elif numpy.sum(new_segment) > 0:
            next_connected_segments[i] = new_segment
        else:
            raise("Unknown situation encountered")
    return next_connected_segments
