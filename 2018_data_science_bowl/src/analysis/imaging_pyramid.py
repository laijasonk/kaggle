import skimage
import pandas as pd
import numpy

from analysis import moving_window, NSM

def run_imaging_pyramid(classifier, image_array, feature_method, feature_options, image_options,  window_sizes,
                        step_sizes, nms_threshold, downscaling_factor, layers):

    """
    run imaging pyramid approach to identify positively identified subimages
    :param classifier: classifier object
    :param image_array: numpy array of image
    :param feature_method: method to extract features
    :param window_sizes: list object, size of the extraction window, in (height, width) format
    :param step_sizes: list object, specifying how far to move the window. It is in (horizontal step, vertical step).
    :param nms_threshold: scalar threshold for the NMS algorithm
    :param downscaling_factor: scalar giving the scaling factor for the imaging pyramid
    :return: Dataframe with positively identified boxes
    """

    pyramid = tuple(skimage.transform.pyramid_gaussian(image_array, downscale=downscaling_factor, max_layer=layers-1))
    boxes = pd.DataFrame()
    for i in range(len(pyramid)):
        new_boxes, new_reduced_boxes = moving_window.run_moving_window(classifier,
                                                         pyramid[i],
                                                         feature_method,
                                                         feature_options,
                                                         image_options,
                                                         [round(j) for j in window_sizes],
                                                         step_sizes, nms_threshold)
        factor = downscaling_factor**i
        new_boxes.window_size = new_boxes.window_size.apply(lambda x: [round(j*factor) for j in x])
        new_boxes.SubImageAnchor = new_boxes.SubImageAnchor.apply(lambda x: [round(j*factor) for j in x])
        boxes = boxes.append(new_boxes)

    boxes = boxes.reset_index()

    reduced_positive_boxes = NSM.remove_boxes_with_NSM(boxes, nms_threshold)
    moving_window.display_boxes(image_array, reduced_positive_boxes, boxes)

    return boxes, reduced_positive_boxes


def rescale_image(image_array, factor=1):

    """
    calls skimage.transform.rescale to rescale the image by the given factor
    :param image_array: numpy array, image data
    :param downsize_factor: float, scaling factor, <1 means downsize
    :return: rescaled image
    """

    return skimage.transform.rescale(image_array, factor)


def recover_box_size(box_anchor, box_shape, resized_image_shape, original_image_shape):

    """
    recover the box location in the original image from the resized image
    :param box_anchor: [upper-most row, left-most column]
    :param box_shape: (box height, box width)
    :param resized_image_shape:  (image height, image width)
    :param original_image_shape:  (image height, image width)
    :return: box anchor and shape in original image
    """

    # first calculate the scaling factor in the two dimensions
    scale_factors = numpy.array(original_image_shape) / numpy.array(resized_image_shape)
    # then apply the factor to the anchor and shape
    original_box_anchor = (numpy.array(box_anchor) * scale_factors).astype(int)
    original_box_shape = (numpy.array(box_shape) * scale_factors).astype(int)
    return original_box_anchor, original_box_shape