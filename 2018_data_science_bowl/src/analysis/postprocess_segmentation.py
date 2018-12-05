import numpy
import scipy
import skimage
import copy
import itertools
import matplotlib
import matplotlib.pyplot as plt
import math

def postprocess_segmentation(segment, image_array, pixel_array, threshold=1, output_plot=False, distance=3, scoring="circularity", max_angle=-1.5, calc_distance_convex_points=3, pen_ratio_circularity=1):
    """
    Postprocess segmentation

    :param segment: boolean mask is_part_of_segment
    :param image_array: image array
    :param threshold: float specifying threshold of minimum ratio between convex area and filled area in order to consider splitting
    :param output_plot: boolean to enable an output plot
    :param distance: integer specifying distance between
    :param scoring: string specifying metric to select splits
    :return: list with boolean masks of segments
    """

    background = numpy.mean(skimage.color.rgb2gray(image_array)[pixel_array==0])
    inv_background_image_array = numpy.abs(skimage.color.rgb2gray(image_array) - background)
    properties_segment = skimage.measure.regionprops(segment)
    if properties_segment[0].convex_area/properties_segment[0].filled_area>threshold:

        postprocessed_segment = segment

        points = non_convex_points(segment, properties_segment, distance, calc_distance_convex_points, max_angle=max_angle)
        point_combinations = list(map(list, itertools.combinations(points, 2)))

        prev_score = 0.2
        for i in point_combinations:
            rn, cn = skimage.draw.line(int(i[0][0]),
                                       int(i[0][1]),
                                       int(i[1][0]),
                                       int(i[1][1]))
            processed_segment = copy.copy(segment)
            processed_segment[rn, cn] = False
            segments = skimage.measure.label(processed_segment, neighbors=4)
            properties_segments = skimage.measure.regionprops(segments)
            if scoring=="convexity":
                score = numpy.sum([x.convex_area for x in properties_segments])
            elif scoring=="distance":
                score = numpy.sqrt(numpy.square(i[0][0]-i[1][0])+numpy.square(i[0][1]-i[1][1]))
            elif scoring == "intensity":
                score = numpy.mean([inv_background_image_array[rn[k], cn[k]] for k in range(len(rn))]) + 1*(len(rn)<2) + 1*(numpy.mean([segment[rn[k], cn[k]] for k in range(len(rn))])==0)
            elif scoring=="pointing_angle":
                vect1to2 = complex(i[1][0]-i[0][0], i[1][1]-i[0][1])
                vect2to1 = complex(i[0][0]-i[1][0], i[0][1]-i[1][1])
                angle1to2 = to_positive_angle(numpy.angle(vect1to2))
                angle2to1 = to_positive_angle(numpy.angle(vect2to1))
                angle1 = to_positive_angle(i[0][3])
                angle2 = to_positive_angle(i[1][3])
                dangle1 = abs(angle1to2-angle1)
                dangle2 = abs(angle2to1-angle2)
                score = (dangle1+dangle2)/2
            elif scoring=="circularity":
                contour_images = [numpy.pad(x.filled_image*1, (1, 1), mode="constant", constant_values=0) for x in properties_segments]
                contour_sizes = [len(skimage.measure.find_contours(x, level=0.5)[0]) for x in contour_images]
                equivalent_r = [x.equivalent_diameter for x in properties_segments]
                circle_contour_sizes =  [2*math.pi*x for x in equivalent_r]
                circularity = [contour_sizes[i]/circle_contour_sizes[i] for i in range(len(contour_sizes))]
                score=len(rn)+pen_ratio_circularity*numpy.mean(circularity)
            else:
                raise("Scoring method postprocessing not specified")
            if output_plot:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(segment)
                for j in points:
                    circ = matplotlib.patches.Circle((j[1], j[0]),
                                                     radius=2, facecolor='r', alpha=0.5)
                ax.add_patch(circ)
                ax.plot((i[0][1],i[1][1]), (i[0][0],i[1][0]))
                plt.show()

            if score < prev_score:
                prev_score = copy.copy(score)
                postprocessed_segment = processed_segment

    else:
        postprocessed_segment = segment

    return postprocessed_segment


def to_positive_angle(angle):
    """
    Convert radian angle to positive radian angle

    :param angle: float specifying angle
    :return: positive angle equal to the argument
    """
    if angle < 0:
        positive_angle = angle + math.pi
    else:
        positive_angle = angle
    return positive_angle


def non_convex_points(segment, properties_segment, distance, calc_distance_convex_points, min_size=2, factor_closest_distance=2, max_angle=-1.5, option="concavity"):
    """
    Calculate non convex points of a segment

    :param segment: boolean mask is_part_of_segment
    :param properties_segment: properties object of segment
    :param distance: integer specifying minimum distance to other points for concavity option
    :param min_size: integer specifying minimum segment size for closest option
    :param factor_closest_distance: integer specifying to filter all points that are at least this factor farther away than the closest point for closest option
    :param max_angle: maximum angle to detect a point as concavity for concavity option
    :param option: string specifying option to select non convex points
    :return: list with coordinates of non convex points
    """

    segment_copy = copy.copy(segment)
    chull = skimage.morphology.convex_hull.convex_hull_image(segment)
    negative_hull = segment_copy - chull
    if option=="concavity":
        hull_contour = 2*segment-negative_hull
        contours = skimage.measure.find_contours(hull_contour, level = 1.5)
        contours = [x for x in contours if x.shape[0] > 2*calc_distance_convex_points+1]
        contours_with_angle = [calc_concavity(x, calc_distance_convex_points) for x in contours]
        contours_with_angle = [filter_concavity_points(x, max_angle, distance) for x in contours_with_angle]
        points = list()
        for i in contours_with_angle:
            for j in range(i.shape[0]):
                points.append(i[j,0:4])
    if option=="closest":
        middle = properties_segment[0].centroid
        negative_hull = skimage.measure.label(negative_hull)
        hull_segmentation = [1*(negative_hull == value) for value in range(1, numpy.amax(negative_hull)+1, 1)]
        hull_segmentation = [x for x in hull_segmentation if numpy.sum(x)>min_size]
        points = [find_closest(x, middle) for x in hull_segmentation]
        if len(points)>0:
           closest_distance = min([x[1] for x in points])
           points = [x[0] for x in points if x[1]<factor_closest_distance*closest_distance]
    if option=="all":
        points = (properties_segment[0].convex_image*1 - properties_segment[0].filled_image*1) > 0
        points = [[numpy.where(points)[0][i], numpy.where(points)[1][i]] for i in range(len(numpy.where(points)[0]))]
    return points


def filter_concavity_points(contours_with_angle, max_angle, distance):
    """
    Filter a list with coordinates of points and their angles on local maxima smaller than max_angle

    :param contours_with_angle: list with coordinates of points and their angles
    :param max_angle: integer specifying maximum angle in radius to consider the point as a concavity
    :param distance: integer specifying minimum distance between detected concavity points
    :return: list with filtered concatity points
    """
    contours_with_angle = contours_with_angle[scipy.signal.argrelextrema(-contours_with_angle[:,2], numpy.greater, order=distance)]
    return contours_with_angle[contours_with_angle[:,2]<max_angle]



def find_closest(x, middle):
    """
    Determine the closest point in a list to the middle

    :param x: list with coordinates of points
    :param middle: coordinates of point to calculate distance to
    :return: list with coordinates of closest point and it's distance
    """

    old_distance = x.size
    points = [[numpy.where(x)[0][i], numpy.where(x)[1][i]] for i in range(len(numpy.where(x)[0]))]
    for i in points:
        distance = numpy.square(i[0]-middle[0])+numpy.square(i[1]-middle[1])
        if distance < old_distance:
            closest = i
            old_distance = distance

    return [closest, old_distance]


def calc_concavity(contour, distance):
    """
    Calculate the angles and the pointing angles in a contour compared to it's nieghbours at a specific distance

    :param contour: list with coordinates of points
    :param distance: integer specifying calculating distance
    :return:  list with coordinates, angles and pointing angles of points
    """

    angle = numpy.zeros((len(contour),1))
    pointing_angle = numpy.zeros((len(contour),1))
    for i in range(len(contour)):
        if i < distance or (i + distance) > len(contour)-1:
            angle[i,:] = 0
            pointing_angle[i,:] = 0
        else:
            v1 = contour[i-distance,:] - contour[i,:]
            v2 = contour[i+distance,:] - contour[i,:]
            angle[i,:] = numpy.arctan2(v1[0]*v2[0]+v1[1]*v2[1], v1[0]*v2[1]-v1[1]*v2[0])
            v_pointing=-v1-v2
            pointing_angle[i,:] = numpy.angle(complex(v_pointing[0], v_pointing[1]))
    contour = numpy.append(contour, angle, axis=1)
    return numpy.append(contour, pointing_angle, axis=1)


def generate_borders_by_canny(image):
    """
    Generate boolean mask is_border with canny

    :param image: image array
    :return: boolean mask is_center
    """

    edges=skimage.feature.canny(image, sigma=2)
    plt.imshow(skimage.morphology.closing(edges, skimage.morphology.disk(2)))
    plt.show()
    return edges
