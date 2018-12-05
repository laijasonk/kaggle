def calculate_overlap(indices_1, indices_2, size_1, size_2):
    """
    Calculates the overlap of two subimages

    :param indices_1: array with [row_index, column_index] of subimage 1
    :param indices_2: array with [row_index, column_index] of subimage 2
    :param size_1: array with [height, width] of subimage 1
    :param size_2: array with [height, width] of subimage 2
    :return: float giving the fraction of overlap
    """

    index_row_1 = indices_1[0]
    index_row_2 = indices_2[0]
    index_column_1 = indices_1[1]
    index_column_2 = indices_2[1]
    height_1 = size_1[0]
    height_2 = size_2[0]
    width_1 = size_1[1]
    width_2 = size_2[1]
    overlap_row_index = max(index_row_1, index_row_2)
    overlap_column_index = max(index_column_1, index_column_2)
    overlap_width = max(0, min(index_row_1+width_1, index_row_2+width_2)-overlap_row_index)
    overlap_height = max(0, min(index_column_1+height_1, index_column_2+height_2)-overlap_column_index)
    overlap = overlap_width*overlap_height
    total = height_1*width_1+height_2*width_2-overlap
    return overlap/total


def remove_boxes_with_NSM(positive_boxes, threshold):

    """
    Use the non-maximum suppression algorithm to remove overlapping boxes

    :param positive_boxes: Dataframe with subimage data
    :param threshold: Threshold parameter specifying the maximum allowable overlap
    :return: Dataframe with the filtered set of subimages
    """

    remaining_boxes = []
    boxes = positive_boxes

    while len(boxes)>0:
        # Add subimage with highest classification score to selection
        remaining_boxes.append(boxes.prediction.idxmax())

        # Calculate overlap with the subimage with the highest classification score
        compared_subimageanchor = boxes[boxes.index.isin([boxes.prediction.idxmax()])]["SubImageAnchor"].tolist()[0]
        compared_windowsize = boxes[boxes.index.isin([boxes.prediction.idxmax()])]["window_size"].tolist()[0]
        boxes = boxes.assign(overlap = boxes.apply(lambda row: calculate_overlap(row["SubImageAnchor"],
                                                                                 compared_subimageanchor,
                                                                                 row["window_size"],
                                                                                 compared_windowsize),
                                                   axis=1))

        # Remove all subimages which have more overlap than the threshold (including the subimage with the highest classification score)
        boxes = boxes[boxes.overlap < threshold]

    return positive_boxes[positive_boxes.index.isin(remaining_boxes)]