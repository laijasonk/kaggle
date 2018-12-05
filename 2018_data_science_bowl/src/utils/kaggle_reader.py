
"""High level reader of kaggle input (use kaggle_io for variable handling)

Example: 
    sys.path.append( os.path.join( '.', 'src' ) ) # set path to project src/
    from utils import kaggle_reader
    output = kaggle_reader( '/path/to/csv' )

Function list:
    read_kaggle_csv( csv_path ):
    load_all_images( image_dir, mode='RGB' )
    load_all_raw_images( image_dir, mode='RGB' )
    find_mask_bounding_box( decodedlabels, imgDF )
    visual_check_boxes( boxDF, imgDF, imgid )
    get_masks_from_directory( mask_directory )

"""

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io 

sys.path.append( os.path.join( '..', '..', 'src' ) )
from utils import kaggle_io


def read_kaggle_csv( f ):

    """ Read the csv file into a dataframe the file should contain a
    column named ImageId and another named EncodedPixels

    :param f: File path to the csv file
    :return: Dataframe containing three columns: ImageId, EncodedPixels, DecodedPixels

    """

    codedlabels = pd.read_csv(f)
    # decode the pixel locations with the function decodepixels
    decodedlabels = codedlabels.assign(
            DecodedPixels=codedlabels.EncodedPixels.apply(_decodepixels))
    return decodedlabels


def load_all_images( image_directory, mode='RGB', fixed_size_only=None ):

    """ Modified and generalized version of Yao's load_all_raw_images
    that can load images recursively in the specified directory plus
    miscellaneous checks. See comments in load_all_raw_images for
    details.

    :param image_directory: Path to the image folder (e.g. ../data/mask/)
    :param mode: Choose the image mode to load as
    :param fixed_size_only: Set to [H,W] to check for a specific size, otherwise set to None
    :return: A dataframe that contains ImagePath and ImageMat as columns.

    """
   
    # Generalized version of load_all_raw_images
    path_list = glob.glob( os.path.join( image_directory, '**/*.png' ), recursive=True )
    images = [ kaggle_io.png_to_pixel_matrix( path, mode=mode ) for path in path_list ]
    out_dataframe = pd.DataFrame( list( zip( path_list, images ) ), columns=[ 'ImagePath', 'ImageMat' ] )

    # delete cases where not matching specified size
    if not fixed_size_only is None:
        delete_index = []
        for index, image in out_dataframe.iterrows():
            rows, cols = image.ImageMat.shape
            if not rows == fixed_size_only[0] or not cols == fixed_size_only[1]:
                delete_index.append( index )
        out_dataframe = out_dataframe.drop( delete_index )

    return out_dataframe


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
    images = [kaggle_io.png_to_pixel_matrix(os.path.join(imgspath, im, 'images', im + '.png'), mode=mode)
            for im in imdirs]
    return pd.DataFrame(list(zip(imdirs, images)), columns=['ImageId', 'ImageMat'])


def find_mask_bounding_box(decodedlabels, imgDF):

    """ Finds the bounding box (top and bottom row number, left and
    right column number) of the nucleis masks listed in decodedp
    (decoded pixel locations) in its corresponding image (in imgDF)

    :param decodedlabels: dataframe of pixel locations of nuclei, containing the decoded locations
    :param imgDF: dataframe containing ImageId and the array of imported image data in each row
    :return: dataframe where each entry is a nuclei, containing its bounding box information
        and corresponding image information

    """
    totalboxes = []
    imgids = []
    # 1. iterates through the images to get their shape information
    for index, img in imgDF.iterrows():
        # 2. then use the shape information to calculate pixel location, in [row, col] format,
        # for each nucleis (a row in decodedlabes) in the image
        dps = decodedlabels.DecodedPixels[decodedlabels.ImageId == img.ImageId]
        for dp in dps:
            # save image ID and box shape information for return
            imgids.append(img.ImageId)
            totalboxes.append(_extractbox(dp, img.ImageMat.shape))
    # combining return into a dataframe
    tboxes = np.array(totalboxes)
    return pd.DataFrame(list(zip(tboxes[:, 0], tboxes[:, 1], tboxes[:, 2], tboxes[:, 3],
                                 tboxes[:, 1] - tboxes[:, 0] + 1, tboxes[:, 3] - tboxes[:, 2] + 1,
                                 tboxes[:, 4], tboxes[:, 5], imgids)),
                        columns=['boxminR', 'boxmaxR', 'boxminC', 'boxmaxC',
                                 'boxH', 'boxW', 'imgW', 'imgH', 'ImageId'])


def visual_check_boxes(boxDF, imgDF, imgid):

    """ Simple plotting function to visually check the boxes on an image

    :param boxDF: dataframe containing boxes (output from find_mask_bounding_box)
    :param imgDF: dataframe containing all the images (output from load_all_images)
    :param imgid: image ID to check
    :return: None, just a plot

    """
    boxes = np.array(boxDF[boxDF.ImageId == imgid][['boxminR', 'boxmaxR', 'boxminC', 'boxmaxC']])
    plt.imshow(imgDF[imgDF.ImageId == imgid].ImageMat.values[0])
    plt.title(imgid)
    for box in boxes:
        plt.plot([box[2], box[2]], [box[0], box[1]], '-r')
        plt.plot([box[2], box[3]], [box[0], box[0]], '-r')


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
    output_masks = np.zeros( ( number_of_masks, height, width ), np.uint16 )
    output_masks[:,:,:] = raw_masks[:,:,:] / 255 

    return output_masks


def _decodepixels( ep ):

    """ A function that decodes the Kaggle nuclei location format

    :param ep: a string list containing the Kaggle format like "142 1 456 3...".
    :return: an array that expands the above format to contain the
        indexes for all the pixels that are in the nuclei (e.g., the above
        example becomes "142, 456, 457, 458, ...") 
    
    """

    splitcode = [int(s) for s in ep.split(' ')]
    codearray = [np.arange(splitcode[i], splitcode[i] + splitcode[i + 1]) 
            for i in range(0, len(splitcode), 2)]
    # before returning need to subtract by 1 to convert to python indexing
    return np.array([i for sublist in codearray for i in sublist]) - 1


def _extractbox(decodedp, imgshape):

    """ Extracting bounding boxes of each nuclei

    :param decodedp: 1-D array of decoded pixel locations of nuclei
    :param imgshape: the dimensions of the corresponding image array
    :return: a list of bounding box geometry

    """

    cols = np.ceil(decodedp / imgshape[0]).astype(int)
    rows = np.mod(decodedp, imgshape[0]).astype(int)
    return [min(rows), max(rows), min(cols), max(cols), imgshape[0], imgshape[1]]


if __name__ == '__main__':
    pass

