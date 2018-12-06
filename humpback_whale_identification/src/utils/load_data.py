#!/usr/bin/env python3

""" Test script to load data. 

    Function List:
        directory_into_file_list( indir )
        file_list_into_file_dataframe( inlist )
        csv_into_dataframe( incsv )

"""

import sys, os
import numpy, pandas
import skimage
import skimage.io
import skimage.transform

__author__ = 'Jason K Lai'
__contact__ = 'http://www.github.com/jklai'

class load_data( object ):

    def __init__( self ):

        """ The class constructor.

        """

        return None


    def directory_into_file_list( self, indir ):

        """ Load a directory of files (path) into a list of filenames

        :param indir: Path to the directory containing multiple image files
        :return: List of files

        """

        try:
            return [ f for f in os.listdir( indir ) if os.path.isfile( os.path.join( indir, f ) ) ]
        except:
            return False


    def file_list_into_file_dataframe( self, inlist ):

        """ Load a list of filenames into a 1 column dataframe

        :param inlist: List of filenames
        :return: Dataframe of filenames (1 column)

        """

        try:
            return pandas.DataFrame( { 'filename': inlist } )
        except:
            return False
    

    def csv_into_dataframe( self, incsv ):

        """ Load in a csv file (path) into a dataframe
        
        :param incsv: Path to csv file
        :return: Dataframe containing csv data

        """

        try:
            return pandas.read_csv( incsv, sep=',', header=0 )
        except:
            return False

    def read_image( self, image_path ):

        """ Load in an image path into a RGB numpy array
        
        :param image_path: Path to image file
        
        """

        image = skimage.io.imread( image_path, 0 )
        return image
        image = skimage.transform.resize( image, ( 300, 300 ) )
        return image

if __name__ == '__main__':
    
    # Train paths
    train_images = '../../data/train'
    train_csv = '../../data/train.csv'
    single_image = '../../data/train/0000e88ab.jpg'

    # Example: Call class
    loader = load_data()

    # Example: Load directory of images
    files_list = loader.directory_into_file_list( train_images )
    files_df = loader.file_list_into_file_dataframe( files_list )
    print( files_df.head( 10 ) )

    # Example: Load csv
    input_csv = loader.csv_into_dataframe( train_csv )
    print( input_csv.head( 10 ) )

    # Example: Read in image file
    input_image = loader.read_image( single_image )
    print( len( input_image ) )
    print( len( input_image[0] ) )
    print( len( input_image[0][0] ) )

