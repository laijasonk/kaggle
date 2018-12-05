
"""Script that splits the raw images into different sets

NOTE: Change the variables on bottom of script before running

"""

import sys
import os
import shutil
import zipfile

sys.path.insert(0, os.path.join('..','src'))
from utils import kaggle_reader
from preprocess import split_data


class split_raw( object ):
    
    def __init__( self, data_path, split_path, train_image_dir, train_zip_file=None, train_csv_file=None, split_dirnames=['train','test'], split_ratios=[0.7,0.3] ):

        """ The class constructor

        :param data_path: Path to the input raw directory (e.g ../data/raw)
        :param split_path: Path to the output split directory (e.g. ../data/split)
        :param train_zip_file: Path to the zip file
        :param train_csv_file: Path to the csv file
        :param train_image_dir: Path to the directory containing the images
        :param split_dirnames: List containing the names of the output split directories
        :param split_ratios: List containing the ratios for each split directory
        :return: None

        """

        self.data_path = data_path
        self.split_path = split_path

        self.train_zip_file = train_zip_file
        self.train_csv_file = train_csv_file
        self.train_image_dir = train_image_dir

        self.split_dirnames = split_dirnames
        self.split_ratios = split_ratios

        return


    def run( self ):
        
        """ Run the main code to split the raw data into the appropriate
        split directories

        :return: None

        """

        # optional functions
        #if not self.train_zip_file = None:
        #    self.unzip_to_raw( self.train_zip_file )
        #if not self.train_csv_file = None:
        #    labels = self.load_labels( self.train_csv_file )

        # load all the images
        imgDF = kaggle_reader.load_all_raw_images( 
                self.train_image_dir, mode='L' )

        # split data randomly
        split_results = split_data.split_data( 
                imgDF.ImageId.values, self.split_dirnames, self.split_ratios, rseed=100 )

        # loop through the dirnames and copy the images to that directory
        for idx in range( len( self.split_dirnames ) ):
            for imgid in split_results[ idx ]:
                shutil.copytree( os.path.join( self.train_image_dir, str(imgid) ), os.path.join( self.split_path, self.split_dirnames[ idx ], str(imgid) ) )


    def unzip_to_raw( self, input_zip_file ):

        """ Extract the files from a zip file

        :param input_zip_file: Path to the zip file
        :return: None

        """

        zip_ref = zipfile.ZipFile( input_zip_file, 'r' )
        zip_ref.extractall( os.path.join( self.data_path, 'raw') )


    def load_labels( self, input_csv_file ):

        """ Load nuclei pixel labels from the csv file. This data set
        could be useful for a more careful splitting of datasets. But it
        is probably better left silent until we get some early
        analytical results to determine if this step is necessary

        :param input_zip_file: Path to the zip file
        :return: decodedlabels

        """

        f = input_csv_file
        decodedlabels = kaggle_reader.read_kaggle_csv( f )

        return decodedlabels


if __name__ == '__main__':

    data_path = os.path.join( '..', 'data' )
    split_path = os.path.join( data_path, 'split' )

    stage1_train_image_dir = os.path.join( data_path, 'raw', 'stage1_train' )
    stage1_train_zip_file = os.path.join( data_path, 'raw', 'stage1_train.zip' )
    stage1_train_csv_file = os.path.join( data_path, 'raw', 'stage1_train_labels.csv' )

    # To generate small versions for testing
    #split_path = os.path.join( data_path, 'small_split' )
    #stage1_train_image_dir = os.path.join( data_path, 'small_raw', 'stage1_train' )
    #stage1_train_zip_file = os.path.join( data_path, 'small_raw', 'stage1_train.zip' )
    #stage1_train_csv_file = os.path.join( data_path, 'small_raw', 'stage1_train_labels.csv' )

    # example format: [ 'train', 'test' ]
    split_dirnames = [ 'train', 'test' ]

    # example format: [ 0.7, 0.3 ]
    split_ratios = [ 0.7, 0.3 ]

    main = split_raw( data_path=data_path, split_path=split_path, train_image_dir=stage1_train_image_dir, split_dirnames=split_dirnames, split_ratios=split_ratios )
    main.run()

