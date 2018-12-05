import sys, os
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/

import matplotlib.pylab as plt
import numpy
import pandas
import skimage

from utils import kaggle_io, kaggle_reader
from analysis import moving_window, tuning, feature_extractions

import sklearn.externals.joblib
import scipy.misc
import sklearn.pipeline as sk_pipeline
import sklearn.svm as sk_svm


class train_classifier( object ):

    def __init__( self, df_path, image_col_name, category_col_name, fixed_size, classifier_output,
                  feature_method, feature_options, image_processing_options_for_feature,
                  kernel='rbf', gamma='auto', C=1 ):
        """ The class constructor
        :param df_path: full path to the input roi DataFrame file (e.g. ../data/roi/roidf.h5)
        :param image_col_name: column name for the image data column
        :param category_col_name: column name for the image category (negative or positive) column
        :param fixed_size: 2x1 matrix containing the height and width of the images, format=[ height, width ]
        :param classifier_output: Path to the output classifier pickle file
        :param kernel: SVM kernel (e.g. rbf, linear, poly, sigmoid)
        :param gamma: kernel coefficient (e.g. 'auto', 0.001, etc.)
        :param C: penalty parameter (e.g. 1.0, 100)
        :return: None
        """
        # parameters that require input
        self.fixed_size = fixed_size
        self.classifier_output = classifier_output

        # parameters for feature extraction
        self.feature_method = feature_method
        self.feature_options = feature_options
        self.image_processing_options_for_feature = image_processing_options_for_feature

        # parameters that has default values, also SVM parameters
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

        # data related parameters, needs to be read from a file
        self.data = None
        self.image_col_name = image_col_name
        self.category_col_name = category_col_name
        self.load_data_from_dataframe_file(df_path)

        return

    def load_data_from_dataframe_file(self, df_path):
        """
        loads data from a saved panda DataFrame file
        :param df_path: file path to the saved DataFrame file
        :return: none
        """
        try:
            df = pandas.read_pickle(df_path)
            # filter out the ROIs with a size not equal to the input fixed_size
            self.data = df[df[self.image_col_name].apply(numpy.shape) == self.fixed_size]
        except IOError:
            print('Please check the training data file path and make sure the file is in pickle format.')
            print('tried: '+df_path)
        return

    def run( self ):

        """ Run the main code to train classifier and save to file

        :return: None

        """

        # extract features from dataframes
        # features = pandas.DataFrame( hog_classifier.feature_extraction(self.data,
        #                                                                image_col_name=self.image_col_name) )
        # print('Extracting features')
        features = feature_extractions.feature_extraction(self.data,
                                                          input_parameters=self.image_col_name,
                                                          method=self.feature_method,
                                                          extraction_options=self.feature_options,
                                                          image_process_options=self.image_processing_options_for_feature)

        # create target array which contains the correct answer
        target = pandas.DataFrame(numpy.repeat([True], len(features)))
        target[self.data[self.category_col_name].values == '0'] = False

        # initialize classifier
        print( "Initializing classifier" )
        classifier = sklearn.svm.SVC( kernel=self.kernel, gamma=self.gamma, C=self.C, verbose=True, probability=True )

        # train the classifier
        print( "Training classifier" )
        classifier.fit( features, target.values.ravel() )
        print()

        # save the classifier to be loaded later
        print( "Saving classifier to file, " + str( classifier_output ) )
        sklearn.externals.joblib.dump( classifier, classifier_output )



if __name__ == '__main__':

    # assign paths to different image directories
    df_path = os.path.realpath( os.path.join( os.getcwd(), '..', 'data', 'small_roi', 'train_dataframe.pkl' ) )

    # Size of the negative image
    fixed_size = (64, 64, 3)

    # feature extraction parameters
    method = 'gkhp'
    extraction_options = {'sigmas':[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 200]}
    image_process_options = {}

    # Information about the classifier
    # name of which classifier will be saved as
    kernel = 'sigmoid' # rbf (radial basis function), linear, poly, sigmoid
    gamma = '1e-2' # kernel coefficient (specify as 'auto' if unsure)
    C = 1 # penalty parameter (default is 1.0)
    classifier_output = os.path.join( '..', 'models', method+'_svm_'+kernel+'_C'+str(C)+'_prob.pkl' )

    model = sk_svm.SVC(probability=True)
    feature_extraction_model = feature_extractions.feature_extraction_class()
    pipeline_model = sk_pipeline.Pipeline(steps=[("extraction", feature_extraction_model),("classify", model)])
    print(pipeline_model.get_params().keys())
    tuned_parameters = [{'classify__kernel': ['sigmoid'],
                         'classify__gamma': [1, 0.3],
                         'classify__C': [1, 3],
                         'extraction__input_parameters': ['image_matrix'],
                         'extraction__method': ['gkhp', 'pixelval'],
                         'extraction__extraction_options': [{'sigmas':[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 200]}],
                         'extraction__image_process_options': [{}]}]

    main = train_classifier( df_path, 'image_matrix', 'is_positive', fixed_size, classifier_output,
                             method, extraction_options, image_process_options,
                             kernel, gamma, C )
    main, clf = tuning.tune_pipeline(main=main, model = pipeline_model, tuned_parameters=tuned_parameters, fraction = 0.01, diversity_vars=None, random_state=1)
