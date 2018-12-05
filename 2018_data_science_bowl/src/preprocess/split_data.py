
""" Split the complete dataset into various different directories

Example: 
    sys.path.append( os.path.join( '.', 'src' ) ) # point to project src/
    from preprocess import split_data
    output = split_data.split_data( args ) 

Function list:
    split_data( imgids, split_dirnames, split_ratios, rseed=100 )

"""

import os
import numpy as np
import pandas as pd
import sklearn


def split_data( imgids, split_dirnames, split_ratios, rseed=100 ):

    """ Splits the image IDs into different sets. The results can be
    used to slice other dataframes as well.

    :param imgids: Unique ImageId for all images
    :param split_dirnames: List containing the names of the output split directories
    :param split_ratios: List containing the ratios for each split directory
    :param rseed: Random number generator seed
    :return: Image ID for the training set, validation set, and test set

    """

    if not len( split_dirnames ) == len( split_ratios ):
        raise Exception( 'The split_dirnames and split_ratios must be the same dimensions' )

    np.random.seed( rseed )
    nimgs = len( imgids )

    sep_loc = ( nimgs * np.array( split_ratios ) ).astype( int )
    sep_loc = np.cumsum( sep_loc )

    random_index = np.random.permutation( nimgs )

    split_imgids = []
    last_idx = 0
    for idx in range( len( split_dirnames ) ):

        begin_idx = last_idx
        try:
            end_idx = sep_loc[ idx ]
        except:
            end_idx = len( random_index )

        split_imgids.append( imgids[ random_index[ begin_idx:end_idx ] ] )
        last_idx = end_idx

    return split_imgids


def split_data_stratified(data, fraction, diversity_vars, random_state):
    if diversity_vars == None:
        diversity_df = np.zeros(len(data.index))
    else:
        diversity_df = data[diversity_vars]
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, train_size=fraction,
                                                         test_size=1 - fraction, random_state=random_state)
    split_indices = sss.split(np.zeros(len(data.index)), diversity_df)
    return next(split_indices)[0]

def slice_smaller_subset_of_data(data, fraction, diversity_vars, random_state):

    """ Create a subset of the data while maintaining diversity
    :param fraction: Fraction of data to keep
    :param diversity_vars: Variable names of which diversity should be maintained
    :param random_state: Basically the random seed (e.g. 1)
    :return: None
    """

    selected_indices = split_data_stratified(data = data, fraction=fraction, diversity_vars=diversity_vars, random_state=random_state)
    print("Slicing data")
    data = data[data.index.isin(data.index[selected_indices])]

    return data

if __name__ == '__main__':
    pass
