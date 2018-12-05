import numpy
import scipy
import sklearn
import matplotlib.pyplot as plt
import sys, os
import ast, configparser
import skimage.morphology
sys.path.append( os.path.join( '..', 'src' ) ) # set path to project src/
from analysis import moving_window, image_segmentation, kaggle_evaluation, hog_classifier
from preprocess import image_processing

if __name__ == '__main__':

    # select a random image from training set to scan
    numpy.random.seed(6)
    image_root = os.path.join('..', 'data', 'split', 'train')
    imdirs = [d for d in os.listdir(image_root)
              if os.path.isdir(os.path.join(image_root, d))]
    test_image_root = numpy.random.choice(imdirs)
    image_array = scipy.misc.imread(os.path.join(image_root, test_image_root, 'images', test_image_root+'.png'))
    expected_masks_directory = os.path.join(image_root, test_image_root, 'masks')
    preprocess_options = [3, 3]
    min_distance = 5
    lim_segment_size = [3, image_array.size/20]

    config = configparser.ConfigParser(allow_no_value=True)

    # finding centers of nuclei
    # config.read('../config/gkhp_svm_tuning_neg8_clahe.cfg')
    #config.read('../config/nn_adam_200_100_nuclei_center_nobuffer_gray_trim.cfg')
    config.read('../config/nn_adam_20_20_20_20_20_nuclei_center_nobuffer.cfg')
    # config.read('../config/gkhp_svm_tuning_center3px_10x10_clahe.cfg')
    center_classifier = sklearn.externals.joblib.load(ast.literal_eval(config.get('paths', 'classifier_pkl')))
    feature_method = ast.literal_eval(config.get('features', 'method'))
    feature_options = ast.literal_eval(config.get('features', 'feature_options'))
    image_options = ast.literal_eval(config.get('features', 'image_processing_options'))
    window_sizes = ast.literal_eval(config.get('moving_window', 'window_size'))
    step_sizes = ast.literal_eval(config.get('moving_window', 'step_size'))
    nms_threshold = ast.literal_eval(config.get('moving_window', 'nms_threshold'))

    center_boxes, center_reduced_positive_boxes = moving_window.run_imaging_pyramid(center_classifier,
                                                                                    image_array,
                                                                                    feature_method,
                                                                                    feature_options,
                                                                                    image_options,
                                                                                    window_sizes,
                                                                                    step_sizes,
                                                                                    nms_threshold,
                                                                                    downscaling_factor=1.5,
                                                                                    layers=2)
    centers = image_segmentation.boxes_to_centers(center_reduced_positive_boxes, image_array.shape)
    replacement = [1, 1]
    # create a center_mask that is supposed to be high confidence nuclei pixels (thus generated with
    # classifier that targets center of nulcei. Use this to calculate the distance. The idea is that
    # this distance map will help separating touching nulcies compared to a distance map generated
    # from the mask that will be generated below (from a classifier that tries to catch all nuclei
    # pixels).
    center_mask = moving_window.boxes_to_mask(center_boxes, image_array, replacement)
    center_mask = skimage.morphology.dilation(center_mask, skimage.morphology.disk(step_sizes[0]+1))
    # center_mask = skimage.morphology.closing(mask, skimage.morphology.disk(step_sizes[0]+1))
    # center_mask = skimage.morphology.erosion(center_mask, skimage.morphology.disk(2*step_sizes[0]+2))

    # extracting pixels in nuclei
    config.read('../config/gkhp_svm_tuning_center3px_10x10_clahe.cfg')
    # config.read('../config/gkhp_svm_tuning_center3px_10x10.cfg')
    classifier = sklearn.externals.joblib.load(ast.literal_eval(config.get('paths', 'classifier_pkl')))
    feature_method = ast.literal_eval(config.get('features', 'method'))
    feature_options = ast.literal_eval(config.get('features', 'feature_options'))
    image_options = ast.literal_eval(config.get('features', 'image_processing_options'))
    window_sizes = ast.literal_eval(config.get('moving_window', 'window_size'))
    step_sizes = ast.literal_eval(config.get('moving_window', 'step_size'))
    nms_threshold = ast.literal_eval(config.get('moving_window', 'nms_threshold'))
    boxes, reduced_positive_boxes = moving_window.run_moving_window(classifier,
                                                                    image_array,
                                                                    feature_method,
                                                                    feature_options,
                                                                    image_options,
                                                                    window_sizes,
                                                                    step_sizes,
                                                                    1)

    # replacement = [step_sizes[0], step_sizes[1]]
    replacement = [1, 1]
    nuclei_mask = moving_window.boxes_to_mask(boxes, image_array, replacement)
    # try to use dilation instead of replacement to filling in the nuclei
    # nuclei_mask = skimage.morphology.dilation(mask, skimage.morphology.disk(step_sizes[0]))
    # center_mask = skimage.morphology.closing(mask, skimage.morphology.disk(step_sizes[0]+1))
    # center_mask = skimage.morphology.erosion(center_mask, skimage.morphology.disk(2*step_sizes[0]+2))
    # try to use a processed image to operate, it is not a good idea...at least with this simple processing
    enhanced_image = image_processing.process_image(image_array, {'rgb2gray': None, 'trim': [1, 99], 'norm': 'clahe'})
    gradx, grady = hog_classifier.derivative_mask_1d_centered(enhanced_image)
    orientation, strength = hog_classifier.find_grad_orientation_strength(gradx, grady)
    # mask = mask*0.5 + strength*0.5
    center_mask = center_mask * nuclei_mask
    centers = (centers.astype(int) * nuclei_mask).astype(bool)
    plt.imshow(center_mask+centers.astype(int))
    plt.show(block=False)
    segmentation = image_segmentation.apply_segmentation(nuclei_mask,
                                                         center_mask,
                                                         type="watershed",
                                                         preprocess=None,
                                                         min_distance=min_distance,
                                                         centers=centers,
                                                         preprocess_options=preprocess_options,
                                                         lim_segment_size=lim_segment_size,
                                                         output_plot=True)
    image_segmentation.plot_segmentation(image_array, segmentation)

    evaluation = kaggle_evaluation.KaggleEvaluation()
    evaluation.set_predicted_masks(segmentation)
    evaluation.set_expected_masks_from_directory( expected_masks_directory )
    evaluation.calculate_iou()
    evaluation.calculate_score_with_thresholds()
    evaluation.print_table()
    plt.show()
