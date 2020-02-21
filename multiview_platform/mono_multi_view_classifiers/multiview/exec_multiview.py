import errno
import logging
import os
import os.path
import time

import h5py
import numpy as np

from . import analyze_results
from .multiview_utils import MultiviewResult
from .. import multiview_classifiers
from ..utils import hyper_parameter_search
from ..utils.multiclass import get_mc_estim

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def init_constants(kwargs, classification_indices, metrics,
                   name, nb_cores, k_folds,
                   dataset_var):
    """
    Used to init the constants
    Parameters
    ----------
    kwargs :

    classification_indices :

    metrics :

    name :

    nb_cores : nint number of cares to execute

    k_folds :

    dataset_var :  {array-like} shape (n_samples, n_features)
        dataset variable

    Returns
    -------
    tuple of (classifier_name, t_start, views_indices,
              classifier_config, views, learning_rate)
    """
    views = kwargs["view_names"]
    views_indices = kwargs["view_indices"]
    if not metrics:
        metrics = [["f1_score", None]]
    classifier_name = kwargs["classifier_name"]
    classifier_config = kwargs[classifier_name]
    learning_rate = len(classification_indices[0]) / float(
        (len(classification_indices[0]) + len(classification_indices[1])))
    t_start = time.time()
    logging.info("Info\t: Classification - Database : " + str(
        name) + " ; Views : " + ", ".join(views) +
                 " ; Algorithm : " + classifier_name + " ; Cores : " + str(
        nb_cores) + ", Train ratio : " + str(learning_rate) +
                 ", CV on " + str(k_folds.n_splits) + " folds")

    for view_index, view_name in zip(views_indices, views):
        logging.info("Info:\t Shape of " + str(view_name) + " :" + str(
            dataset_var.get_shape()))
    labels = dataset_var.get_labels()
    return classifier_name, t_start, views_indices, \
           classifier_config, views, learning_rate, labels


def save_results(classifier, labels_dictionary, string_analysis, views,
                 classifier_module,
                 classification_kargs, directory, learning_rate, name,
                 images_analysis):
    """
    Save results in derectory

    Parameters
    ----------

    classifier : classifier class

    labels_dictionary : dict dictionary of labels

    string_analysis : str

    views :

    classifier_module : module of the classifier

    classification_kargs :

    directory : str directory

    learning_rate :

    name :

    images_analysis :

    """
    labels_set = set(labels_dictionary.values())
    logging.info(string_analysis)
    # views_string = "-".join(views)
    views_string = "mv"
    cl_type_string = classifier.short_name
    output_file_name = os.path.join(directory, cl_type_string,
                                    cl_type_string + "-" + views_string + '-' + name)
    if not os.path.exists(os.path.dirname(output_file_name)):
        try:
            os.makedirs(os.path.dirname(output_file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    output_text_file = open(output_file_name + 'summary.txt', 'w')
    output_text_file.write(string_analysis)
    output_text_file.close()

    if images_analysis is not None:
        for image_name in images_analysis.keys():
            if os.path.isfile(output_file_name + image_name + ".png"):
                for i in range(1, 20):
                    test_file_name = output_file_name + image_name + "-" + str(
                        i) + ".png"
                    if not os.path.isfile(testFileName):
                        images_analysis[image_name].savefig(test_file_name,
                                                            transparent=True)
                        break

            images_analysis[image_name].savefig(
                output_file_name + image_name + '.png', transparent=True)


def exec_multiview_multicore(directory, core_index, name, learning_rate,
                             nb_folds,
                             database_type, path, labels_dictionary,
                             random_state, labels,
                             hyper_param_search=False, nb_cores=1, metrics=None,
                             n_iter=30, **arguments):
    """
    execute multiview process on

    Parameters
    ----------

    directory : indicate the directory

    core_index :

    name : name of the data file to perform

    learning_rate :

    nb_folds :

    database_type :

    path : path to the data name

    labels_dictionary

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    labels :

    hyper_param_search :

    nb_cores : in number of cores

    metrics : metric to use

    n_iter : int number of iterations

    arguments : others arguments

    Returns
    -------
    exec_multiview on directory, dataset_var, name, learning_rate, nb_folds, 1,
        database_type, path, labels_dictionary,
        random_state, labels,
        hyper_param_search=hyper_param_search, metrics=metrics,
        n_iter=n_iter, **arguments
    """
    """Used to load an HDF5 dataset_var for each parallel job and execute multiview classification"""
    dataset_var = h5py.File(path + name + str(core_index) + ".hdf5", "r")
    return exec_multiview(directory, dataset_var, name, learning_rate, nb_folds,
                          1,
                          database_type, path, labels_dictionary,
                          random_state, labels,
                          hyper_param_search=hyper_param_search,
                          metrics=metrics,
                          n_iter=n_iter, **arguments)


def exec_multiview(directory, dataset_var, name, classification_indices,
                   k_folds,
                   nb_cores, database_type, path,
                   labels_dictionary, random_state, labels,
                   hyper_param_search=False, metrics=None, n_iter=30, **kwargs):
    """Used to execute multiview classification and result analysis

    Parameters
    ----------

    directory : indicate the directory


    dataset_var :

    name

    classification_indices

    k_folds

    nb_cores

    database_type

    path

    labels_dictionary : dict dictionary of labels

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    labels

    hyper_param_search

    metrics

    n_iter : int number of iterations

    kwargs

    Returns
    -------

    ``MultiviewResult``
    """

    logging.debug("Start:\t Initialize constants")
    cl_type, \
    t_start, \
    views_indices, \
    classifier_config, \
    views, \
    learning_rate, \
    labels = init_constants(kwargs, classification_indices, metrics, name,
                            nb_cores, k_folds, dataset_var)
    logging.debug("Done:\t Initialize constants")

    extraction_time = time.time() - t_start
    logging.info("Info:\t Extraction duration " + str(extraction_time) + "s")

    logging.debug("Start:\t Getting train/test split")
    learning_indices, validation_indices = classification_indices
    logging.debug("Done:\t Getting train/test split")

    logging.debug("Start:\t Getting classifiers modules")
    # classifierPackage = getattr(multiview_classifiers,
    #                             CL_type)  # Permet d'appeler un module avec une string
    classifier_module = getattr(multiview_classifiers, cl_type)
    classifier_name = classifier_module.classifier_class_name
    # classifierClass = getattr(classifierModule, CL_type + "Class")
    logging.debug("Done:\t Getting classifiers modules")

    logging.debug("Start:\t Optimizing hyperparameters")
    if hyper_param_search != "None":
        classifier_config = hyper_parameter_search.search_best_settings(
            dataset_var, dataset_var.get_labels(), classifier_module,
            classifier_name,
            metrics[0], learning_indices, k_folds, random_state,
            directory, nb_cores=nb_cores, views_indices=views_indices,
            searching_tool=hyper_param_search, n_iter=n_iter,
            classifier_config=classifier_config)
    classifier = get_mc_estim(
        getattr(classifier_module, classifier_name)(random_state=random_state,
                                                    **classifier_config),
        random_state, multiview=True,
        y=dataset_var.get_labels())
    logging.debug("Done:\t Optimizing hyperparameters")
    logging.debug("Start:\t Fitting classifier")
    classifier.fit(dataset_var, dataset_var.get_labels(),
                   train_indices=learning_indices,
                   view_indices=views_indices)
    logging.debug("Done:\t Fitting classifier")

    logging.debug("Start:\t Predicting")
    pred_train_labels = classifier.predict(dataset_var,
                                           example_indices=learning_indices,
                                           view_indices=views_indices)
    pred_test_labels = classifier.predict(dataset_var,
                                          example_indices=validation_indices,
                                          view_indices=views_indices)
    full_labels = np.zeros(dataset_var.get_labels().shape, dtype=int) - 100
    full_labels[learning_indices] = pred_train_labels
    full_labels[validation_indices] = pred_test_labels
    logging.info("Done:\t Pertidcting")

    classification_time = time.time() - t_start
    logging.info(
        "Info:\t Classification duration " + str(extraction_time) + "s")

    # TODO: get better cltype

    logging.info("Start:\t Result Analysis for " + cl_type)
    times = (extraction_time, classification_time)
    string_analysis, images_analysis, metrics_scores = analyze_results.execute(
        classifier, pred_train_labels,
        pred_test_labels, dataset_var,
        classifier_config, classification_indices,
        labels_dictionary, views, nb_cores, times,
        name, k_folds,
        hyper_param_search, n_iter, metrics,
        views_indices, random_state, labels, classifier_module, directory)
    logging.info("Done:\t Result Analysis for " + cl_type)

    logging.debug("Start:\t Saving preds")
    save_results(classifier, labels_dictionary, string_analysis, views,
                 classifier_module,
                 classifier_config, directory,
                 learning_rate, name, images_analysis)
    logging.debug("Start:\t Saving preds")

    return MultiviewResult(cl_type, classifier_config, metrics_scores,
                           full_labels)
    # return CL_type, classificationKWARGS, metricsScores, fullLabels, testLabelsMulticlass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='This methods is used to execute a multiclass classification with one single view. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('-log', action='store_true',
                               help='Use option to activate Logging to Console')
    groupStandard.add_argument('--type', metavar='STRING', action='store',
                               help='Type of dataset', default=".hdf5")
    groupStandard.add_argument('--name', metavar='STRING', action='store',
                               help='Name of Database (default: %(default)s)',
                               default='DB')
    groupStandard.add_argument('--view', metavar='STRING', action='store',
                               help='Name of Feature for Classification (default: %(default)s)',
                               default='View0')
    groupStandard.add_argument('--pathF', metavar='STRING', action='store',
                               help='Path to the views (default: %(default)s)',
                               default='results-FeatExtr/')
    groupStandard.add_argument('--directory', metavar='STRING', action='store',
                               help='Path to the views (default: %(default)s)',
                               default='results-FeatExtr/')
    groupStandard.add_argument('--labels_dictionary', metavar='STRING',
                               action='store', nargs='+',
                               help='Name of classLabels CSV-file  (default: %(default)s)',
                               default='classLabels.csv')
    groupStandard.add_argument('--classificationIndices', metavar='STRING',
                               action='store',
                               help='Name of classLabels-Description CSV-file  (default: %(default)s)',
                               default='classLabels-Description.csv')
    groupStandard.add_argument('--nbCores', metavar='INT', action='store',
                               help='Number of cores, -1 for all', type=int,
                               default=1)
    groupStandard.add_argument('--randomState', metavar='INT', action='store',
                               help='Seed for the random state or pickable randomstate file',
                               default=42)
    groupStandard.add_argument('--hyper_param_search', metavar='STRING',
                               action='store',
                               help='The type of method used tosearch the best set of hyper parameters',
                               default='randomizedSearch')
    groupStandard.add_argument('--metrics', metavar='STRING', action='store',
                               nargs="+",
                               help='metrics used in the experimentation, the first will be the one used in CV',
                               default=[''])
    groupStandard.add_argument('--nIter', metavar='INT', action='store',
                               help='Number of itetarion in hyper parameter search',
                               type=int,
                               default=10)

    args = parser.parse_args()

    directory = args.directory
    name = args.name
    labels_dictionary = args.labels_dictionary
    classification_indices = args.classification_indices
    k_folds = args.k_folds
    nb_cores = args.nb_cores
    databaseType = None
    path = args.path_f
    random_state = args.random_state
    hyper_param_search = args.hyper_param_search
    metrics = args.metrics
    n_iter = args.n_iter
    kwargs = args.kwargs

    # Extract the data using MPI ?
    dataset_var = None
    labels = None  # (get from CSV ?)

    logfilename = "gen a good logfilename"

    logfile = os.path.join(directory, logfilename)
    if os.path.isfile(logfile + ".log"):
        for i in range(1, 20):
            testFileName = logfilename + "-" + str(i) + ".log"
            if not os.path.isfile(os.path.join(directory, testFileName)):
                logfile = os.path.join(directory, testFileName)
                break
    else:
        logfile += ".log"

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        filename=logfile, level=logging.DEBUG,
                        filemode='w')

    if args.log:
        logging.getLogger().addHandler(logging.StreamHandler())

    res = exec_multiview(directory, dataset_var, name, classification_indices,
                         k_folds,
                         nb_cores, databaseType, path,
                         labels_dictionary, random_state, labels,
                         hyper_param_search=hyper_param_search, metrics=metrics,
                         n_iter=n_iter, **kwargs)

    # Pickle the res
    # Go put your token
