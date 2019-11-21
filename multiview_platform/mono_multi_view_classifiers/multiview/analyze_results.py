from .. import metrics

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def printMetricScore(metricScores, metric_list):
    """
    this function print the metrics scores

    Parameters
    ----------
    metricScores : the score of metrics

    metric_list : list of metrics

    Returns
    -------
    metric_score_string string constaining all metric results
    """
    metric_score_string = "\n\n"
    for metric in metric_list:
        metric_module = getattr(metrics, metric[0])
        if metric[1] is not None:
            metric_kwargs = dict((index, metricConfig) for index, metricConfig in
                                enumerate(metric[1]))
        else:
            metric_kwargs = {}
        metric_score_string += "\tFor " + metric_module.getConfig(
            **metric_kwargs) + " : "
        metric_score_string += "\n\t\t- Score on train : " + str(
            metricScores[metric[0]][0])
        metric_score_string += "\n\t\t- Score on test : " + str(
            metricScores[metric[0]][1])
        metric_score_string += "\n\n"
    return metric_score_string


def getTotalMetricScores(metric, trainLabels, testLabels, validationIndices,
                         learningIndices, labels):
    """

    Parameters
    ----------

    metric :

    trainLabels : labels of train

    testLabels :  labels of test

    validationIndices :

    learningIndices :

    labels :

    Returns
    -------
    list of [trainScore, testScore]
    """
    metricModule = getattr(metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in
                            enumerate(metric[1]))
    else:
        metricKWARGS = {}
    try:
        trainScore = metricModule.score(labels[learningIndices], trainLabels,
                                        **metricKWARGS)
    except:
        print(labels[learningIndices])
        print(trainLabels)
        import pdb;
        pdb.set_trace()
    testScore = metricModule.score(labels[validationIndices], testLabels,
                                   **metricKWARGS)
    return [trainScore, testScore]


def getMetricsScores(metrics, trainLabels, testLabels,
                     validationIndices, learningIndices, labels):
    metricsScores = {}
    for metric in metrics:
        metricsScores[metric[0]] = getTotalMetricScores(metric, trainLabels,
                                                        testLabels,
                                                        validationIndices,
                                                        learningIndices, labels)
    return metricsScores


def execute(classifier, trainLabels,
            testLabels, DATASET,
            classificationKWARGS, classificationIndices,
            labels_dictionary, views, nbCores, times,
            name, KFolds,
            hyper_param_search, nIter, metric_list,
            views_indices, random_state, labels, classifierModule):
    """

    Parameters
    ----------
    classifier : classifier used

    trainLabels : labels of train

    testLabels : labels of test

    DATASET :

    classificationKWARGS

    classificationIndices

    labels_dictionary

    views

    nbCores

    times

    name

    KFolds

    hyper_param_search

    nIter

    metric_list

    views_indices

    random_state

    labels

    classifierModule

    Returns
    -------
    retuern tuple of (stringAnalysis, imagesAnalysis, metricsScore)
    """
    classifier_name = classifier.short_name
    learningIndices, validationIndices, testIndicesMulticlass = classificationIndices

    metricModule = getattr(metrics, metric_list[0][0])
    if metric_list[0][1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in
                            enumerate(metric_list[0][1]))
    else:
        metricKWARGS = {}
    scoreOnTrain = metricModule.score(labels[learningIndices],
                                      labels[learningIndices],
                                      **metricKWARGS)
    scoreOnTest = metricModule.score(labels[validationIndices],
                                     testLabels, **metricKWARGS)

    stringAnalysis = "\t\tResult for multiview classification with " + classifier_name + \
                     "\n\n" + metric_list[0][0] + " :\n\t-On Train : " + str(
        scoreOnTrain) + "\n\t-On Test : " + str(
        scoreOnTest) + \
                     "\n\nDataset info :\n\t-Database name : " + name + "\n\t-Labels : " + \
                     ', '.join(
                         labels_dictionary.values()) + "\n\t-Views : " + ', '.join(
        views) + "\n\t-" + str(
        KFolds.n_splits) + \
                     " folds\n\nClassification configuration : \n\t-Algorithm used : " + classifier_name + " with : " + classifier.getConfig()

    metricsScores = getMetricsScores(metric_list, trainLabels, testLabels,
                                     validationIndices, learningIndices, labels)
    stringAnalysis += printMetricScore(metricsScores, metric_list)
    stringAnalysis += "\n\n Interpretation : \n\n" + classifier.get_interpretation()
    imagesAnalysis = {}
    return stringAnalysis, imagesAnalysis, metricsScores
