from .. import metrics

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def printMetricScore(metricScores, metrics):
    metricScoreString = "\n\n"
    for metric in metrics:
        metricModule = getattr(metrics, metric[0])
        if metric[1] is not None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in
                                enumerate(metric[1]))
        else:
            metricKWARGS = {}
        metricScoreString += "\tFor " + metricModule.getConfig(
            **metricKWARGS) + " : "
        metricScoreString += "\n\t\t- Score on train : " + str(
            metricScores[metric[0]][0])
        metricScoreString += "\n\t\t- Score on test : " + str(
            metricScores[metric[0]][1])
        metricScoreString += "\n\n"
    return metricScoreString


def getTotalMetricScores(metric, trainLabels, testLabels, validationIndices,
                         learningIndices, labels):
    metricModule = getattr(metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in
                            enumerate(metric[1]))
    else:
        metricKWARGS = {}
    trainScore = metricModule.score(labels[learningIndices], trainLabels,
                                        **metricKWARGS)
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
