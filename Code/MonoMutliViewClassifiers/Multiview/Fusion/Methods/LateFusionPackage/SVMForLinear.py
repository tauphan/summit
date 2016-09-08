from ...Methods.LateFusion import LateFusionClassifier
import MonoviewClassifiers
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from utils.Dataset import getV


def gridSearch(DATASET, classificationKWARGS, trainIndices, nIter=30):
    return None


class SVMForLinear(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.SVMClassifier = None

    def fit_hdf5(self, DATASET, trainIndices=None):
        if trainIndices == None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        nbViews = DATASET.get("Metadata").attrs["nbView"]
        for viewIndex in range(nbViews):
            monoviewClassifier = getattr(MonoviewClassifiers, self.monoviewClassifiersNames[viewIndex])
            self.monoviewClassifiers.append(
                monoviewClassifier.fit(getV(DATASET, viewIndex, trainIndices),
                                       DATASET.get("Labels")[trainIndices],
                                       NB_CORES=self.nbCores,
                                       **dict((str(configIndex), config) for configIndex, config in
                                              enumerate(self.monoviewClassifiersConfigs[viewIndex]))))
        self.SVMForLinearFusionFit(DATASET, usedIndices=trainIndices)

    def predict_hdf5(self, DATASET, usedIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            monoviewDecisions = np.zeros((len(usedIndices), DATASET.get("Metadata").attrs["nbView"]), dtype=int)
            for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
                monoviewClassifier = getattr(MonoviewClassifiers, self.monoviewClassifiersNames[viewIndex])
                monoviewDecisions[:, viewIndex] = self.monoviewClassifiers[viewIndex].predict(
                    getV(DATASET, viewIndex, usedIndices))
            predictedLabels = self.SVMClassifier.predict(monoviewDecisions)
        else:
            predictedLabels = []
        return predictedLabels

    def SVMForLinearFusionFit(self, DATASET, usedIndices=None):
        self.SVMClassifier = OneVsOneClassifier(SVC())
        monoViewDecisions = np.zeros((len(usedIndices), DATASET.get("Metadata").attrs["nbView"]), dtype=int)
        for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
            monoViewDecisions[:, viewIndex] = self.monoviewClassifiers[viewIndex].predict(
                getV(DATASET, viewIndex, usedIndices))

        self.SVMClassifier.fit(monoViewDecisions, DATASET.get("Labels")[usedIndices])

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with SVM for linear \n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString