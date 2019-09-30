import numpy as np
from sklearn.metrics import accuracy_score
import pkgutil

from .....utils.dataset import get_v
from ..... import monoview_classifiers
from ..LateFusion import LateFusionClassifier, getClassifiers, getConfig


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    nbView = classificationKWARGS["nbView"]
    paramsSets = []
    for _ in range(nIter):
        randomWeightsArray = randomState.random_sample(nbView)
        normalizedArray = randomWeightsArray / np.sum(randomWeightsArray)
        paramsSets.append([normalizedArray])
    return paramsSets


def getArgs(benchmark, args, views, viewsIndices, directory, resultsMonoview, classificationIndices):
    if args.FU_L_cl_names != ['']:
        args.FU_L_select_monoview = "user_defined"
    else:
        monoviewClassifierModulesNames = benchmark["monoview"]
        args.FU_L_cl_names = getClassifiers(args.FU_L_select_monoview, monoviewClassifierModulesNames, directory,
                                            viewsIndices, resultsMonoview, classificationIndices)
    monoviewClassifierModules = [getattr(monoview_classifiers, classifierName)
                                 for classifierName in args.FU_L_cl_names]
    if args.FU_L_cl_names == [""] and args.CL_type == ["multiview"]:
        raise AttributeError("You must perform monoview classification or specify "
                             "which monoview classifier to use Late fusion")
    if args.FU_L_cl_config != ['']:
        classifiersConfigs = [
            monoviewClassifierModule.getKWARGS([arg.split(":") for arg in classifierConfig.split(",")])
            for monoviewClassifierModule, classifierConfig
            in zip(monoviewClassifierModules, args.FU_L_cl_config)]
    else:
        classifiersConfigs = getConfig(args.FU_L_cl_names, resultsMonoview, viewsIndices)
    arguments = {"CL_type": "fusion",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FusionKWARGS": {"fusionType": "LateFusion",
                                  "fusionMethod": "BayesianInference",
                                  "classifiersNames": args.FU_L_cl_names,
                                  "classifiersConfigs": classifiersConfigs,
                                  'fusionMethodConfig': args.FU_L_method_config,
                                  'monoviewSelection': args.FU_L_select_monoview,
                                  "nbView": (len(viewsIndices))}}
    return [arguments]


class BayesianInference(LateFusionClassifier):
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, randomState, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                      kwargs["monoviewSelection"],
                                      NB_CORES=NB_CORES)
        if kwargs['fusionMethodConfig'][0] is None or kwargs['fusionMethodConfig'] == ['']:
            self.weights = np.array([1.0 for _ in kwargs['classifiersNames']])
        else:
            self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))
        self.needProbas = True

    def setParams(self, paramsSet):
        self.weights = paramsSet[0]

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if viewsIndices is None:
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if sum(self.weights) != 1.0:
            self.weights = self.weights / sum(self.weights)

        viewScores = []#np.zeros((nbView, len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]))
        for index, viewIndex in enumerate(viewsIndices):
            viewScores.append(np.power(
                self.monoviewClassifiers[index].predict_proba(get_v(DATASET, viewIndex, usedIndices)),
                self.weights[index]))
        viewScores = np.array(viewScores)
        predictedLabels = np.argmax(np.prod(viewScores, axis=0), axis=1)
        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames, monoviewClassifiersConfigs):
        configString = "with Bayesian Inference using a weight for each view : " + ", ".join(map(str, self.weights)) + \
                       "\n\t-With monoview classifiers : "
        for monoviewClassifier in self.monoviewClassifiers:

            configString += monoviewClassifier.getConfig()
        configString += "\n\t -Method used to select monoview classifiers : " + self.monoviewSelection
        return configString
