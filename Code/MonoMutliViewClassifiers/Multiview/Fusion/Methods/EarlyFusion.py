#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
from utils.Dataset import getV


class EarlyFusionClassifier(object):
    def __init__(self, monoviewClassifierName, monoviewClassifierConfig, NB_CORES=1):
        self.monoviewClassifierName = monoviewClassifierName[0]
        self.monoviewClassifiersConfig = monoviewClassifierConfig[0]
        self.monoviewClassifier = None
        self.nbCores = NB_CORES
        self.monoviewData = None

    def makeMonoviewData_hdf5(self, DATASET, weights=None, usedIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        if not usedIndices:
            uesdIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if weights== None:
            weights = np.array([1/nbView for i in range(nbView)])
        if sum(weights)!=1:
            weights = weights/sum(weights)
        self.monoviewData = np.concatenate([weights[index]*getV(DATASET, viewIndex, usedIndices)
                                                         for index, viewIndex in enumerate(viewsIndices)], axis=1)

