from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import time

import numpy as np

from .. import metrics
from .additions.PregenUtils import PregenClassifier
from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier, \
    change_label_to_zero, CustomUniform

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "Bagging"

class Bagging(BaggingClassifier, BaseMonoviewClassifier,):
    """

    """
    def __init__(self, random_state=None, n_estimators=50, bootstrap_features=True, max_samples=1.0, max_features=1.0,
                 base_estimator=None, **kwargs):
        super(Bagging, self).__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap_features=bootstrap_features,
            base_estimator=base_estimator,
            max_features=max_features
        )
        self.param_names = ["n_estimators", "base_estimator",
                            "random_state", "bootstrap_features", "max_samples", "max_features"]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=500),
                         [DecisionTreeClassifier(max_depth=1)],
                         [random_state], [True], [1.0], CustomUniform()]
        self.weird_strings = {"base_estimator": "class_name"}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"

    def fit(self, X, y, sample_weight=None):
        """
        """
        self.max_features = float(self.max_features)
        if self.max_features == 0.0 or self.max_features*X.shape[1]==0:
            self.max_features = 1
        begin = time.time()
        super(Bagging, self).fit(X, y, sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = X.shape


    def predict(self, X):
        """

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------

        """
        begin = time.time()
        pred = super(Bagging, self).predict(X)
        end = time.time()
        self.pred_time = end - begin
        return change_label_to_zero(pred)

    def getInterpret(self, directory, y_test):
        interpretString = ""
        np.savetxt(directory + "times.csv",
                   np.array([self.train_time, self.pred_time]), delimiter=',')
        return interpretString


# def paramsToSet(nIter, random_state):
#     """Used for weighted linear early fusion to generate random search sets"""
#     paramsSet = []
#     for _ in range(nIter):
#         paramsSet.append({"n_estimators": random_state.randint(1, 500),
#                           "base_estimator": None})
#     return paramsSet
