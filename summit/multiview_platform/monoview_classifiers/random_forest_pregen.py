from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import time

import numpy as np

from .. import metrics
from .additions.PregenUtils import PregenClassifier
from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier, \
    change_label_to_zero
from ..utils.hyper_parameter_search import CustomRandint, CustomUniform

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "RandomForestPregen"

class RandomForestPregen(RandomForestClassifier, BaseMonoviewClassifier,
                    PregenClassifier):
    """

    """
    def __init__(self, random_state=None, n_estimators=50, criterion="gini", max_depth=1, n_stumps=1, estimators_generator="Stumps",
                 max_depth_pregen=1, self_complemeted=True,
                 **kwargs):
        super(RandomForestPregen, self).__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
        )
        self.param_names = ["n_estimators", "n_stumps",
                            "estimators_generator", "max_depth_pregen",
                            "random_state", "criterion", "max_depth", ]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=500),
                         [n_stumps],
                         ["Stumps",], CustomRandint(low=1, high=5),
                         [random_state], ["gini", "entropy"], CustomRandint(low=1, high=5), ]
        self.weird_strings = {}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None
        self.estimators_generator = estimators_generator
        self.max_depth_pregen = max_depth_pregen
        self.n_stumps = n_stumps
        self.self_complemented = self_complemeted

    def fit(self, X, y, sample_weight=None):
        """
        """
        begin = time.time()
        pregen_X, pregen_y = self.pregen_voters(X, y)
        super(RandomForestPregen, self).fit(pregen_X, pregen_y,
                                           sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = pregen_X.shape
        self.base_predictions = np.array(
            [change_label_to_zero(estim.predict(pregen_X)) for estim in
             self.estimators_])



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
        pregen_X, _ = self.pregen_voters(X)
        pred = super(RandomForestPregen, self).predict(pregen_X)
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
