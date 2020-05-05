from sklearn.neighbors import KNeighborsClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "KNN"


class KNN(KNeighborsClassifier, BaseMonoviewClassifier):
    """
     This class is an adaptation of scikit-learn's `KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_


     """

    def __init__(self, random_state=None, n_neighbors=5,
                 weights='uniform', algorithm='auto', p=2, **kwargs):
        KNeighborsClassifier.__init__(self,
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      algorithm=algorithm,
                                      p=p
                                      )
        self.param_names = ["n_neighbors", "weights", "algorithm", "p",
                            "random_state", ]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=10), ["uniform", "distance"],
                         ["auto", "ball_tree", "kd_tree", "brute"], [1, 2],
                         [random_state]]
        self.weird_strings = {}
        self.random_state = random_state
