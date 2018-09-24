from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ..Monoview.MonoviewUtils import CustomRandint, BaseMonoviewClassifier
from ..Monoview.Additions.BoostUtils import get_accuracy_graph

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class Adaboost(AdaBoostClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_estimators=50,
                 base_estimator=None, **kwargs):
        super(Adaboost, self).__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            algorithm="SAMME"
            )
        self.param_names = ["n_estimators", "base_estimator"]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=500), [DecisionTreeClassifier(max_depth=1)]]
        self.weird_strings = {"base_estimator": "class_name"}

    def fit(self, X, y, sample_weight=None):
        super(Adaboost, self).fit(X, y, sample_weight=sample_weight)

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        interpretString = ""
        interpretString += self.getFeatureImportance(directory)
        interpretString += "\n\n Estimator error | Estimator weight\n"
        interpretString += "\n".join([str(error) +" | "+ str(weight/sum(self.estimator_weights_)) for error, weight in zip(self.estimator_errors_, self.estimator_weights_)])
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {'n_estimators': args.Ada_n_est,
                  'base_estimator': DecisionTreeClassifier(max_depth=1)}
    return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": random_state.randint(1, 500),
                          "base_estimator": None})
    return paramsSet
