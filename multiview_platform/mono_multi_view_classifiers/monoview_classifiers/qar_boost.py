from ..monoview.additions.CGDescUtils import ColumnGenerationClassifierQar
from ..monoview.monoview_utils import BaseMonoviewClassifier


class QarBoost(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, **kwargs):
        super(QarBoost, self).__init__(n_max_iterations=500,
                                       random_state=random_state,
                                       self_complemented=True,
                                       twice_the_same=True,
                                       c_bound_choice=True,
                                       random_start=False,
                                       n_stumps=10,
                                       use_r=True,
                                       c_bound_sol=False
                                       )
        # n_stumps_per_attribute=10,
        self.param_names = []
        self.distribs = []
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        return self.getInterpretQar(directory, y_test)

    def get_name_for_fusion(self):
        return "QB"


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet