from ..Monoview.MonoviewUtils import BaseMonoviewClassifier
from ..Monoview.Additions.BoostUtils import getInterpretBase
from ..Monoview.Additions.QarBoostUtils import ColumnGenerationClassifierQar


class QarBoostv3(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, **kwargs):
        super(QarBoostv3, self).__init__(
            random_state=random_state,
            self_complemented=True,
            twice_the_same=False,
            old_fashioned=False,
            previous_vote_weighted=False,
            c_bound_choice=True,
            random_start=True,
            two_wieghts_problem=False,
            divided_ponderation=False,
            n_stumps_per_attribute=1
        )
        self.param_names = []
        self.distribs = []
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        return getInterpretBase(self, directory, "QarBoostv3", self.weights_, self.break_cause)

    def get_name_for_fusion(self):
        return "QBv3"

def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {}
    return kwargsDict

def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
