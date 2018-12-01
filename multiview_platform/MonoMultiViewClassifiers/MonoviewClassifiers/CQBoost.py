from ..Monoview.MonoviewUtils import CustomUniform, CustomRandint, BaseMonoviewClassifier
from ..Monoview.Additions.CQBoostUtils import ColumnGenerationClassifier
from ..Monoview.Additions.BoostUtils import getInterpretBase


class CQBoost(ColumnGenerationClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, epsilon=1e-06, **kwargs):
        super(CQBoost, self).__init__(
            random_state=random_state,
            mu=mu,
            epsilon=epsilon
        )
        self.param_names = ["mu", "epsilon"]
        self.distribs = [CustomUniform(loc=0.5, state=1.0, multiplier="e-"),
                         CustomRandint(low=1, high=15, multiplier="e-")]
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        return getInterpretBase(self, directory, "CQBoost", self.voters_weights, )


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"mu": args.CQB_mu,
                  "epsilon": args.CQB_epsilon}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": 10**-randomState.uniform(0.5, 1.5),
                          "epsilon": 10**-randomState.randint(1, 15)})
    return paramsSet
