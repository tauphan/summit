import numpy as np

from ..monoview.additions.BoostUtils import getInterpretBase
from ..monoview.additions.CQBoostUtils import ColumnGenerationClassifier
from ..monoview.monoview_utils import CustomUniform, CustomRandint, \
    BaseMonoviewClassifier


class CQBoostTree(ColumnGenerationClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, epsilon=1e-06, n_stumps=1,
                 max_depth=2, n_max_iterations=100, **kwargs):
        super(CQBoostTree, self).__init__(
            random_state=random_state,
            mu=mu,
            epsilon=epsilon,
            estimators_generator="Trees",
            n_max_iterations=n_max_iterations
        )
        self.param_names = ["mu", "epsilon", "n_stumps", "random_state",
                            "max_depth", "n_max_iterations"]
        self.distribs = [CustomUniform(loc=0.5, state=1.0, multiplier="e-"),
                         CustomRandint(low=1, high=15, multiplier="e-"),
                         [n_stumps], [random_state], [max_depth],
                         [n_max_iterations]]
        self.classed_params = []
        self.weird_strings = {}
        self.n_stumps = n_stumps
        self.max_depth = max_depth
        if "nbCores" not in kwargs:
            self.nbCores = 1
        else:
            self.nbCores = kwargs["nbCores"]

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        np.savetxt(directory + "train_metrics.csv", self.train_metrics,
                   delimiter=',')
        np.savetxt(directory + "c_bounds.csv", self.c_bounds,
                   delimiter=',')
        np.savetxt(directory + "y_test_step.csv", self.step_decisions,
                   delimiter=',')
        step_metrics = []
        for step_index in range(self.step_decisions.shape[1] - 1):
            step_metrics.append(self.plotted_metric.score(y_test,
                                                          self.step_decisions[:,
                                                          step_index]))
        step_metrics = np.array(step_metrics)
        np.savetxt(directory + "step_test_metrics.csv", step_metrics,
                   delimiter=',')
        return getInterpretBase(self, directory, "CQBoost", self.weights_,
                                y_test)


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"mu": args.CQBT_mu,
#                   "epsilon": args.CQBT_epsilon,
#                   "n_stumps": args.CQBT_trees,
#                   "max_depth": args.CQBT_max_depth,
#                   "n_max_iterations": args.CQBT_n_iter}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": 10 ** -randomState.uniform(0.5, 1.5),
                          "epsilon": 10 ** -randomState.randint(1, 15)})
    return paramsSet