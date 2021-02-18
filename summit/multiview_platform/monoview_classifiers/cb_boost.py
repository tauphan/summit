from .additions.CBBoostUtils import CBBoostClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..monoview.monoview_utils import BaseMonoviewClassifier


classifier_class_name = "CBBoost"

class CBBoost(CBBoostClassifier, BaseMonoviewClassifier):
    """

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    n_max_iterations :

    n_stumps :

    kwargs : others arguments

    Attributes
    ----------
    param_names : names of parameter used for hyper parameter search

    distribs :

    classed_params :

    weird_strings :

    """
    def __init__(self, random_state=None, n_max_iterations=200, n_stumps=1,
                 **kwargs):

        super(CBBoost, self).__init__(n_max_iterations=n_max_iterations,
                                     random_state=random_state,
                                     self_complemented=True,
                                     twice_the_same=False,
                                     random_start=False,
                                     n_stumps=n_stumps,
                                     c_bound_sol=True,
                                     estimators_generator="Stumps",
                                     mincq_tracking=False
                                     )
        self.param_names = ["n_max_iterations", "n_stumps", "random_state"]
        self.distribs = [CustomRandint(low=2, high=500), [1,2,10],
                         [random_state]]
        self.classed_params = []
        self.weird_strings = {}

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     True
    #     """
    #     return True


    def get_interpretation(self, directory, base_file_name, y_test, multi_class=False):
        """
        return interpretation string

        Parameters
        ----------

        directory :

        y_test :

        Returns
        -------

        """
        interpret_string = self.getInterpretCBBoost(directory, base_file_name, y_test)
        interpret_string += self.get_feature_importance(directory, base_file_name)
        return interpret_string

    def get_name_for_fusion(self):
        """

        Returns
        -------
        string name of fusion
        """
        return "CBB"


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"n_stumps": args.CBB_stumps,
#                   "n_max_iterations": args.CBB_n_iter}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
