from pyscm.scm import SetCoveringMachineClassifier as scm

from ..monoview.monoview_utils import CustomRandint, CustomUniform, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


# class Decis
classifier_class_name = "SCM"

class SCM(scm, BaseMonoviewClassifier):
    """
    SCM  Classifier
    Parameters
    ----------
    random_state (default : None)
    model_type : string (default: "conjunction")
    max_rules : int number maximum of rules (default : 10)
    p : float value(default : 0.1 )

    kwarg : others arguments

    Attributes
    ----------
    param_names

    distribs

    classed_params

    weird_strings

    """

    def __init__(self, random_state=None, model_type="conjunction",
                 max_rules=10, p=0.1, **kwargs):
        """

        Parameters
        ----------
        random_state
        model_type
        max_rules
        p
        kwargs
        """
        super(SCM, self).__init__(
            random_state=random_state,
            model_type=model_type,
            max_rules=max_rules,
            p=p
        )
        self.param_names = ["model_type", "max_rules", "p", "random_state"]
        self.distribs = [["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [random_state]]
        self.classed_params = []
        self.weird_strings = {}

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     return False in any case
    #     """
    #     return False

    def getInterpret(self, directory, y_test):
        interpretString = "Model used : " + str(self.model_)
        return interpretString


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"model_type": args.SCM_model_type,
#                   "p": args.SCM_p,
#                   "max_rules": args.SCM_max_rules}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append(
            {"model_type": random_state.choice(["conjunction", "disjunction"]),
             "max_rules": random_state.randint(1, 15),
             "p": random_state.random_sample()})
    return paramsSet
