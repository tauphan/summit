from .additions.CBBoostUtils import CBBoostClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..monoview.monoview_utils import BaseMonoviewClassifier

import numpy as np
import numpy.ma as ma
import math

classifier_class_name = "CBBoostGradientBoosting"

class CBBoostGradientBoosting(CBBoostClassifier, BaseMonoviewClassifier):
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

        CBBoostClassifier.__init__(self, n_max_iterations=n_max_iterations,
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
        self.distribs = [CustomRandint(low=2, high=500), [n_stumps],
                         [random_state]]
        self.classed_params = []
        self.weird_strings = {}

    def _find_new_voter(self, y_kernel_matrix, y):
        """Here, we solve the two_voters_mincq_problem for each potential new voter,
        and select the one that has the smallest minimum"""
        m = y_kernel_matrix.shape[0]
        previous_sum = np.multiply(y,
                                            self.previous_vote.reshape(m, 1))
        margin_old = np.sum(previous_sum)

        bad_margins = np.where(np.sum(y_kernel_matrix, axis=0) <= 0.0)[0]

        self.B2 = m
        self.B1s = np.sum(
            2 * np.multiply(previous_sum, y_kernel_matrix),
            axis=0)
        self.B0 = np.sum(previous_sum ** 2)

        self.A2s = np.sum(y_kernel_matrix, axis=0) ** 2
        self.A1s = np.sum(y_kernel_matrix, axis=0) * margin_old * 2
        self.A0 = margin_old ** 2

        C2s = (self.A1s * self.B2 - self.A2s * self.B1s)
        C1s = 2 * (self.A0 * self.B2 - self.A2s * self.B0)
        C0s = self.A0 * self.B1s - self.A1s * self.B0

        #### MODIF ####
        # Get the negative gradient
        neg_grads = -(1/m)*(2*y*margin_old*(margin_old*previous_sum-self.B0)/self.B0**2)
        ###############

        sols = np.zeros(C0s.shape) - 3
        sols[np.where(C2s != 0)[0]] = (-C1s[np.where(C2s != 0)[0]] + np.sqrt(
            C1s[np.where(C2s != 0)[0]] * C1s[np.where(C2s != 0)[0]] - 4 * C2s[
                np.where(C2s != 0)[0]] * C0s[np.where(C2s != 0)[0]])) / (
                                                  2 * C2s[
                                              np.where(C2s != 0)[0]])
        #### MODIF ####
        # The best hypothesis is chosen according to the gradient
        masked_grads, c_bounds = self.make_masked_grads(sols, bad_margins, neg_grads)
        best_hyp_index = np.argmax(masked_grads)
        ################
        if masked_grads[best_hyp_index] == ma.masked:
            return "No more pertinent voters", 0
        else:

            self.c_bounds.append(c_bounds[best_hyp_index])
            self.margins.append(math.sqrt(self.A2s[best_hyp_index] / m))
            self.disagreements.append(0.5 * self.B1s[best_hyp_index] / m)
            return sols[best_hyp_index], best_hyp_index

    def make_masked_grads(self, sols, bad_margins, neg_grads):
        """
        Masking the gradients in the forbidden directions
        Similar to cb-boost with gradients instead of cbounds

        Parameters
        ----------
        sols
        bad_margins
        neg_grads

        Returns
        -------

        """
        c_bounds = self.compute_c_bounds(sols)

        masked_grads = ma.array(
            np.sum(neg_grads * self.classification_matrix, axis=0),
            fill_value=np.inf)
        # If margin is negative
        masked_grads[bad_margins] = ma.masked
        # If weight is negative (self complemented voter is available)
        masked_grads[sols < 0] = ma.masked
        # If cbound is NaN
        masked_grads[np.isnan(c_bounds)] = ma.masked

        if not self.twice_the_same:
            masked_grads[self.chosen_columns_] = ma.masked
        return masked_grads, c_bounds

    def get_interpretation(self, directory, base_file_name,  y_test, multi_class=False):
        """
        return interpretation string

        Parameters
        ----------

        directory :

        y_test :

        Returns
        -------

        """
        return self.getInterpretCBBoost(directory, y_test)

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
