from scm_bagging.scm_bagging_classifier import ScmBaggingClassifier


from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomUniform, CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "ScmBagging"

import numpy as np
from six import iteritems

MAX_INT = np.iinfo(np.int32).max


class ScmBagging(ScmBaggingClassifier, BaseMonoviewClassifier):
    """A Bagging classifier. for SetCoveringMachineClassifier()
    The base estimators are built on subsets of both samples
    and features.
    Parameters
    ----------
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator with
        replacement.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement.
        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.
    p_options : list of float with len =< n_estimators, default=[1.0]
        The estimators will be fitted with values of p found in p_options
        let k be k = n_estimators/len(p_options),
        the k first estimators will have p=p_options[0],
        the next k estimators will have p=p_options[1] and so on...
    random_state : int or RandomState, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    n_features_ : int
        The number of features when :meth:`fit` is performed.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    estim_features : list of arrays
        The subset of drawn features for each base estimator.

    Examples
    --------
    >>> @TODO

    References
    ----------
    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.
    .. [2] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    """

    def __init__(self,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 max_rules=10,
                 p_options=[0.316],
                 model_type="conjunction",
                 min_cq_combination=False,
                 min_cq_mu=10e-3,
                 random_state=None):
        if isinstance(p_options, float):
            p_options = [p_options]
        ScmBaggingClassifier.__init__(self, n_estimators=n_estimators,
                 max_samples=max_samples,
                 max_features=max_features,
                 max_rules=max_rules,
                 p_options=p_options,
                 model_type=model_type,
                 min_cq_combination=min_cq_combination,
                 min_cq_mu=min_cq_mu,
                 random_state=random_state)
        self.param_names = ["n_estimators", "max_rules", "max_samples", "max_features", "model_type", "p_options", "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300), CustomRandint(low=1, high=20),
                         CustomUniform(), CustomUniform(), ["conjunction", "disjunction"], CustomUniform(), [random_state]]
        self.weird_strings = {}

    def set_params(self, p_options=[0.316], **kwargs):
        if not isinstance(p_options, list):
            p_options = [p_options]
        kwargs["p_options"] = p_options
        for parameter, value in iteritems(kwargs):
            setattr(self, parameter, value)
        return self

    def get_interpretation(self, directory, base_file_name, y_test,
                           multi_class=False):
        self.features_importance()
        interpret_string = self.get_feature_importance(directory, base_file_name)
        return interpret_string
