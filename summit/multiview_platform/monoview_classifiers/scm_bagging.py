import numpy as np
from pyscm.scm import SetCoveringMachineClassifier as scm


from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomUniform, CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "ScmBaggingClassifier"

from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from pyscm import SetCoveringMachineClassifier

from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import accuracy_score
import numbers
import numpy as np
from six import iteritems
from warnings import warn
import logging

MAX_INT = np.iinfo(np.int32).max


class ScmBaggingClassifier(BaseEnsemble, ClassifierMixin, BaseMonoviewClassifier):
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
                 random_state=None):
        if isinstance(p_options, float):
            p_options = [p_options]
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_rules = max_rules
        self.p_options = p_options
        self.model_type = model_type
        self.random_state = random_state
        self.labels_to_binary = {}
        self.binary_to_labels = {}
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


    def p_for_estimators(self):
        """Return the value of p for each estimator to fit."""
        options_len = len(self.p_options)  # number of options
        estims_with_same_p = self.n_estimators // options_len  # nb of estimators to fit with the same p
        p_of_estims = []
        if options_len > 1:
            for k in range(options_len - 1):
                opt = self.p_options[k]  # an option
                p_of_estims = p_of_estims + ([
                                                 opt] * estims_with_same_p)  # estims_with_same_p estimators with p=opt
        p_of_estims = p_of_estims + ([self.p_options[-1]] * (
                    self.n_estimators - len(p_of_estims)))
        return p_of_estims

    def get_estimators(self):
        """Return the list of estimators of the classifier"""
        if hasattr(self, 'estimators'):
            return self.estimators
        else:
            return "not defined (model not fitted)"

    def get_hyperparams(self):
        """Return the setted hyperparameters"""
        hyperparams = {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'max_rules': self.max_rules,
            'p_options': self.p_options,
            'model_type': self.model_type,
            'random_state': self.random_state
        }
        return hyperparams

    # def set_params(self, **parameters):
    #     for parameter, value in iteritems(parameters):
    #         setattr(self, parameter, value)
    #     return self

    def labels_conversion(self, labels_list):
        l = list(set(labels_list))
        labels_dict = {c: idx for idx, c in enumerate(l)}
        if len(l) < 2:
            raise ValueError("Only 1 classe given to the model, needs 2")
        elif len(l) > 2:
            raise ValueError(
                "{} classes were given, multiclass prediction is not implemented".format(
                    len(l)))
        return labels_dict

    def fit(self, X, y):

        # Check if 2 classes are inputed and convert labels to binary labels
        self.labels_to_binary = self.labels_conversion(y)
        self.binary_to_labels = {bin_label: str_label for str_label, bin_label
                                 in self.labels_to_binary.items()}
        y = np.array([self.labels_to_binary[l] for l in y])
        self.n_features = X.shape[1]

        estimators = []
        self.estim_features = []
        max_rules = self.max_rules
        p_of_estims = self.p_for_estimators()
        model_type = self.model_type

        # seeds for reproductibility
        random_state = self.random_state
        random_state = check_random_state(random_state)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        self._seeds = seeds

        pop_samples, pop_features = X.shape
        max_samples, max_features = self.max_samples, self.max_features

        # validate max_samples
        if not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * pop_samples)
        if not (0 < max_samples <= pop_samples):
            raise ValueError("max_samples must be in (0, n_samples)")
        # store validated integer row sampling values
        self._max_samples = max_samples
        self._pop_samples = pop_samples

        # validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            max_features = self.max_features * pop_features
        else:
            raise ValueError("max_features must be int or float")
        if not (0 < max_features <= pop_features):
            raise ValueError("max_features must be in (0, n_features)")
        max_features = max(1, int(max_features))
        # store validated integer feature sampling values
        self._max_features = max_features
        self._pop_features = pop_features

        for k in range(self.n_estimators):
            p_param = p_of_estims[k]  # p param for the classifier to fit
            random_state = seeds[k]
            estim = SetCoveringMachineClassifier(p=p_param, max_rules=max_rules,
                                                 model_type=model_type,
                                                 random_state=random_state)
            feature_indices = sample_without_replacement(pop_features,
                                                         max_features,
                                                         random_state=random_state)
            samples_indices = sample_without_replacement(pop_samples,
                                                         max_samples,
                                                         random_state=random_state)
            Xk = (X[samples_indices])[:, feature_indices]
            yk = y[samples_indices]
            if len(list(set(yk))) < 2:
                raise ValueError(
                    "One of the subsamples contains elements from only 1 class, try increase max_samples value")
            estim.fit(Xk, yk)
            estimators.append(estim)
            self.estim_features.append(feature_indices)
        self.estimators = estimators

    def predict(self, X):
        results = []
        for (est, features_idx) in zip(self.estimators, self.estim_features):
            res = est.predict(X[:, features_idx])
            results.append(res)
        results = np.array(results)
        votes = np.mean(results, axis=0)
        predictions = np.array(np.round(votes, 0), dtype=int)
        predictions = np.array([self.binary_to_labels[l] for l in predictions])
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities
        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.
        Returns:
        --------
        p : array of shape = [n_examples, 2]
            The class probabilities for each example. Classes are ordered by lexicographic order.
        """
        warn(
            "ScmBaggingClassifier do not support probabilistic predictions. The returned values will be zero or one.",
            RuntimeWarning)
        # X = check_array(X) # TODO: check this
        pos_proba = self.predict(X)
        neg_proba = 1.0 - pos_proba
        return np.hstack((neg_proba.reshape(-1, 1), pos_proba.reshape(-1, 1)))

    def decision_rules(self):
        # @TODO : overview of the most important decision rules over estimators
        pass

    def features_importance(self):
        """
        Compute features importances in estimators rules
        Returns:
        --------
        importances : dict (feature id as key, importance as value)
            The mean importance of each feature over the estimators.
        """
        importances = {}  # sum of the feature/rule importances
        feature_id_occurences = {}  # number of occurences of a feature in subsamples
        for (estim, features_idx) in zip(self.estimators, self.estim_features):
            # increment the total occurences of the feature :
            for id_feat in features_idx:
                if id_feat in feature_id_occurences:
                    feature_id_occurences[id_feat] += 1
                else:
                    feature_id_occurences[id_feat] = 1
            # sum the rules importances :
            # rules_importances = estim.get_rules_importances() #activate it when pyscm will implement importance
            rules_importances = np.ones(len(
                estim.model_.rules))  # delete it when pyscm will implement importance
            for rule, importance in zip(estim.model_.rules, rules_importances):
                global_feat_id = features_idx[rule.feature_idx]
                if global_feat_id in importances:
                    importances[global_feat_id] += importance
                else:
                    importances[global_feat_id] = importance
        print(feature_id_occurences)
        importances = {k: round(v / feature_id_occurences[k], 3) for k, v in
                       importances.items()}
        self.feature_importances_ = np.array([importances[k]
                                              if k in importances else 0
                                              for k in range(self.n_features)])
        self.feature_importances_ /= np.sum(self.feature_importances_)
        return importances

    def get_estimators_indices(self):
        # get drawn indices along both sample and feature axes
        for seed in self._seeds:
            # operations accessing random_state must be performed identically
            # to those in 'fit'
            feature_indices = sample_without_replacement(self._pop_features,
                                                         self._max_features,
                                                         random_state=seed)
            samples_indices = sample_without_replacement(self._pop_samples,
                                                         self._max_samples,
                                                         random_state=seed)
            yield samples_indices

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_interpretation(self, directory, base_file_name, y_test,
                           multi_class=False):
        self.features_importance()
        interpret_string = self.get_feature_importance(directory, base_file_name)
        return interpret_string
