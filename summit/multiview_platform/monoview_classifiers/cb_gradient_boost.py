import os
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble._gb_losses import ClassificationLossFunction, LOSS_FUNCTIONS, MultinomialDeviance, BinomialDeviance
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator
import warnings
import numbers
import scipy

from .. import metrics
from ..monoview.monoview_utils import BaseMonoviewClassifier, get_accuracy_graph
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint
from ..monoview.monoview_utils import change_label_to_minus
from .additions.BoostUtils import StumpsClassifiersGenerator, TreeClassifiersGenerator, BaseBoost

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "CBGradientBoosting"


class CustomDecisionTreeGB(DecisionTreeClassifier):
    def predict(self, X, check_input=True):
        y_pred = DecisionTreeClassifier.predict(self, X,
                                                check_input=check_input)
        return y_pred.reshape((y_pred.shape[0], 1)).astype(float)

class CBoundCompatibleGB(GradientBoostingClassifier, BaseBoost):

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        # if (self.loss not in self._SUPPORTED_LOSS+("cbound",)
        #         or self.loss not in LOSS_FUNCTIONS):
        #     raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))
        if self.loss == 'cbound':
            loss_class = CBoundLoss
        elif self.loss == 'deviance':
            loss_class = (MultinomialDeviance
                          if len(self.classes_) > 2
                          else BinomialDeviance)
        else:
            loss_class = LOSS_FUNCTIONS[self.loss]

        if self.loss in ('huber', 'quantile'):
            self.loss_ = loss_class(self.n_classes_, self.alpha)
        else:
            self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self.loss_.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == 'zero'):
                raise ValueError(
                    "The init parameter must be an estimator or 'zero'. "
                    "Got init={}".format(self.init)
                )

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change,
                          (numbers.Integral, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed"
                             % self.n_iter_no_change)

        if self.presort != 'deprecated':
            warnings.warn("The parameter 'presort' is deprecated and has no "
                          "effect. It will be removed in v0.24. You can "
                          "suppress this warning by not passing any value "
                          "to the 'presort' parameter. We also recommend "
                          "using HistGradientBoosting models instead.",
                          FutureWarning)


class CBoundLoss(ClassificationLossFunction):
    """Cbound loss for binary classification.

    Binary classification is a special case; here, we only need to
    fit one tree instead of ``n_classes`` trees.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    """
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class(es)"
                             .format(self.__class__.__name__, n_classes))
        # we only need to fit one tree for binary clf.
        super().__init__(n_classes=1)

    def init_estimator(self):
        # return the most common class, taking into account the samples
        # weights
        return DummyClassifier(strategy='prior')

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the cbound.

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble.

        sample_weight : 1d array , shape (n_samples,), optional
            Sample weights.
        """
        negs = np.ones(y.shape, dtype=int)
        negs[y==0] = -1
        margins = raw_predictions.ravel()*negs
        num = np.sum(margins)**2
        den = np.sum(raw_predictions.ravel()**2)
        return 1 - num/(den*y.shape[0])

    def negative_gradient(self, y, raw_predictions, **kargs):
        """Compute the residual (= negative gradient).

        Parameters
        ----------
        y : 1d array, shape (n_samples,)
            True labels.

        raw_predictions : 2d array, shape (n_samples, K)
            The raw_predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        """
        negs = np.ones(y.shape, dtype=int)
        negs[y == 0] = -1
        margins = raw_predictions.ravel()*negs
        num = 2* margins*(raw_predictions.ravel()-negs*raw_predictions.ravel()**2)
        den = y.shape[0]*(raw_predictions.ravel()**2**2)
        return -num/den

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        """Make a single Newton-Raphson step.

        our node estimate is given by:

            sum(w * (y - prob)) / sum(w * prob * (1 - prob))

        we take advantage that: y - prob = residual
        """
        terminal_region = np.where(terminal_regions == leaf)[0]
        raw_predictions = raw_predictions.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        neg = np.ones(y.shape)
        neg[y==0] = -1
        first_der = np.sum(-self.negative_gradient(y, raw_predictions))
        g_f = np.sum(neg*raw_predictions.ravel())
        n_f = np.sum(raw_predictions.ravel()**2)
        s_f = np.sum(raw_predictions.ravel())
        s_y = np.sum(neg)
        m = y.shape[0]

        scnd_der = 2*(g_f**2*n_f - 4*g_f**2*s_f**2 + 4*g_f*n_f*s_f*s_y - n_f**2*s_y**2)/(m*n_f**7)

        numerator = np.sum(2*self.negative_gradient(y, raw_predictions.ravel()))
        denominator = scnd_der

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = np.ones((raw_predictions.shape[0], 2), dtype=np.float64)
        proba[:, 1] = raw_predictions.ravel()
        proba[:, 0] -= proba[:, 1]
        return proba

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        probas[:, 0] *= -1
        best_indices = np.argmax(np.abs(probas), axis=1)
        raw_predictions = np.array([probas[i,best_index] for i, best_index in enumerate(best_indices)])
        return raw_predictions.reshape(-1, 1).astype(np.float64)


class CBGradientBoosting(CBoundCompatibleGB, BaseMonoviewClassifier):
    """
     This class is an adaptation of scikit-learn's `GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_


     """

    def __init__(self, random_state=None, loss="cbound", max_depth=1.0,
                 n_estimators=10,
                 init=CustomDecisionTreeGB(max_depth=1),
                 **kwargs):
        CBoundCompatibleGB.__init__(self, loss=loss, max_depth=max_depth,
                                    n_estimators=n_estimators, init=init,
                                    random_state=random_state)
        self.n_stumps=1
        self.self_complemented=True
        self.max_depth_pregen=3
        self.param_names = ["n_estimators", "max_depth"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=50, high=500),
                         CustomRandint(low=1, high=10), ]
        self.weird_strings = {}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None

    def format_X_y(self, X, y):
        """Formats the data  : X -the examples- and y -the labels- to be used properly by the algorithm """
        # Initialization
        y_neg = change_label_to_minus(y)
        y_neg = y_neg.reshape((y.shape[0], 1))
        return X, y_neg

    def pregen_voters(self, X, y=None, generator="Stumps"):
        if y is not None:
            neg_y = change_label_to_minus(y)
            if generator is "Stumps":
                self.estimators_generator = StumpsClassifiersGenerator(
                    n_stumps_per_attribute=self.n_stumps,
                    self_complemented=self.self_complemented)
            elif generator is "Trees":
                self.estimators_generator = TreeClassifiersGenerator(
                    n_trees=self.n_stumps, max_depth=self.max_depth_pregen)
            self.estimators_generator.fit(X, neg_y)
        else:
            neg_y = None
        classification_matrix = self._binary_classification_matrix(X)
        return classification_matrix, neg_y

    def fit(self, X, y, sample_weight=None, monitor=None):
        pregen_X, pregen_y = self.pregen_voters(X, y)
        begin = time.time()
        CBoundCompatibleGB.fit(self, pregen_X, pregen_y, sample_weight=sample_weight)
        print(self.train_score_)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = pregen_X.shape
        self.base_predictions = np.array(
            [estim[0].predict(pregen_X) for estim in self.estimators_])
        self.metrics = np.array(
            [self.plotted_metric.score(pred, pregen_y) for pred in
             self.staged_predict(pregen_X)])
        # self.bounds = np.array([np.prod(
        #     np.sqrt(1 - 4 * np.square(0.5 - self.estimator_errors_[:i + 1]))) for i
        #                         in range(self.estimator_errors_.shape[0])])
        return self

    def predict(self, X):
        begin = time.time()
        pregen_X,_ = self.pregen_voters(X)
        pred = GradientBoostingClassifier.predict(self, pregen_X)
        end = time.time()
        self.pred_time = end - begin
        if X.shape != self.train_shape:
            self.step_predictions = np.array(
                [step_pred for step_pred in self.staged_predict(pregen_X)])
        signs_array = np.array([int(x) for x in pred])
        signs_array[signs_array == -1] = 0
        return signs_array

    def get_interpretation(self, directory, base_file_name, y_test,
                           multi_class=False):
        interpretString = ""
        if multi_class:
            return interpretString
        else:
            interpretString += self.get_feature_importance(directory,
                                                           base_file_name)
            step_test_metrics = np.array(
                [self.plotted_metric.score(y_test, step_pred) for step_pred in
                 self.step_predictions])
            get_accuracy_graph(step_test_metrics, "AdaboostClassic",
                               os.path.join(directory, "test_metrics.png"),
                               self.plotted_metric_name, set="test")
            get_accuracy_graph(self.metrics, "AdaboostClassic",
                               os.path.join(directory, "metrics.png"),
                               self.plotted_metric_name)
            np.savetxt(
                os.path.join(directory, base_file_name + "test_metrics.csv"),
                step_test_metrics,
                delimiter=',')
            np.savetxt(
                os.path.join(directory, base_file_name + "train_metrics.csv"),
                self.metrics,
                delimiter=',')
            np.savetxt(os.path.join(directory, base_file_name + "times.csv"),
                       np.array([self.train_time, self.pred_time]),
                       delimiter=',')
            return interpretString

