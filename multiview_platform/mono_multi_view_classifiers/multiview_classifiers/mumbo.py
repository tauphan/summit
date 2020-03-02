from sklearn.tree import DecisionTreeClassifier
from  sklearn.base import BaseEstimator
import numpy as np

from multimodal.boosting.mumbo import MumboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_examples_views_indices
from ..utils.base import base_boosting_estimators
from .. import monoview_classifiers

classifier_class_name = "Mumbo"

class Mumbo(BaseMultiviewClassifier, MumboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 best_view_mode="edge"):
        BaseMultiviewClassifier.__init__(self, random_state)
        base_estimator = self.set_base_estim_from_dict(base_estimator)
        MumboClassifier.__init__(self, base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,
                                    best_view_mode=best_view_mode)
        self.param_names = ["base_estimator", "n_estimators", "random_state", "best_view_mode"]
        self.distribs = [base_boosting_estimators,
                         CustomRandint(5,200), [random_state], ["edge", "error"]]

    def set_base_estim_from_dict(self, base_estim_dict):
        if base_estim_dict is None:
            base_estimator = DecisionTreeClassifier()
        elif isinstance(base_estim_dict, dict):
            estim_name = next(iter(base_estim_dict))
            estim_module = getattr(monoview_classifiers, estim_name)
            estim_class = getattr(estim_module,
                                  estim_module.classifier_class_name)
            base_estimator = estim_class(**base_estim_dict[estim_name])
        elif isinstance(base_estim_dict, BaseEstimator):
            base_estimator = base_estim_dict
        else:
            raise ValueError("base_estimator should be either None, a dictionary"
                             " or a BaseEstimator child object, "
                             "here it is {}".format(type(base_estim_dict)))
        return base_estimator

    def set_params(self, base_estimator=None, **params):
        """
        Sets the base estimator from a dict.
        :param base_estimator:
        :param params:
        :return:
        """
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier()
        elif isinstance(base_estimator, dict):
            self.base_estimator = self.set_base_estim_from_dict(base_estimator)
            MumboClassifier.set_params(self, **params)
        else:
            MumboClassifier.set_params(self, base_estimator=base_estimator, **params)


    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        self.used_views = view_indices
        self.view_names = [X.get_view_name(view_index)
                           for view_index in view_indices]
        numpy_X, view_limits = X.to_numpy_array(example_indices=train_indices,
                                                view_indices=view_indices)
        return MumboClassifier.fit(self, numpy_X, y[train_indices],
                                                view_limits)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                 example_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=example_indices,
                                                view_indices=view_indices)
        return MumboClassifier.predict(self, numpy_X)

    def get_interpretation(self, directory, labels, multiclass=False):
        self.view_importances = np.zeros(len(self.used_views))
        for best_view, estimator_weight in zip(self.best_views_, self.estimator_weights_):
            self.view_importances[best_view] += estimator_weight
        self.view_importances /= np.sum(self.view_importances)
        sorted_view_indices = np.argsort(-self.view_importances)
        interpret_string = "Mumbo used {} iterations to converge.".format(self.best_views_.shape[0])
        interpret_string+= "\n\nViews importance : \n"
        for view_index in sorted_view_indices:
            interpret_string+="- View {} ({}), importance {}\n".format(view_index,
                                                                      self.view_names[view_index],
                                                                      self.view_importances[view_index])
        interpret_string +="\n The boosting process selected views : \n" + ", ".join(map(str, self.best_views_))
        interpret_string+="\n\n With estimator weights : \n"+ "\n".join(map(str,self.estimator_weights_/np.sum(self.estimator_weights_)))
        return interpret_string
