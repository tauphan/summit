from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import numpy as np
import os

from multimodal.boosting.mumbo import MumboClassifier

from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_samples_views_indices
from ..utils.base import base_boosting_estimators
from ..utils.organization import secure_file_path
from .. import monoview_classifiers

classifier_class_name = "Mumbo"

class Mumbo(BaseMultiviewClassifier, MumboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 best_view_mode="edge", **kwargs):
        BaseMultiviewClassifier.__init__(self, random_state)
        if type(base_estimator) is list:
            if type(base_estimator[0]) is dict:
                base_estimator = [self.set_base_estim_from_dict(estim, **kwargs) for estim in base_estimator]
            elif isinstance(base_estimator[0], BaseEstimator):
                base_estimator = base_estimator
            else:
                raise ValueError("base_estimator should ba a list of dict or a sklearn classifier list")
        else:
            base_estimator = self.set_base_estim_from_dict(base_estimator, **kwargs)
        MumboClassifier.__init__(self, base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,
                                    best_view_mode=best_view_mode)
        self.param_names = ["base_estimator", "n_estimators", "random_state", "best_view_mode"]
        self.distribs = [base_boosting_estimators,
                         CustomRandint(5,200), [random_state], ["edge", "error"]]

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
        train_indices, view_indices = get_samples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        self.used_views = view_indices
        self.view_names = [X.get_view_name(view_index)
                           for view_index in view_indices]
        numpy_X, view_limits = X.to_numpy_array(sample_indices=train_indices,
                                                view_indices=view_indices)
        self.view_shapes = [view_limits[ind+1]-view_limits[ind]
                            for ind in range(len(self.used_views)) ]

        return MumboClassifier.fit(self, numpy_X, y[train_indices],
                                                view_limits)

    def predict(self, X, sample_indices=None, view_indices=None):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        numpy_X, view_limits = X.to_numpy_array(sample_indices=sample_indices,
                                                view_indices=view_indices)
        return MumboClassifier.predict(self, numpy_X)

    def get_interpretation(self, directory, base_file_name, labels, multiclass=False):
        self.view_importances = np.zeros(len(self.used_views))
        self.feature_importances_ = [np.zeros(view_shape)
                                    for view_shape in self.view_shapes]
        for best_view, estimator_weight, estimator in zip(self.best_views_, self.estimator_weights_, self.estimators_):
            self.view_importances[best_view] += estimator_weight
            if hasattr(estimator, "feature_importances_"):
                self.feature_importances_[best_view] += estimator.feature_importances_
        importances_sum = sum([np.sum(feature_importances)
                               for feature_importances
                               in self.feature_importances_])
        self.feature_importances_ = [feature_importances/importances_sum
                                     for feature_importances
                                     in self.feature_importances_]
        for feature_importances, view_name in zip(self.feature_importances_, self.view_names):
            secure_file_path(os.path.join(directory, "feature_importances",
                                    base_file_name+view_name+"-feature_importances.csv"))
            np.savetxt(os.path.join(directory, "feature_importances",
                                    base_file_name+view_name+"-feature_importances.csv"),
                       feature_importances, delimiter=',')
        self.view_importances /= np.sum(self.view_importances)
        np.savetxt(os.path.join(directory, base_file_name+"view_importances.csv"), self.view_importances,
                   delimiter=',')

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

    def accepts_multi_class(self, random_state, n_samples=10, dim=2,
                            n_classes=3, n_views=2):
        return True
