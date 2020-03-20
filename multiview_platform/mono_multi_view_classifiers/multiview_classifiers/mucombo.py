from sklearn.tree import DecisionTreeClassifier


from multimodal.boosting.cumbo import MuCumboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_examples_views_indices
from ..utils.base import base_boosting_estimators

classifier_class_name = "MuCumbo"


class MuCumbo(BaseMultiviewClassifier, MuCumboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,**kwargs):
        BaseMultiviewClassifier.__init__(self, random_state)
        base_estimator = self.set_base_estim_from_dict(base_estimator, **kwargs)
        MuCumboClassifier.__init__(self, base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,)
        self.param_names = ["base_estimator", "n_estimators", "random_state",]
        self.distribs = [base_boosting_estimators,
                         CustomRandint(5,200), [random_state],]

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        self.used_views = view_indices
        numpy_X, view_limits = X.to_numpy_array(example_indices=train_indices,
                                                view_indices=view_indices)
        return MuCumboClassifier.fit(self, numpy_X, y[train_indices],
                                                view_limits)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                 example_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=example_indices,
                                                view_indices=view_indices)
        return MuCumboClassifier.predict(self, numpy_X)
