from sklearn.tree import DecisionTreeClassifier


from multimodal.boosting.cumbo import MuCumboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_examples_views_indices

classifier_class_name = "MuCumbo"


class MuCumbo(BaseMultiviewClassifier, MuCumboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,):
        super().__init__(random_state)
        super(BaseMultiviewClassifier, self).__init__(base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,)
        self.param_names = ["base_estimator", "n_estimators", "random_state",]
        self.distribs = [[DecisionTreeClassifier(max_depth=1)],
                         CustomRandint(5,200), [random_state],]

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=train_indices,
                                                view_indices=view_indices)
        return super(BaseMultiviewClassifier, self).fit(numpy_X, y[train_indices],
                                                view_limits)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                 example_indices,
                                                                 view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=example_indices,
                                                view_indices=view_indices)
        return super(BaseMultiviewClassifier, self).predict(numpy_X)
