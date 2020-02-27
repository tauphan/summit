from sklearn.tree import DecisionTreeClassifier
import numpy as np

from multimodal.boosting.mumbo import MumboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_examples_views_indices

classifier_class_name = "Mumbo"

class Mumbo(BaseMultiviewClassifier, MumboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 best_view_mode="edge"):
        BaseMultiviewClassifier.__init__(self, random_state)
        MumboClassifier.__init__(self, base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,
                                    best_view_mode=best_view_mode)
        self.param_names = ["base_estimator", "n_estimators", "random_state", "best_view_mode"]
        self.distribs = [[DecisionTreeClassifier(max_depth=1),
                          DecisionTreeClassifier(max_depth=2),
                          DecisionTreeClassifier(max_depth=3),
                          DecisionTreeClassifier(max_depth=4)],
                         CustomRandint(5,200), [random_state], ["edge", "error"]]

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
        if not np.array_equiv(np.sort(view_indices,axis=0), np.sort(self.used_views,axis=0)):
            raise ValueError("Fitted with {} views, asking a prediction on {}".format(self.used_views, view_indices))
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
            interpret_string+="- View {}({}), importance {}\n".format(view_index,
                                                                      self.view_names[view_index],
                                                                      self.view_importances[view_index])
        interpret_string +="\n The boosting process selected views : \n" + ", ".join(map(str, self.best_views_))
        interpret_string+="\n\n With estimator weights : \n"+ "\n".join(map(str,self.estimator_weights_/np.sum(self.estimator_weights_)))
        return interpret_string
