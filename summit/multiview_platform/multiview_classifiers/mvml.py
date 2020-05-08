
from multimodal.kernels.mvml import MVML

from .additions.kernel_learning import KernelClassifier, KernelConfigGenerator
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint


classifier_class_name = "MVMLClassifier"


class MVMLClassifier(KernelClassifier, MVML):

    def __init__(self, random_state=None, lmbda=0.1, eta=0.1, nystrom_param=1,
                 n_loops=50,
                 precision=0.0001, learn_A=0, kernel="rbf", learn_w=0,
                 kernel_params=None):
        KernelClassifier.__init__(self, random_state)
        MVML.__init__(self, lmbda=lmbda, eta=eta,
                                                      nystrom_param=nystrom_param,
                                                      kernel=kernel,
                                                      n_loops=n_loops,
                                                      precision=precision,
                                                      learn_A=learn_A,
                                                      learn_w=learn_w,
                                                      kernel_params=kernel_params)
        self.param_names = ["lmbda", "eta",  "nystrom_param", "learn_A",
                            "learn_w", "n_loops", "kernel_params", "kernel",
                            "precision"]
        self.distribs = [CustomUniform(),
                         CustomUniform(),
                         CustomUniform(),
                         [1,3,4],
                         [0,1],
                         CustomRandint(low=5, high=25),
                         KernelConfigGenerator(),
                         ['rbf', 'additive_chi2', 'poly' ],
                         CustomRandint(low=3, high=6, multiplier="e-")]

    def fit(self, X, y, train_indices=None, view_indices=None):
        formatted_X, train_indices = self.format_X(X, train_indices, view_indices)
        self.init_kernels(nb_view=len(formatted_X))
        return MVML.fit(self, formatted_X, y[train_indices])

    def predict(self, X, sample_indices=None, view_indices=None):
        new_X, _ = self.format_X(X, sample_indices, view_indices)
        return self.extract_labels(MVML.predict(self, new_X))


#
