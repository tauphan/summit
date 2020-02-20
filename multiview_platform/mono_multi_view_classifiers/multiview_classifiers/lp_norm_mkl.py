
from multimodal.kernels.lpMKL import MKL

from ..multiview.multiview_utils import BaseMultiviewClassifier, FakeEstimator
from .additions.kernel_learning import KernelClassifier, KernelConfigGenerator, KernelGenerator
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint


classifier_class_name = "LPNormMKL"

class LPNormMKL(KernelClassifier, MKL):
    def __init__(self, random_state=None, lmbda=0.1, m_param=1, n_loops=50,
                 precision=0.0001, use_approx=True, kernel="rbf",
                 kernel_params=None):
        super().__init__(random_state)
        super(BaseMultiviewClassifier, self).__init__(lmbda, m_param=m_param,
                                                      kernel=kernel,
                                                      n_loops=n_loops,
                                                      precision=precision,
                                                      use_approx=use_approx,
                                                      kernel_params=kernel_params)
        self.param_names = ["lmbda", "kernel", "kernel_params"]
        self.distribs = [CustomUniform(), ['rbf', 'additive_chi2', 'poly' ],
                         KernelConfigGenerator()]

    def fit(self, X, y, train_indices=None, view_indices=None):
        formatted_X, train_indices = self.format_X(X, train_indices, view_indices)
        try:
            self.init_kernels(nb_view=len(formatted_X))
        except:
            return FakeEstimator()

        return super(BaseMultiviewClassifier, self).fit(formatted_X, y[train_indices])

    def predict(self, X, example_indices=None, view_indices=None):
        new_X, _ = self.format_X(X, example_indices, view_indices)
        print(super(BaseMultiviewClassifier, self).predict(new_X))
        return self.extract_labels(super(BaseMultiviewClassifier, self).predict(new_X))



