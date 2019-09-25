import logging
import math
import time

import numpy as np
import numpy.ma as ma
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .BoostUtils import StumpsClassifiersGenerator, sign, BaseBoost, \
    getInterpretBase, get_accuracy_graph, TreeClassifiersGenerator
from ..monoview_utils import change_label_to_minus
from ... import metrics


# Used for QarBoost and CGreed

class ColumnGenerationClassifierQar(BaseEstimator, ClassifierMixin, BaseBoost):
    def __init__(self, n_max_iterations=None, estimators_generator="Stumps", max_depth=1,
                 random_state=42, self_complemented=True, twice_the_same=False,
                 c_bound_choice=True, random_start=True,
                 n_stumps=1, use_r=True, c_bound_sol=True,
                 plotted_metric=metrics.zero_one_loss, save_train_data=True,
                 test_graph=True, mincq_tracking=False):
        super(ColumnGenerationClassifierQar, self).__init__()
        r"""

            Parameters
            ----------
            n_max_iterations : int
                Maximum number of iterations for the boosting algorithm.
            estimators_generator : object
                Sk-learn classifier object used to generate the hypotheses with the data.
            random_state : np.random.RandomState or int
                The random state, used in order to be reproductible
            self_complemented : bool
                If True, in the hypotheses generation process, for each hypothesis, it's complement will be generated too.
            twice_the_same : bool
                If True, the algorithm will be allowed to select twice the same hypothesis in the boosting process.
            c_bound_choice : bool
                If True, the C-Bound will be used to select the hypotheses. If False, the margin will be the criterion.
            n_stumps_per_attribute : int
                The number of hypotheses generated by data attribute 
            use_r : bool
                If True, uses edge to compute the performance of a voter. If False, use the error instead.
            plotted_metric : Metric module
                The metric that will be plotted for each iteration of boosting. 
            """
        if type(random_state) is int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        self.train_time = 0
        self.train_shape = None
        self.step_decisions = None
        self.step_prod = None
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        self.self_complemented = self_complemented
        self.twice_the_same = twice_the_same
        self.c_bound_choice = c_bound_choice
        self.random_start = random_start
        self.plotted_metric = plotted_metric
        self.n_stumps = n_stumps
        self.use_r = use_r
        self.c_bound_sol = c_bound_sol
        self.save_train_data = save_train_data
        self.test_graph = test_graph
        self.printed_args_name_list = ["n_max_iterations", "self_complemented",
                                       "twice_the_same",
                                       "c_bound_choice", "random_start",
                                       "n_stumps", "use_r", "c_bound_sol"]
        self.mincq_tracking = mincq_tracking
        self.max_depth = max_depth

    def fit(self, X, y):

        formatted_X, formatted_y = self.format_X_y(X, y)

        self.init_info_containers()

        m, n, y_kernel_matrix = self.init_hypotheses(formatted_X, formatted_y)

        start = time.time()
        self.n_total_hypotheses_ = n
        self.n_total_examples = m

        self.init_boosting(m, formatted_y, y_kernel_matrix)

        self.break_cause = " the maximum number of iterations was attained."

        for k in range(min(n - 1,
                           self.n_max_iterations - 1 if self.n_max_iterations is not None else np.inf)):

            # Print dynamically the step and the error of the current classifier
            self.it = k

            # print(
            #     "Resp. bound : {}, {}; {}/{}, eps :{}, ".format(
            #         self.respected_bound,
            #         self.bounds[-1] > self.train_metrics[-1],
            #         k + 2,
            #         self.n_max_iterations,
            #         self.voter_perfs[-1],
            #     ),
            #     end="\r")
            sol, new_voter_index = self.choose_new_voter(y_kernel_matrix,
                                                             formatted_y)
            if type(sol) == str:
                self.break_cause = sol  #
                break
            self.append_new_voter(new_voter_index)
            voter_perf = self.compute_voter_perf(formatted_y)
            self.compute_voter_weight(voter_perf, sol)
            self.update_example_weights(formatted_y)
            self.update_info_containers(formatted_y, voter_perf, k)

        self.nb_opposed_voters = self.check_opposed_voters()
        self.estimators_generator.choose(self.chosen_columns_)

        if self.save_train_data:
            self.X_train = self.classification_matrix[:, self.chosen_columns_]
            self.raw_weights = self.weights_
            self.y_train = formatted_y

        # print(self.classification_matrix)
        # print(self.weights_, self.break_cause)
        self.weights_ = np.array(self.weights_)
        if np.sum(self.weights_) != 1:
            self.weights_ /= np.sum(self.weights_)

        formatted_y[formatted_y == -1] = 0
        formatted_y = formatted_y.reshape((m,))

        end = time.time()
        self.train_time = end - start
        return self

    def predict(self, X):
        start = time.time()
        check_is_fitted(self, 'weights_')
        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())
        classification_matrix = self._binary_classification_matrix(X)
        margins = np.sum(classification_matrix * self.weights_, axis=1)
        signs_array = np.array([int(x) for x in sign(margins)])
        signs_array[signs_array == -1] = 0
        end = time.time()
        self.predict_time = end - start
        self.step_predict(classification_matrix)
        return signs_array

    def step_predict(self, classification_matrix):
        """Used to predict with each step of the greedy algorithm to analyze its performance increase"""
        if classification_matrix.shape != self.train_shape:
            self.step_decisions = np.zeros(classification_matrix.shape)
            self.mincq_step_decisions = np.zeros(classification_matrix.shape)
            self.step_prod = np.zeros(classification_matrix.shape)
            for weight_index in range(self.weights_.shape[0] - 1):
                margins = np.sum(
                    classification_matrix[:, :weight_index + 1] * self.weights_[
                                                                  :weight_index + 1],
                    axis=1)
                signs_array = np.array([int(x) for x in sign(margins)])
                signs_array[signs_array == -1] = 0
                self.step_decisions[:, weight_index] = signs_array
                self.step_prod[:, weight_index] = np.sum(
                    classification_matrix[:, :weight_index + 1] * self.weights_[
                                                                  :weight_index + 1],
                    axis=1)
                if self.mincq_tracking:
                    if weight_index == 0:
                        self.mincq_step_decisions[:, weight_index] = signs_array
                    else:
                        mincq_margins = np.sum(self.mincq_learners[
                                                   weight_index - 1].majority_vote._weights * classification_matrix[
                                                                                              :,
                                                                                              :weight_index + 1],
                                               axis=1)
                        mincq_signs_array = np.array(
                            [int(x) for x in sign(mincq_margins)])
                        mincq_signs_array[mincq_signs_array == -1] = 0
                        self.mincq_step_decisions[:,
                        weight_index] = mincq_signs_array
                # self.mincq_step_cbounds = self.mincq_learners[weight_index-1].majority_vote.cbound_value()

    def update_info_containers(self, y, voter_perf, k):
        """Is used at each iteration to compute and store all the needed quantities for later analysis"""
        self.example_weights_.append(self.example_weights)
        self.tau.append(
            np.sum(np.multiply(self.previous_vote, self.new_voter)) / float(
                self.n_total_examples))
        # print(np.sum(np.multiply(self.previous_vote, self.new_voter))/float(self.n_total_examples))
        self.previous_vote += self.q * self.new_voter
        self.norm.append(np.linalg.norm(self.previous_vote) ** 2)
        self.previous_votes.append(self.previous_vote)
        self.previous_margins.append(
            np.sum(np.multiply(y, self.previous_vote)) / float(
                self.n_total_examples))
        self.selected_margins.append(
            np.sum(np.multiply(y, self.new_voter)) / float(
                self.n_total_examples))
        train_metric = self.plotted_metric.score(y, np.sign(self.previous_vote))
        if self.use_r:
            bound = self.bounds[-1] * math.sqrt(1 - voter_perf ** 2)
        else:
            bound = np.prod(
                np.sqrt(1 - 4 * np.square(0.5 - np.array(self.voter_perfs))))

        if train_metric > bound:
            self.respected_bound = False

        self.train_metrics.append(train_metric)
        self.bounds.append(bound)
        if self.mincq_tracking:  # Used to compute the optimal c-bound distribution on the chose set
            from ...monoview_classifiers.min_cq import MinCqLearner
            mincq = MinCqLearner(10e-3, "stumps", n_stumps_per_attribute=1,
                                 self_complemented=False)
            training_set = self.classification_matrix[:, self.chosen_columns_]
            mincq.fit(training_set, y)
            mincq_pred = mincq.predict(training_set)
            self.mincq_learners.append(mincq)
            self.mincq_train_metrics.append(
                self.plotted_metric.score(y, change_label_to_minus(mincq_pred)))
            self.mincq_weights.append(mincq.majority_vote._weights)
            self.mincq_c_bounds.append(
                mincq.majority_vote.cbound_value(training_set,
                                                 y.reshape((y.shape[0],))))

    def compute_voter_weight(self, voter_perf, sol):
        """used to compute the voter's weight according to the specified method (edge or error) """
        if self.c_bound_sol:
            self.q = sol
        else:
            if self.use_r:
                self.q = 0.5 * math.log((1 + voter_perf) / (1 - voter_perf))
            else:
                self.q = math.log((1 - voter_perf) / voter_perf)
        self.weights_.append(self.q)
        # self.weights_ = [weight/(1.0*sum(self.weights_)) for weight in self.weights_]

    def compute_voter_perf(self, formatted_y):
        """Used to computer the performance (error or edge) of the selected voter"""
        if self.use_r:
            r = self._compute_r(formatted_y)
            self.voter_perfs.append(r)
            return r
        else:
            epsilon = self._compute_epsilon(formatted_y)
            self.voter_perfs.append(epsilon)
            return epsilon

    def append_new_voter(self, new_voter_index):
        """Used to append the voter to the majority vote"""
        self.chosen_columns_.append(new_voter_index)
        # print((self.classification_matrix[:, new_voter_index] == self.chosen_one).all())
        self.new_voter = self.classification_matrix[:, new_voter_index].reshape(
            (self.n_total_examples, 1))

    def choose_new_voter(self, y_kernel_matrix, formatted_y):
        """Used to choose the voter according to the specified criterion (margin or C-Bound"""
        if self.c_bound_choice:
            sol, new_voter_index = self._find_new_voter(y_kernel_matrix,
                                                        formatted_y)
        else:
            new_voter_index, sol = self._find_best_weighted_margin(
                y_kernel_matrix)
        return sol, new_voter_index

    def init_boosting(self, m, y, y_kernel_matrix):
        """THis initialization corressponds to the first round of boosting with equal weights for each examples and the voter chosen by it's margin."""
        self.example_weights = self._initialize_alphas(m).reshape((m, 1))

        self.example_weights_.append(self.example_weights)

        if self.random_start:
            first_voter_index = self.random_state.choice(
                np.where(np.sum(y_kernel_matrix, axis=0) > 0)[0])
        else:
            first_voter_index, _ = self._find_best_weighted_margin(
                y_kernel_matrix)

        self.chosen_columns_.append(first_voter_index)
        self.new_voter = np.array(self.classification_matrix[:,
                                  first_voter_index].reshape((m, 1)), copy=True)

        self.previous_vote = self.new_voter
        self.norm.append(np.linalg.norm(self.previous_vote) ** 2)

        if self.use_r:
            r = self._compute_r(y)
            self.voter_perfs.append(r)
        else:
            epsilon = self._compute_epsilon(y)
            self.voter_perfs.append(epsilon)
        if self.c_bound_sol:
            self.q = 1
        else:
            if self.use_r:
                self.q = 0.5 * math.log((1 + r) / (1 - r))
            else:
                self.q = math.log((1 - epsilon) / epsilon)
        self.weights_.append(self.q)

        # Update the distribution on the examples.
        self.update_example_weights(y)
        self.example_weights_.append(self.example_weights)

        self.previous_margins.append(
            np.sum(np.multiply(y, self.previous_vote)) / float(
                self.n_total_examples))
        self.selected_margins.append(np.sum(np.multiply(y, self.previous_vote)))
        self.tau.append(
            np.sum(np.multiply(self.previous_vote, self.new_voter)) / float(
                self.n_total_examples))

        train_metric = self.plotted_metric.score(y, np.sign(self.previous_vote))
        if self.use_r:
            bound = math.sqrt(1 - r ** 2)
        else:
            bound = np.prod(np.sqrt(1 - 4 * np.square(0.5 - np.array(epsilon))))

        if train_metric > bound:
            self.respected_bound = False

        self.train_metrics.append(train_metric)

        self.bounds.append(bound)
        if self.mincq_tracking:
            self.mincq_train_metrics.append(train_metric)

    def format_X_y(self, X, y):
        """Formats the data  : X -the examples- and y -the labels- to be used properly by the algorithm """
        if scipy.sparse.issparse(X):
            logging.info('Converting to dense matrix.')
            X = np.array(X.todense())
        # Initialization
        y_neg = change_label_to_minus(y)
        y_neg = y_neg.reshape((y.shape[0], 1))
        return X, y_neg

    def init_hypotheses(self, X, y):
        """Inintialization for the hyptotheses used to build the boosted vote"""
        if self.estimators_generator == "Stumps":
            self.estimators_generator = StumpsClassifiersGenerator(
                n_stumps_per_attribute=self.n_stumps,
                self_complemented=self.self_complemented)
        if self.estimators_generator == "Trees":
            self.estimators_generator = TreeClassifiersGenerator(
                n_trees=self.n_stumps, max_depth=self.max_depth,
                self_complemented=self.self_complemented)
        self.estimators_generator.fit(X, y)
        self.classification_matrix = self._binary_classification_matrix(X)
        self.train_shape = self.classification_matrix.shape

        m, n = self.classification_matrix.shape
        y_kernel_matrix = np.multiply(y, self.classification_matrix)

        return m, n, y_kernel_matrix

    def init_info_containers(self):
        """Initialize the containers that will be collected at each iteration for the analysis"""
        self.weights_ = []
        self.chosen_columns_ = []
        self.fobidden_columns = []
        self.c_bounds = []
        self.voter_perfs = []
        self.example_weights_ = []
        self.train_metrics = []
        self.bounds = []
        self.disagreements = []
        self.margins = []
        self.previous_votes = []
        self.previous_margins = []
        self.respected_bound = True
        self.selected_margins = []
        self.tau = []
        self.norm = []
        self.mincq_train_metrics = []
        self.mincq_c_bounds = []
        self.mincq_weights = []
        self.mincq_learners = []
        self.mincq_step_decisions = []

    def _compute_epsilon(self, y):
        """Updating the error variable, the old fashioned way uses the whole majority vote to update the error"""
        ones_matrix = np.zeros(y.shape)
        ones_matrix[np.multiply(y, self.new_voter.reshape(
            y.shape)) < 0] = 1  # can np.divide if needed
        epsilon = np.average(np.multiply(y, self.new_voter.reshape(
            y.shape)), axis=0)
        return epsilon

    def _compute_r(self, y):
        ones_matrix = np.ones(y.shape)

        ones_matrix[np.multiply(y, self.new_voter.reshape(
            y.shape)) < 0] = -1  # can np.divide if needed
        r = np.average(ones_matrix, weights=self.example_weights, axis=0)
        return r

    def update_example_weights(self, y):
        """Old fashioned exaple weights update uses the whole majority vote, the other way uses only the last voter."""
        new_weights = self.example_weights.reshape(
            (self.n_total_examples, 1)) * np.exp(
            -self.q * np.multiply(y, self.new_voter))
        self.example_weights = new_weights / np.sum(new_weights)

    def _find_best_weighted_margin(self, y_kernel_matrix, upper_bound=1.0,
                                   lower_bound=0.0):
        """Finds the new voter by choosing the one that has the best weighted margin between 0.5 and 0.55
        to avoid too god voters that will get all the votes weights"""
        weighted_kernel_matrix = np.multiply(y_kernel_matrix,
                                             self.example_weights.reshape(
                                                 (self.n_total_examples, 1)))
        pseudo_h_values = ma.array(np.sum(weighted_kernel_matrix, axis=0),
                                   fill_value=-np.inf)
        pseudo_h_values[self.chosen_columns_] = ma.masked
        return np.argmax(pseudo_h_values), [0]

    def _find_new_voter(self, y_kernel_matrix, y):
        """Here, we solve the two_voters_mincq_problem for each potential new voter,
        and select the one that has the smallest minimum"""
        m = y_kernel_matrix.shape[0]
        weighted_previous_sum = np.multiply(y,
                                            self.previous_vote.reshape(m, 1))
        margin_old = np.sum(weighted_previous_sum)
        if self.c_bound_sol:
            weighted_hypothesis = y_kernel_matrix
        else:
            weighted_hypothesis = y_kernel_matrix * self.example_weights.reshape(
                (m, 1))

        bad_margins = np.where(np.sum(weighted_hypothesis, axis=0) <= 0.0)[0]

        self.B2 = m
        self.B1s = np.sum(
            2 * np.multiply(weighted_previous_sum, weighted_hypothesis),
            axis=0)
        self.B0 = np.sum(weighted_previous_sum ** 2)

        self.A2s = np.sum(weighted_hypothesis, axis=0) ** 2
        self.A1s = np.sum(weighted_hypothesis, axis=0) * margin_old * 2
        self.A0 = margin_old ** 2
        import matplotlib.pyplot as plt
        # plt.plot(self.A2s * 0.5 * self.B1s / m**3)
        # plt.plot(np.array([margin_old/m for _ in range(len(self.A2s))]))
        # plt.savefig("try.png")

        # print("C2 < 0 :", np.where(np.array([margin_old/m for _ in range(len(self.A2s))]) < np.sqrt(self.A2s) * 0.5 * self.B1s / m**2)[0])
        # print("C1 < 0 :", np.where(np.array([margin_old ** 2 / m for _ in range(
        #     len(self.A2s))]) < self.A2s * self.B0 / m ** 2)[0])
        # print("Double root:", np.where((0.5 * self.B1s / m)**2 * m > self.B0)[0])


        C2s = (self.A1s * self.B2 - self.A2s * self.B1s)
        # print("Wrong C2 :" , np.where(C2s < 0)[0].shape, bad_margins.shape)
        C1s = 2 * (self.A0 * self.B2 - self.A2s * self.B0)
        # print("Wrong C2 :", np.where(C1s < 0)[0].shape, bad_margins.shape)
        C0s = self.A0 * self.B1s - self.A1s * self.B0

        # print(np.where(C2s==0))
        # print(self.chosen_columns_)

        sols = np.zeros(C0s.shape) - 3
        # sols[np.where(C2s == 0)[0]] = C0s[np.where(C2s == 0)[0]] / C1s[np.where(C2s == 0)[0]]
        sols[np.where(C2s != 0)[0]] = (-C1s[np.where(C2s != 0)[0]] + np.sqrt(
            C1s[np.where(C2s != 0)[0]] * C1s[np.where(C2s != 0)[0]] - 4 * C2s[
                np.where(C2s != 0)[0]] * C0s[np.where(C2s != 0)[0]])) / (
                                                  2 * C2s[
                                              np.where(C2s != 0)[0]])

        masked_c_bounds = self.make_masked_c_bounds(sols, bad_margins)
        if masked_c_bounds.mask.all():
            return "No more pertinent voters", 0
        else:
            best_hyp_index = np.argmin(masked_c_bounds)
            self.c_bounds.append(masked_c_bounds[best_hyp_index])
            self.margins.append(math.sqrt(self.A2s[best_hyp_index] / m))
            self.disagreements.append(0.5 * self.B1s[best_hyp_index] / m)
            return sols[best_hyp_index], best_hyp_index

    def make_masked_c_bounds(self, sols, bad_margins):
        c_bounds = self.compute_c_bounds(sols)
        trans_c_bounds = self.compute_c_bounds(sols + 1)
        masked_c_bounds = ma.array(c_bounds, fill_value=np.inf)
        # Masing Maximums
        masked_c_bounds[c_bounds >= trans_c_bounds] = ma.masked
        # Masking magrins <= 0
        masked_c_bounds[bad_margins] = ma.masked
        # Masking weights < 0 (because self-complemented)
        masked_c_bounds[sols < 0] = ma.masked
        # Masking nan c_bounds
        masked_c_bounds[np.isnan(c_bounds)] = ma.masked
        if not self.twice_the_same:
            masked_c_bounds[self.chosen_columns_] = ma.masked
        return masked_c_bounds

    def compute_c_bounds(self, sols):
        return 1 - (self.A2s * sols ** 2 + self.A1s * sols + self.A0) / ((
                                                                                 self.B2 * sols ** 2 + self.B1s * sols + self.B0) * self.n_total_examples)

    def _cbound(self, sol):
        """Computing the objective function"""
        return 1 - (self.A2 * sol ** 2 + self.A1 * sol + self.A0) / ((
                                                                             self.B2 * sol ** 2 + self.B1 * sol + self.B0) * self.n_total_examples)

    def disagreement(self, sol):
        return (
                           self.B2 * sol ** 2 + self.B1 * sol + self.B0) / self.n_total_examples

    def margin(self, sol):
        return (
                           self.A2 * sol ** 2 + self.A1 * sol + self.A0) / self.n_total_examples

    def _best_sol(self, sols):
        """Return the best min in the two possible sols"""
        values = np.array([self._cbound(sol) for sol in sols])
        return sols[np.argmin(values)]

    def _initialize_alphas(self, n_examples):
        """Initialize the examples wieghts"""
        return 1.0 / n_examples * np.ones((n_examples,))

    def get_step_decision_test_graph(self, directory, y_test):
        np.savetxt(directory + "y_test_step.csv", self.step_decisions,
                   delimiter=',')
        step_metrics = []
        for step_index in range(self.step_decisions.shape[1] - 1):
            step_metrics.append(self.plotted_metric.score(y_test,
                                                          self.step_decisions[:,
                                                          step_index]))
        step_metrics = np.array(step_metrics)
        np.savetxt(directory + "step_test_metrics.csv", step_metrics,
                   delimiter=',')
        get_accuracy_graph(step_metrics, self.__class__.__name__,
                           directory + 'step_test_metrics.png',
                           self.plotted_metric, set="test")

        if self.mincq_tracking:
            step_mincq_test_metrics = []
            for step_index in range(self.step_decisions.shape[1] - 1):
                step_mincq_test_metrics.append(self.plotted_metric.score(y_test,
                                                                         self.mincq_step_decisions[
                                                                         :,
                                                                         step_index]))
            np.savetxt(directory + "mincq_step_test_metrics.csv",
                       step_mincq_test_metrics,
                       delimiter=',')
            get_accuracy_graph(step_metrics, self.__class__.__name__,
                               directory + 'step_test_metrics_comparaison.png',
                               self.plotted_metric, step_mincq_test_metrics,
                               "MinCQ metric", set="test")

        step_cbounds = []
        for step_index in range(self.step_prod.shape[1]):
            num = np.sum(y_test * self.step_prod[:, step_index]) ** 2
            den = np.sum((self.step_prod[:, step_index]) ** 2)
            step_cbounds.append(1 - num / (den * self.step_prod.shape[0]))
        step_cbounds = np.array(step_cbounds)
        np.savetxt(directory + "step_test_c_bounds.csv", step_cbounds,
                   delimiter=',')
        get_accuracy_graph(step_cbounds, self.__class__.__name__,
                           directory + 'step_test_c_bounds.png',
                           "C_bound", set="test")

    def getInterpretQar(self, directory, y_test=None):
        self.directory = directory
        """Used to interpret the functionning of the algorithm"""
        if self.step_decisions is not None:
            self.get_step_decision_test_graph(directory, y_test)
        # get_accuracy_graph(self.voter_perfs[:20], self.__class__.__name__,
        #                    directory + 'voter_perfs.png', "Rs")
        get_accuracy_graph(self.weights_, self.__class__.__name__,
                           directory + 'vote_weights.png', "weights",
                           zero_to_one=False)
        get_accuracy_graph(self.c_bounds, self.__class__.__name__,
                           directory + 'c_bounds.png', "C-Bounds")
        if self.mincq_tracking:
            get_accuracy_graph(self.c_bounds, self.__class__.__name__,
                               directory + 'c_bounds_comparaison.png',
                               "1-var mins", self.mincq_c_bounds, "MinCQ min",
                               zero_to_one=False)
            get_accuracy_graph(self.train_metrics, self.__class__.__name__,
                               directory + 'train_metrics_comparaison.png',
                               self.plotted_metric,
                               self.mincq_train_metrics, "MinCQ metrics")
        get_accuracy_graph(self.previous_margins, self.__class__.__name__,
                           directory + 'margins.png', "Margins",
                           zero_to_one=False)
        get_accuracy_graph(self.selected_margins, self.__class__.__name__,
                           directory + 'selected_margins.png',
                           "Selected Margins")
        self.tau[0] = 0
        get_accuracy_graph(self.tau, self.__class__.__name__,
                           directory + 'disagreements.png', "disagreements",
                           zero_to_one=False)
        get_accuracy_graph(self.train_metrics[:-1], self.__class__.__name__,
                           directory + 'c_bounds_train_metrics.png',
                           self.plotted_metric, self.c_bounds, "C-Bound",
                           self.bounds[:-1])
        get_accuracy_graph(self.norm, self.__class__.__name__,
                           directory + 'norms.png',
                           "squared 2-norm", zero_to_one=False)
        interpretString = getInterpretBase(self, directory,
                                           self.__class__.__name__,
                                           self.weights_, self.break_cause)
        if self.save_train_data:
            np.savetxt(directory + "x_train.csv", self.X_train, delimiter=',')
            np.savetxt(directory + "y_train.csv", self.y_train, delimiter=',')
            np.savetxt(directory + "raw_weights.csv", self.raw_weights,
                       delimiter=',')
            np.savetxt(directory + "c_bounds.csv", self.c_bounds, delimiter=',')
            np.savetxt(directory + "train_metrics.csv", self.train_metrics,
                       delimiter=',')
            np.savetxt(directory + "margins.csv", self.previous_margins,
                       delimiter=',')
            np.savetxt(directory + "disagreements.csv", self.tau,
                       delimiter=',')
            np.savetxt(directory + "disagreements.csv", self.norm,
                       delimiter=',')
            if self.mincq_tracking:
                np.savetxt(directory + "mincq_cbounds.csv", self.mincq_c_bounds,
                           delimiter=',')
                np.savetxt(directory + "mincq_train_metrics.csv",
                           self.mincq_train_metrics,
                           delimiter=',')
        args_dict = dict(
            (arg_name, str(self.__dict__[arg_name])) for arg_name in
            self.printed_args_name_list)
        interpretString += "\n \n With arguments : \n" + u'\u2022 ' + (
                "\n" + u'\u2022 ').join(['%s: \t%s' % (key, value)
                                         for (key, value) in
                                         args_dict.items()])
        if not self.respected_bound:
            interpretString += "\n\n The bound was not respected"

        return interpretString
