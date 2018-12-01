import scipy
import logging
import numpy as np
import numpy.ma as ma
import math
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import matplotlib.pyplot as plt

from .BoostUtils import StumpsClassifiersGenerator, sign, BaseBoost, \
    getInterpretBase, get_accuracy_graph
from ..MonoviewUtils import format_Xy
from ... import Metrics


class ColumnGenerationClassifierQar(BaseEstimator, ClassifierMixin, BaseBoost):
    def __init__(self, n_max_iterations=350, estimators_generator=None, random_state=42, self_complemented=True, twice_the_same=False,
                 c_bound_choice = True, random_start = True, n_stumps_per_attribute=None, use_r=True, plotted_metric=Metrics.zero_one_loss):
        r"""This is the QarBoost constructor.

            Parameters
            ----------
            n_max_iterations : int
                Maximum number of boosting iterations.
            estimators_generator : object
                THe generator for the hypotheses.
            benchmarkArgumentsDictionaries : list of dictionaries
                All the needed arguments for the benchmarks.
            random_state : int or np.random.RandomState
                The random state for the algorithm.
            self_complemented : bool
                If True, the generation of hypothesis will also generate the opposite of each hypothesis
            twice_the_same : bool
                If True, the boosting algorithm will be able to use twice (or more) the same hypothesis
            c_bound_choice : bool
                If True the C-bound is used to find the next voter, if false, the margin will be used
            random_start : bool
                If True, the first voter will be randomly selected.
            n_stumps_per_attribute : int
                The number of stumps generated per attribute in X.
            use_r : True
                If True, uses the margin instead of the error to update the weights.
            plotted_metric : module
                The module of the metric used in the plots.
            """
        super(ColumnGenerationClassifierQar, self).__init__()
        self.train_time = 0
        self.n_max_iterations = n_max_iterations
        self.estimators_generator = estimators_generator
        if type(random_state) is int:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state
        self.self_complemented = self_complemented
        self.twice_the_same = twice_the_same
        self.c_bound_choice = c_bound_choice
        self.random_start = random_start
        self.plotted_metric = plotted_metric
        if n_stumps_per_attribute:
            self.n_stumps_per_attribute = n_stumps_per_attribute
        self.use_r = use_r
        self.printed_args_name_list = ["n_max_iterations", "self_complemented", "twice_the_same",
                                       "c_bound_choice", "random_start",
                                       "n_stumps", "use_r"]

    def set_params(self, **params):
        # TODO
        self.self_complemented = params["self_complemented"]
        self.twice_the_same = params["twice_the_same"]
        self.c_bound_choice = params["c_bound_choice"]
        self.random_start = params["random_start"]

    def fit(self, X, y):
        start = time.time()
        formatted_y, formatted_X = format_Xy(X, y)

        m, n, y_kernel_matrix = self.init_hypotheses(formatted_X, formatted_y)

        self.init_boosting(y_kernel_matrix, formatted_y, m, n, )

        for k in range(min(n-1,
                           self.n_max_iterations-1 if
                           self.n_max_iterations is not None else np.inf)):

            # Print dynamically the step
            print("{}/{}".format(k, self.n_max_iterations,), end="\r")

            sol, new_voter_index = self.choose_new_voter(formatted_y,
                                                         y_kernel_matrix)

            # If the new voter selector could not find one, break the loop
            if type(sol) == str:
                self.break_cause = new_voter_index  #
                break

            # Append the weak hypothesis.
            self.chosen_columns_.append(new_voter_index)
            self.new_voter = self.classification_matrix[:,
                             new_voter_index].reshape((m, 1))

            # Generate the new weights
            weighted_score = self.compute_weighted_score(formatted_y)
            self.compute_voter_weight(weighted_score, k)
            self._update_example_weights(formatted_y)

            self.record_infos(formatted_y, weighted_score, k)

        self.estimators_generator.estimators_ = self.estimators_generator.estimators_[self.chosen_columns_]
        self.voters_weights = np.array(self.voters_weights)

        self.voters_weights /= np.sum(self.voters_weights)
        self.nb_opposed_voters = self.check_opposed_voters()
        end = time.time()
        self.train_time = end - start
        return self

    def predict(self, X):
        start = time.time()
        check_is_fitted(self, 'voters_weights')
        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())
        classification_matrix = self._binary_classification_matrix(X)
        margins = np.squeeze(np.asarray(np.matmul(classification_matrix, self.voters_weights)))
        signs_array = np.array([int(x) for x in sign(margins)])
        signs_array[signs_array == -1] = 0
        end = time.time()
        self.predict_time = end - start
        return signs_array

    def init_hypotheses(self, X, y):
        if self.estimators_generator is None:
            self.estimators_generator = StumpsClassifiersGenerator(n_stumps_per_attribute=self.n_stumps_per_attribute,
                                                                   self_complemented=self.self_complemented)
        self.get_classification_matrix(X, y)
        m, n = self.classification_matrix.shape
        y_kernel_matrix = np.multiply(y, self.classification_matrix)
        return m, n, y_kernel_matrix

    def init_boosting(self, y_kernel_matrix, y, m, n):
        self.init_info_containers()

        self.n_total_hypotheses_ = n
        self.n_total_examples = m
        self.n_max_iterations = n

        self.example_weights = self._initialize_alphas(m).reshape((m, 1))

        if self.random_start:
            first_voter_index = self.random_state.choice(
                self.get_possible(y_kernel_matrix, y))
        else:
            first_voter_index, _ = self._find_best_weighted_margin(
                y_kernel_matrix)
        if type(first_voter_index) is str:
            raise ValueError("Jambon")
        self.chosen_columns_ = [first_voter_index]
        self.new_voter = self.classification_matrix[:, first_voter_index].reshape((m, 1))
        self.bounds.append(math.sqrt(1 - self._compute_r(y) ** 2))
        self.previous_vote = self.new_voter
        self.voters_weights.append(1)

    def record_infos(self, y, r,k):
        self.previous_vote = np.matmul(
            self.classification_matrix[:, self.chosen_columns_],
            np.array(self.voters_weights).reshape((k + 2, 1))).reshape(
            (self.n_total_examples, 1))
        self.previous_votes.append(self.previous_vote)
        self.previous_margins.append(
            np.multiply(y, self.previous_vote))
        self.train_metrics.append(
            self.plotted_metric.score(y, np.sign(self.previous_vote)))
        
        # self.bounds.append(np.prod(np.sqrt(1-4*np.square(0.5-np.array(self.epsilons)))))

        self.bounds.append(self.bounds[-1] * math.sqrt(1 - r ** 2))

    def compute_voter_weight(self, weighted_score, k):
        if self.use_r:
            self.q = 0.5 * math.log((1 + weighted_score) / (1 - weighted_score))
        else:
            self.q = math.log((1 - weighted_score) / weighted_score)
        self.voters_weights.append(self.q)

    def compute_weighted_score(self,y):
        if not self.use_r:
            return self._compute_epsilon(y)
        else:
            return self._compute_r(y)

    def choose_new_voter(self, y, y_kernel_matrix, ):
        if self.c_bound_choice:
            sol, new_voter_index = self._find_new_voter(y_kernel_matrix,
                                                        y)
        else:
            new_voter_index, sol = self._find_best_weighted_margin(
                y_kernel_matrix)
        return sol, new_voter_index

    def init_info_containers(self, ):
        self.chosen_columns_ = []
        self.c_bounds = []
        self.epsilons = []
        self.example_weights_ = []
        self.train_metrics = []
        self.bounds = []
        self.voters_weights = []
        self.previous_votes = []
        self.previous_margins = []
        self.voters_weights = []
        self.break_cause = None
        self.break_cause = " the maximum number of iterations was attained."

    def get_classification_matrix(self, X, y):
        self.estimators_generator.fit(X, y)
        self.classification_matrix = self._binary_classification_matrix(X)

    def _compute_epsilon(self,y):
        """Updating the error variable, the old fashioned way uses the whole majority vote to update the error"""
        ones_matrix = np.zeros(y.shape)
        ones_matrix[np.multiply(y, self.new_voter.reshape(y.shape)) < 0] = 1  # can np.divide if needed
        epsilon = np.average(ones_matrix, weights=self.example_weights, axis=0)
        self.epsilons.append(epsilon)
        return epsilon

    def _compute_r(self, y):
        ones_matrix = np.ones(y.shape)
        ones_matrix[np.multiply(y, self.new_voter.reshape(y.shape)) < 0] = -1  # can np.divide if needed
        r = np.average(ones_matrix, weights=self.example_weights, axis=0)
        return r

    def _update_example_weights(self, y):
        """Old fashioned exaple weights update uses the whole majority vote, the other way uses only the last voter."""
        new_weights = self.example_weights.reshape((self.n_total_examples, 1))*np.exp(-self.q*np.multiply(y,self.new_voter))
        self.example_weights = new_weights/np.sum(new_weights)
        self.example_weights_.append(self.example_weights)



    def _find_best_margin(self, y_kernel_matrix):
        """Used only on the first iteration to select the voter with the largest margin"""
        pseudo_h_values = ma.array(np.sum(y_kernel_matrix, axis=0), fill_value=-np.inf)
        worst_h_index = ma.argmax(pseudo_h_values)
        return worst_h_index

    def _find_best_weighted_margin(self, y_kernel_matrix, upper_bound=0.56):
        """Finds the new voter by choosing the one that has the best weighted margin between 0.5 and 0.56
        to avoid too god voters that will get all the votes weights"""
        print(y_kernel_matrix)
        weighted_kernel_matrix = np.multiply(y_kernel_matrix, self.example_weights.reshape((self.n_total_examples, 1)))
        pseudo_h_values = ma.array(np.sum(weighted_kernel_matrix, axis=0), fill_value=-np.inf)
        pseudo_h_values[self.chosen_columns_] = ma.masked
        print(np.sort(pseudo_h_values))
        acceptable_indices = np.where(np.logical_and(np.greater(upper_bound, pseudo_h_values), np.greater(pseudo_h_values, 0.5)))[0]
        if acceptable_indices.size > 0:
            worst_h_index = self.random_state.choice(acceptable_indices)
            return worst_h_index, [0]
        else:
            return " no margin over random and acceptable", ""

    def _is_not_too_wrong(self, hypothese, y):
        """Check if the weighted margin is better than random"""
        ones_matrix = np.zeros(y.shape)
        ones_matrix[hypothese.reshape(y.shape) < 0] = 1
        epsilon = np.average(ones_matrix, weights=self.example_weights, axis=0)
        return epsilon < 0.5

    def get_possible(self, y_kernel_matrix, y):
        """Get all the indices of the hypothesis that are good enough to be chosen"""
        possibleIndices = []
        for hypIndex, hypothese in enumerate(np.transpose(y_kernel_matrix)):
            if self._is_not_too_wrong(hypothese, y):
                possibleIndices.append(hypIndex)
        return np.array(possibleIndices)

    def _find_new_voter(self, y_kernel_matrix, y):
        """Here, we solve the two_voters_mincq_problem for each potential new voter,
        and select the one that has the smallest minimum"""
        c_borns = []
        possible_sols = []
        indices = []
        causes = []
        for hypothese_index, hypothese in enumerate(y_kernel_matrix.transpose()):
            if (hypothese_index not in self.chosen_columns_ or self.twice_the_same)\
            and set(self.chosen_columns_)!={hypothese_index} \
            and self._is_not_too_wrong(hypothese, y):
                w = self._solve_one_weight_min_c(hypothese, y)
                if w[0] != "break":
                    c_borns.append(self._cbound(w[0]))
                    possible_sols.append(w)
                    indices.append(hypothese_index)
                else:
                    causes.append(w[1])
        if not causes:
            causes = ["no feature was better than random and acceptable"]
        if c_borns:
            min_c_born_index = ma.argmin(c_borns)
            self.c_bounds.append(c_borns[min_c_born_index])
            selected_sol = possible_sols[min_c_born_index]
            selected_voter_index = indices[min_c_born_index]
            return selected_sol, selected_voter_index
        else:
            return "break", " and ".join(set(causes))

    def _solve_one_weight_min_c(self, next_column, y):
        """Here we solve the min C-bound problem for two voters using one weight only and return the best weight
        No precalc because longer ; see the "derivee" latex document for more precision"""
        m = next_column.shape[0]
        zero_diag = np.ones((m, m)) - np.identity(m)
        weighted_previous_sum = np.multiply(y, self.previous_vote.reshape((m, 1)))
        weighted_next_column = np.multiply(next_column.reshape((m,1)), self.example_weights.reshape((m,1)))

        self.B2 = np.sum(weighted_next_column ** 2)
        self.B1 = np.sum(2 * weighted_next_column * weighted_previous_sum)
        self.B0 = np.sum(weighted_previous_sum ** 2)

        M2 = np.sum(np.multiply(np.matmul(weighted_next_column, np.transpose(weighted_next_column)), zero_diag))
        M1 = np.sum(np.multiply(np.matmul(weighted_previous_sum, np.transpose(weighted_next_column)) +
                                np.matmul(weighted_next_column, np.transpose(weighted_previous_sum))
                                , zero_diag))
        M0 = np.sum(np.multiply(np.matmul(weighted_previous_sum, np.transpose(weighted_previous_sum)), zero_diag))

        self.A2 = self.B2 + M2
        self.A1 = self.B1 + M1
        self.A0 = self.B0 + M0

        C2 = (M1 * self.B2 - M2 * self.B1)
        C1 = 2 * (M0 * self.B2 - M2 * self.B0)
        C0 = M0 * self.B1 - M1 * self.B0
        if C2 == 0:
            if C1 == 0:
                return ['break', "the derivate was constant"]
            else :
                is_acceptable, sol = self._analyze_solutions_one_weight(np.array(float(C0)/C1).reshape((1,1)))
                if is_acceptable:
                    return np.array([sol])
        try:
            sols = np.roots(np.array([C2, C1, C0]))
        except:
            return ["break", "nan"]

        is_acceptable, sol = self._analyze_solutions_one_weight(sols)
        if is_acceptable:
            return np.array([sol])
        else:
            return ["break", sol]

    def _analyze_solutions_one_weight(self, sols):
        """"We just check that the solution found by np.roots is acceptable under our constraints
        (real, a minimum and over 0)"""
        if sols.shape[0] == 1:
            if self._cbound(sols[0]) < self._cbound(sols[0] + 1):
                best_sol = sols[0]
            else:
                return False, "the only solution was a maximum."
        elif sols.shape[0] == 2:
            best_sol = self._best_sol(sols)
        else:
            return False, "no solution were found"

        if isinstance(best_sol, complex):
            return False, "the sol was complex"
        else:
            return True, best_sol

    def _cbound(self, sol):
        """Computing the objective function"""
        return 1 - (self.A2*sol**2 + self.A1*sol + self.A0)/(self.B2*sol**2 + self.B1*sol + self.B0)/self.n_total_examples

    def _best_sol(self, sols):
        """Return the best min in the two possible sols"""
        values = np.array([self._cbound(sol) for sol in sols])
        return sols[np.argmin(values)]

    def _initialize_alphas(self, n_examples):
        """Initialize the examples wieghts"""
        return 1.0 / n_examples * np.ones((n_examples,))

    def getInterpretQar(self, directory):
        """Used to interpret the functionning of the algorithm"""
        path = "/".join(directory.split("/")[:-1])
        try:
            import os
            os.makedirs(path+"/gif_images")
        except:
            raise
        filenames=[]
        max_weight = max([np.max(examples_weights) for examples_weights in self.example_weights_])
        min_weight = min([np.max(examples_weights) for examples_weights in self.example_weights_])
        for iterIndex, examples_weights in enumerate(self.example_weights_):
            r = np.array(examples_weights)
            theta = np.arange(self.n_total_examples)
            colors = np.sign(self.previous_margins[iterIndex])
            fig = plt.figure(figsize=(5, 5), dpi=80)
            ax = fig.add_subplot(111)
            c = ax.scatter(theta, r, c=colors, cmap='RdYlGn', alpha=0.75)
            ax.set_ylim(min_weight, max_weight)
            filename = path+"/gif_images/"+str(iterIndex)+".png"
            filenames.append(filename)
            plt.savefig(filename)
            plt.close()

        import imageio
        images = []
        logging.getLogger("PIL").setLevel(logging.WARNING)
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(path+'/weights.gif', images, duration=1. / 2)
        import shutil
        shutil.rmtree(path+"/gif_images")
        get_accuracy_graph(self.epsilons, self.__class__.__name__, directory + 'epsilons.png', "Errors")
        interpretString = getInterpretBase(self, directory, "QarBoost", self.voters_weights, self.break_cause)

        args_dict = dict((arg_name, str(self.__dict__[arg_name])) for arg_name in self.printed_args_name_list)
        interpretString += "\n \n With arguments : \n"+u'\u2022 '+ ("\n"+u'\u2022 ').join(['%s: \t%s' % (key, value)
                                                                                         for (key, value) in args_dict.items()])

        return interpretString






  # def _solve_two_weights_min_c(self, next_column, y):
  #       """Here we solve the min C-bound problem for two voters and return the best 2-weights array
  #       No precalc because longer"""
  #       m = next_column.shape[0]
  #       zero_diag = np.ones((m, m)) - np.identity(m)
  #       weighted_previous_sum = np.multiply(y, self.previous_vote.reshape((m, 1)))
  #       weighted_next_column = np.multiply(next_column.reshape((m,1)), self.example_weights.reshape((m,1)))
  #
  #       self.B2 = np.sum((weighted_previous_sum - weighted_next_column) ** 2)
  #       self.B1 = np.sum(2 * weighted_next_column * (weighted_previous_sum - weighted_next_column))
  #       self.B0 = np.sum(weighted_next_column * weighted_next_column)
  #
  #       M2 = np.sum(np.multiply(np.matmul((weighted_previous_sum - weighted_next_column), np.transpose(weighted_previous_sum - weighted_next_column)), zero_diag))
  #       M1 = np.sum(np.multiply(np.matmul(weighted_previous_sum, np.transpose(weighted_next_column)) + np.matmul(weighted_next_column, np.transpose(weighted_previous_sum)) - 2*np.matmul(weighted_next_column, np.transpose(weighted_next_column)), zero_diag))
  #       M0 = np.sum(np.multiply(np.matmul(weighted_next_column, np.transpose(weighted_next_column)), zero_diag))
  #
  #       self.A2 = self.B2 + M2
  #       self.A1 = self.B1 + M1
  #       self.A0 = self.B0 + M0
  #
  #       C2 = (M1 * self.B2 - M2 * self.B1)
  #       C1 = 2 * (M0 * self.B2 - M2 * self.B0)
  #       C0 = M0 * self.B1 - M1 * self.B0
  #
  #       if C2 == 0:
  #           if C1 == 0:
  #               return np.array([0.5, 0.5])
  #           elif abs(C1) > 0:
  #               return np.array([0., 1.])
  #           else:
  #               return ['break', "the derivate was constant"]
  #       elif C2 == 0:
  #           return ["break", "the derivate was affine"]
  #       try:
  #           sols = np.roots(np.array([C2, C1, C0]))
  #       except:
  #           return ["break", "nan"]
  #       is_acceptable, sol = self._analyze_solutions(sols)
  #       if is_acceptable:
  #           return np.array([sol, 1-sol])
  #       else:
  #           return ["break", sol]
  #
  #   def _analyze_solutions(self, sols):
  #       """"We just check that the solution found by np.roots is acceptable under our constraints
  #       (real, a minimum and between 0 and 1)"""
  #       for sol_index, sol in enumerate(sols):
  #           if isinstance(sol, complex):
  #               sols[sol_index] = -1
  #       if sols.shape[0] == 1:
  #           if self._cbound(sols[0]) < self._cbound(sols[0] + 1):
  #               best_sol = sols[0]
  #           else:
  #               return False, "the only solution was a maximum."
  #       elif sols.shape[0] == 2:
  #           best_sol = self._best_sol(sols)
  #       else:
  #           return False, "no solution were found"
  #
  #       if 0 < best_sol < 1:
  #           return True, self._best_sol(sols)
  #
  #       elif best_sol <= 0:
  #           return False, "the minimum was below 0"
  #       else:
  #           return False, "the minimum was over 1"
