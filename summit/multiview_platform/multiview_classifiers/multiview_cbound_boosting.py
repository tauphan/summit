from sklearn.tree import DecisionTreeClassifier
import numpy as np
import numpy.ma as ma
import math
import os
import pandas as pd

from multimodal.boosting.mumbo import MumboClassifier

from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..monoview_classifiers.additions import CBBoostUtils, BoostUtils
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_samples_views_indices
from .. import metrics
from ..utils.base import base_boosting_estimators
from ..utils.organization import secure_file_path
from .. import monoview_classifiers

classifier_class_name = "MultiviewCBoundBoosting"

class MultiviewCBoundBoosting(BaseMultiviewClassifier, CBBoostUtils.CBBoostClassifier):

    def __init__(self, n_max_iterations=10, random_state=None,
                 self_complemented=True, twice_the_same=False,
                 random_start=False, n_stumps=1, c_bound_sol=True,
                 estimators_generator="Stumps", mincq_tracking=False,
                 weight_add=3, weight_strategy="c_bound_based_broken",
                 weight_update = "multiplicative", **kwargs):
        BaseMultiviewClassifier.__init__(self, random_state)
        self.param_names = ["n_max_iterations","random_state"]
        self.distribs = [CustomRandint(5,200), [random_state]]
        self.n_max_iterations = n_max_iterations
        self.random_state = random_state
        self.self_complemented = self_complemented
        self.twice_the_same = twice_the_same
        self.random_start = random_start
        self.n_stumps = n_stumps
        self.c_bound_sol = c_bound_sol
        self.estimators_generator = estimators_generator
        self.estimators_generator_name = estimators_generator
        self.mincq_tracking = mincq_tracking
        self.plotted_metric=metrics.zero_one_loss
        self.weight_add = weight_add
        self.weight_strategy = weight_strategy
        self.weight_update = weight_update

    def init_lists(self, X, view_indices,):
        self.used_views = view_indices
        self.view_names = [X.get_view_name(view_index)
                           for view_index in view_indices]

        # Todo HDF5 compatible
        self.view_n_stumps = [self.n_stumps for _ in range(X.nb_view)]
        self.view_n_features = [X.get_v(view_index).shape[1] for view_index in
                                range(X.nb_view)]

        self.view_estimators_generator = [_ for _ in range(X.nb_view)]
        self.view_classification_matrix = [_ for _ in range(X.nb_view)]
        self.view_train_shapes = [_ for _ in range(X.nb_view)]

        self.view_chosen_columns_ = [[] for _ in range(X.nb_view)]
        self.view_new_voter = [_ for _ in range(X.nb_view)]
        self.view_previous_vote = [_ for _ in range(X.nb_view)]
        self.view_q = [_ for _ in range(X.nb_view)]
        self.view_train_metrics = [[] for _ in range(X.nb_view)]
        self.view_norm = [[] for _ in range(X.nb_view)]
        self.view_weights_ = [[] for _ in range(X.nb_view)]
        self.view_previous_margins = [[] for _ in range(X.nb_view)]
        self.view_selected_margins = [[] for _ in range(X.nb_view)]
        self.view_tau = [[] for _ in range(X.nb_view)]
        self.view_voter_perfs = [[] for _ in range(X.nb_view)]
        self.view_chosen_features = [[] for _ in range(X.nb_view)]
        self.view_previous_votes = [[] for _ in range(X.nb_view)]
        self.view_c_bounds = [[] for _ in range(X.nb_view)]
        self.view_margins = [[] for _ in range(X.nb_view)]
        self.view_disagreements = [[] for _ in range(X.nb_view)]
        self.view_decisions = [[] for _ in range(X.nb_view)]
        self.margin = [_ for _ in range(self.n_view_total)]
        self.view_considered_misclass = [_ for _ in range(self.n_view_total)]

    def fit(self, X, y, train_indices=None, view_indices=None):

        train_indices, view_indices = get_samples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        self.used_labels = y[train_indices]
        self.n_view_total = X.nb_view
        view_y_kernel_matrices = [_ for _ in range(X.nb_view)]
        view_first_voter_index = [_ for _ in range(X.nb_view)]
        self.general_voters = []
        self.general_weights = []
        self.general_c_bounds = []
        self.margins = []
        self.general_margins = []
        self.sample_weightings = []
        self.broken_views = [False for _ in range(self.n_view_total)]
        self.general_disagreements = []
        self.decisions = []
        self.considered_misclass = [np.zeros((self.n_view_total, len(train_indices)))]
        self.init_lists(X, view_indices, )
        self.sample_weighting = [np.ones((train_indices.shape[0], 1)) / train_indices.shape[0] if _ in view_indices else "" for _ in range(X.nb_view)]
        self.sample_weightings.append([s.copy() for s in self.sample_weighting])


        self.view_first_voters = [[] for _ in range(X.nb_view)]
        for view_index in view_indices:

            formatted_X, formatted_y = self.format_X_y(X.get_v(view_index, sample_indices=train_indices), y[train_indices])
            self.init_estimator_generator(view_index)
            self.view_estimators_generator[view_index].fit(formatted_X, formatted_y)

            self.view_classification_matrix[view_index] =  self.get_classification_matrix(X, train_indices, view_index)

            self.view_train_shapes[view_index] = self.view_classification_matrix[view_index].shape
            self.n_total_examples, n = self.view_classification_matrix[view_index].shape

            view_y_kernel_matrices[view_index] = np.multiply(formatted_y, self.view_classification_matrix[view_index])

            view_first_voter_index, margin = self.get_first_voter(view_index, view_first_voter_index, view_y_kernel_matrices)
            self.margin[view_index] = view_y_kernel_matrices[view_index][:,view_first_voter_index]
            self.view_first_voters[view_index] = [margin, view_first_voter_index[view_index]]
            self.init_boosting(view_index, view_first_voter_index, formatted_y)
        self.decisions.append([d.copy() for d in self.view_decisions])
        self.margins.append([m.copy() for m in self.margin])
        self.view_break_cause = [" the maximum number of iterations was attained." for _ in range(X.nb_view)]
        self.available_view_indices = view_indices.copy()
        self.get_best_view_voter()

        for boosting_iter_index in range(self.n_max_iterations):
            self.it = boosting_iter_index+1
            print("iteration ", self.it)
            for view_index in self.available_view_indices:

                self.view_q[view_index], new_voter_index = self.get_new_voter(view_index, view_y_kernel_matrices, formatted_y)

                if type(self.view_q[view_index]) == str:
                    self.view_break_cause[view_index] = new_voter_index  #
                    self.available_view_indices.remove(view_index)
                    self.broken_views[view_index] = True
                    break

                self.append_new_voter(new_voter_index, view_index)
                self.update_sample_weighting(view_index, formatted_y)
                self.update_infos(view_index, formatted_y)
            self.sample_weightings.append([s.copy() for s in self.sample_weighting])
            self.decisions.append([d.copy() for d in self.view_decisions])
            self.margins.append([m.copy() for m in self.margin])
            self.considered_misclass.append([c.copy() for c in self.view_considered_misclass])

            print("\tn_cols_chosen\t", [len(self.view_chosen_columns_[i]) for i in self.used_views])
            print("\tbroken\t\t", self.broken_views)
            self.general_q, new_voter_view_index = self.choose_new_general_voter(formatted_y)
            print("\tChosen_view\t", new_voter_view_index)
            if type(self.general_q) == str:
                self.break_cause = new_voter_view_index
                self.it -=1
                break
            self.general_voters.append([new_voter_view_index, self.view_chosen_columns_[new_voter_view_index][-1]])
            self.general_weights.append(self.general_q)



        for view_index in view_indices:
            self.view_estimators_generator[view_index].choose(self.view_chosen_columns_[view_index])
            self.view_weights_[view_index] = np.array(self.view_weights_[view_index]) / np.sum(
                np.array(self.view_weights_[view_index]))
        self.general_weights = np.array(self.general_weights)/np.sum(np.array(self.general_weights))
        # quit()
        return self

    def predict(self, X, sample_indices=None, view_indices=None):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        view_classification_matrix = [_ for _ in range(self.n_view_total)]
        vote = []
        for view_index in range(self.n_view_total):
            if view_index in view_indices:
                view_classification_matrix[view_index] = self.get_classification_matrix(X,
                                                                   sample_indices,
                                                                   view_index, )
            else:
                pass
        for iter_index, (view_index, voter_indice) in enumerate(self.general_voters):
            vote.append(view_classification_matrix[view_index][:,iter_index])
        vote = np.array(vote)
        print((vote * self.general_weights.reshape((self.it+1, 1))).shape)
        margins = np.sum(vote * self.general_weights.reshape((self.it+1, 1)), axis=0)
        print(margins.shape)
        signs_array = np.array([int(x) for x in BoostUtils.sign(margins)])
        signs_array[signs_array == -1] = 0
        return signs_array

    def transform_sample_weights(self):
        df = pd.DataFrame(columns=["weight", "view", "sample", 'iteration', "right", "margin", "mixed_sample_view", "considered_mis_class"])
        i=0
        self.min_weight = 100
        self.max_weight = -100

        for iter_index, view_sample_weights in enumerate(self.sample_weightings):
            print(iter_index)
            for view_index, sample_weights in enumerate(view_sample_weights):
                for sample_index, weight in enumerate(sample_weights):
                    weight = weight[0]*10
                    df.loc[i] = [weight, view_index, sample_index, iter_index, self.decisions[iter_index][view_index][sample_index][0], abs(self.margins[iter_index][view_index][sample_index][0]), view_index+sample_index*self.n_view_total, self.considered_misclass[iter_index][view_index][sample_index]]
                    i+=1
                    if weight < self.min_weight:
                        self.min_weight = weight
                    elif weight > self.max_weight:
                        self.max_weight = weight
        return df

    def get_interpretation(self, directory, base_file_name, labels, multiclass=False):
        self.view_importances = np.zeros(self.n_view_total)
        for (view, index), weight in zip(self.general_voters, self.general_weights):
            self.view_importances[view]+=weight
        self.view_importances/=np.sum(self.view_importances)
        interpret_string = str(self.view_importances)
        df = self.transform_sample_weights()
        import plotly.express as px
        fig = px.scatter(df, x="mixed_sample_view", y="weight", animation_frame="iteration", animation_group='mixed_sample_view', size="margin",
                   color="right", text="view", hover_name="sample", hover_data=["weight", "view", "sample", 'iteration', "right", "margin", "mixed_sample_view", "considered_mis_class"], range_x=[0, self.n_total_examples*self.n_view_total], range_y=[self.min_weight, self.max_weight]
                   )
        fig.show()
        quit()

        return interpret_string

    def update_sample_weighting(self, view_index, formatted_y,):
        weight_strategies = ['c_bound_based', ]
        import math
        print("\t 1st voter\t",  self.view_previous_vote[0][0])
        print("\t Sol\t\t", self.view_q[view_index])
        new_vote = np.multiply(self.view_previous_vote[view_index]+self.view_q[view_index]*self.view_new_voter[view_index],self.sample_weighting[view_index])
        well_class = np.zeros((self.n_total_examples, self.n_view_total))
        for view_ind in range(self.n_view_total):
            class_vote = np.multiply(
                self.view_previous_vote[view_ind] + self.view_q[view_ind] *
                self.view_new_voter[view_ind],
                self.sample_weighting[view_ind])
            margins = formatted_y * class_vote
            self.margin[view_index] = margins
            well_class[:, view_ind] = np.array([mg[0] > 0 for mg in margins])
        print(well_class)
        considered_well_class = well_class[:, view_index] + np.logical_not(well_class.any(axis=1))
        self.view_considered_misclass[view_index] = considered_well_class
        self.view_decisions[view_index] = np.array([vote[0]*y>0 for vote, y in zip(new_vote, formatted_y)])
        if self.weight_strategy == 'c_bound_based':
            c_bound_based_weighting = formatted_y*new_vote/(new_vote**2+self.weight_add)
            normalized_cbound_weights = c_bound_based_weighting+(math.sqrt(self.weight_add)/2*self.weight_add)
            normalized_cbound_weights/= np.sum(normalized_cbound_weights)
            sample_weights = normalized_cbound_weights
        elif self.weight_strategy == 'c_bound_based_broken':
            c_bound_based_weighting = np.array([y * vote - math.sqrt(self.weight_add) / (
                    (vote - math.sqrt(self.weight_add)) ** 2 + self.weight_add)
                                                if not considered_well_class[sample_index] else y * vote  + math.sqrt(self.weight_add) / (
                    (vote + math.sqrt(self.weight_add)) ** 2 + self.weight_add)
                                                for sample_index, (y, vote) in enumerate(zip(formatted_y, new_vote))]).reshape((self.n_total_examples, 1))
            # normalized_cbound_weights = c_bound_based_weighting + (
            #         math.sqrt(self.weight_add) / 2 * self.weight_add)
            sample_weights = c_bound_based_weighting/np.sum(c_bound_based_weighting)

        elif self.weight_strategy == 'c_bound_based_dec':
            c_bound_based_weighting = np.array([-vote**2 + math.sqrt(self.weight_add)/(2*self.weight_add)
                                                if not considered_well_class[sample_index] else y * vote  + math.sqrt(self.weight_add) / (
                    (vote + math.sqrt(self.weight_add)) ** 2 + self.weight_add)
                                                for sample_index, (y, vote) in enumerate(zip(formatted_y, new_vote))]).reshape((self.n_total_examples, 1))
            # normalized_cbound_weights = c_bound_based_weighting + (
            #         math.sqrt(self.weight_add) / 2 * self.weight_add)
            sample_weights = c_bound_based_weighting/np.sum(c_bound_based_weighting)

        else:
            raise ValueError("weight_strategy must be in {}, here it is {}".format(weight_strategies, self.weight_strategy))

        well_class = np.zeros((self.n_total_examples, self.n_view_total))
        for view_ind in range(self.n_view_total):
            new_vote = self.view_previous_vote[view_ind] + self.view_q[
                view_ind] * self.view_new_voter[view_ind]
            margins = formatted_y * new_vote
            self.margin[view_index] = margins
            well_class[:, view_ind] = np.array([mg[0] > 0 for mg in margins])
        min_sample_weights = np.min(sample_weights)
        max_sample_weights = np.max(sample_weights)
        sample_weights = self.normalize(sample_weights)

        if self.weight_update =="additive":
            sample_weights = self.normalize(sample_weights, range=1, min_interval=-0.5)
            self.sample_weighting[view_index] += sample_weights
        elif self.weight_update == "multiplicative":
            sample_weights = self.normalize(sample_weights, range=2,
                                            min_interval=-1)

            self.sample_weighting[view_index] *= sample_weights
        elif self.weight_update == "replacement":
            sample_weights = self.normalize(sample_weights, range=1,
                                            min_interval=0)
            self.sample_weighting[view_index] = sample_weights.reshape((self.n_total_examples,1))

        self.sample_weighting[view_index] /= np.max(self.sample_weighting[view_index])-np.min(self.sample_weighting[view_index])
        self.sample_weighting[view_index] -= np.min(self.sample_weighting[view_index])
        self.sample_weighting[view_index] /= np.sum(self.sample_weighting[view_index])
        print("\tMin\t\t", np.min(self.sample_weighting[view_index]))
        print("\tMax\t\t", np.max(self.sample_weighting[view_index]))

    def normalize(self, sample_weights, range=2, min_interval=-1.0):
        min_sample_weights = np.min(sample_weights)
        max_sample_weights = np.max(sample_weights)
        if range is None:
            pass
        else:
            sample_weights = sample_weights*(range/(max_sample_weights-min_sample_weights))-(-min_interval+(range*min_sample_weights)/(max_sample_weights-min_sample_weights))
        return sample_weights


    def get_best_view_voter(self, ):
        best_margin = 0
        for view_index, (margin, voter_index) in enumerate(self.view_first_voters):
            if margin > best_margin:
                best_margin = margin
                best_view = view_index
                best_voter = voter_index
        self.general_voters.append([best_view, best_voter])
        self.general_weights.append(1.0)
        self.general_previous_vote = np.array(
            self.view_classification_matrix[best_view][:,
            best_voter].reshape(
                (self.n_total_examples, 1)),
            copy=True)



    def choose_new_general_voter(self, formatted_y):
        previous_sum = np.multiply(formatted_y, self.general_previous_vote )
        margin_old = np.sum(previous_sum)
        worst_example = 0
        # worst_example = np.argmin(previous_sum)
        y_kernel_matrix = np.array([self.view_new_voter[view_index]*formatted_y if not self.broken_views[view_index] and view_index in self.used_views else np.zeros((self.n_total_examples, 1))-1 for view_index in range(self.n_view_total)]).reshape((self.n_total_examples, self.n_view_total))
        bad_margins = \
            np.where(np.sum(y_kernel_matrix, axis=0) <= 0.0)[
                0]
        self.B2 = self.n_total_examples
        self.B1s = np.sum(
            2 * np.multiply(previous_sum, y_kernel_matrix), axis=0)
        self.B0 = np.sum(previous_sum ** 2)

        self.A2s = np.sum(
            y_kernel_matrix, axis=0) ** 2
        self.A1s = np.sum(
            y_kernel_matrix,
            axis=0) * margin_old * 2
        self.A0 = margin_old ** 2

        C2s = (self.A1s * self.B2 - self.A2s * self.B1s)
        C1s = 2 * (self.A0 * self.B2 - self.A2s * self.B0)
        C0s = self.A0 * self.B1s - self.A1s * self.B0

        sols = np.zeros(C0s.shape) - 3
        sols[np.where(C2s != 0)[0]] = (-C1s[
            np.where(C2s != 0)[0]] + np.sqrt(
            C1s[np.where(C2s != 0)[0]] * C1s[
                np.where(C2s != 0)[0]] - 4 * C2s[
                np.where(C2s != 0)[0]] * C0s[
                np.where(C2s != 0)[0]])) / (
                                              2 * C2s[
                                          np.where(C2s != 0)[0]])

        c_bounds = self.compute_c_bounds(sols)
        print('\tCbounds\t\t', c_bounds)
        print("\tSols \t\t", sols)
        trans_c_bounds = self.compute_c_bounds(sols + 1)
        masked_c_bounds = ma.array(c_bounds, fill_value=np.inf)
        # Masing Maximums
        masked_c_bounds[c_bounds >= trans_c_bounds] = ma.masked
        # Masking magrins <= 0
        masked_c_bounds[bad_margins] = ma.masked
        print("\tbad_margins\t", bad_margins)
        # Masking weights < 0 (because self-complemented)
        # masked_c_bounds[sols < 0] = ma.masked
        # Masking nan c_bounds
        masked_c_bounds[np.isnan(c_bounds)] = ma.masked
        for view_index, broken in enumerate(self.broken_views):
            if broken:
                masked_c_bounds[view_index] = ma.masked
        print('\tCbounds\t\t', masked_c_bounds)

        if masked_c_bounds.mask.all():
            return "No more pertinent voters", 0
        else:
            best_hyp_index = np.argmin(masked_c_bounds)
            self.general_c_bounds.append(
                masked_c_bounds[best_hyp_index])
            self.general_margins.append(
                math.sqrt(self.A2s[best_hyp_index] / self.n_total_examples))
            self.general_disagreements.append(
                0.5 * self.B1s[best_hyp_index] / self.n_total_examples)
            return sols[best_hyp_index], best_hyp_index

    def update_infos(self, view_index, formatted_y):
        self.view_weights_[view_index].append(self.view_q[view_index])

        ones_matrix = np.zeros(formatted_y.shape)
        ones_matrix[
            np.multiply(formatted_y, self.view_new_voter[view_index].reshape(
                formatted_y.shape)) < 0] = 1  # can np.divide if needed
        epsilon = np.average(
            np.multiply(formatted_y, self.view_new_voter[view_index].reshape(
                formatted_y.shape)), axis=0)
        self.view_voter_perfs[view_index].append(epsilon)

        self.view_tau[view_index].append(
            np.sum(np.multiply(self.view_previous_vote[view_index],
                               self.view_new_voter[view_index])) / float(
                self.n_total_examples))
        self.view_previous_vote[view_index] += self.view_q[view_index] * \
                                               self.view_new_voter[view_index]
        self.view_norm[view_index].append(
            np.linalg.norm(self.view_previous_vote[view_index]) ** 2)
        self.view_previous_votes[view_index].append(
            self.view_previous_vote[view_index])
        self.view_previous_margins[view_index].append(
            np.sum(np.multiply(formatted_y,
                               self.view_previous_vote[view_index])) / float(
                self.n_total_examples))
        self.view_selected_margins[view_index].append(
            np.sum(np.multiply(formatted_y,
                               self.view_new_voter[view_index])) / float(
                self.n_total_examples))
        train_metric = self.plotted_metric.score(formatted_y, np.sign(
            self.view_previous_vote[view_index]))
        self.view_train_metrics[view_index].append(train_metric)

    def append_new_voter(self, new_voter_index, view_index):
        self.view_chosen_columns_[view_index].append(new_voter_index)
        if self.estimators_generator_name == "Stumps":
            self.view_chosen_features[view_index].append(
                [(int(new_voter_index % (
                        self.view_n_stumps[view_index] * self.view_n_features[
                    view_index]) / self.view_n_stumps[view_index]),
                  1)])
        elif self.estimators_generator_name == "Trees":
            self.view_chosen_features[view_index].append([(
                self.view_estimators_generator[view_index].attribute_indices[
                    new_voter_index][fake_ind],
                importance)
                for fake_ind, importance
                in enumerate(
                    self.view_estimators_generator[view_index].estimators_[
                        new_voter_index].feature_importances_)
                if importance > 0])
        self.view_new_voter[view_index] = self.view_classification_matrix[
                                              view_index][:,
                                          new_voter_index].reshape(
            (self.n_total_examples, 1))

    def get_new_voter(self, view_index, view_y_kernel_matrices, formatted_y):
        m = view_y_kernel_matrices[view_index].shape[0]
        previous_sum = np.multiply(formatted_y,
                                   (self.view_previous_vote[view_index] * self.sample_weighting[view_index]).reshape(
                                       m, 1))
        margin_old = np.sum(previous_sum)
        worst_example = 0
        # worst_example = np.argmin(previous_sum)

        bad_margins = \
        np.where(np.sum(view_y_kernel_matrices[view_index], axis=0) <= 0.0)[
            0]

        self.B2 = 1
        self.B1s = np.sum(
            2 * np.multiply(previous_sum, view_y_kernel_matrices[view_index] * self.sample_weighting[view_index]),
            axis=0)
        self.B0 = np.sum(previous_sum ** 2)

        self.A2s = np.sum(view_y_kernel_matrices[view_index] * self.sample_weighting[view_index], axis=0) ** 2
        self.A1s = np.sum(view_y_kernel_matrices[view_index] * self.sample_weighting[view_index],
                          axis=0) * margin_old * 2
        self.A0 = margin_old ** 2

        C2s = (self.A1s * self.B2 - self.A2s * self.B1s)
        C1s = 2 * (self.A0 * self.B2 - self.A2s * self.B0)
        C0s = self.A0 * self.B1s - self.A1s * self.B0

        sols = np.zeros(C0s.shape) - 3
        sols[np.where(C2s != 0)[0]] = m*(-C1s[
            np.where(C2s != 0)[0]] + np.sqrt(
            C1s[np.where(C2s != 0)[0]] * C1s[
                np.where(C2s != 0)[0]] - 4 * C2s[
                np.where(C2s != 0)[0]] * C0s[
                np.where(C2s != 0)[0]])) / (
                                              2 * C2s[
                                          np.where(C2s != 0)[0]])

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
            masked_c_bounds[self.view_chosen_columns_[view_index]] = ma.masked

        if masked_c_bounds.mask.all():
           return "No more pertinent voters", 0
        else:
            best_hyp_index = np.argmin(masked_c_bounds)
            # self.try_.append(np.ravel(previous_sum) )
            #
            # self.try_2.append(np.reshape(previous_sum ** 2, (87,)) + (2 * sols[best_hyp_index]*y_kernel_matrix[:, best_hyp_index]*np.reshape(previous_sum, (87, ))))
            self.view_c_bounds[view_index].append(
                masked_c_bounds[best_hyp_index])
            self.view_margins[view_index].append(
                math.sqrt(self.A2s[best_hyp_index]))
            self.view_disagreements[view_index].append(
                0.5 * self.B1s[best_hyp_index])
            return sols[best_hyp_index], best_hyp_index

    def compute_c_bounds(self, sols):
        return 1 - (self.A2s * sols ** 2 + self.A1s * sols + self.A0) / (
                                                                                 self.B2 * sols ** 2 + self.B1s * sols + self.B0)

    def init_boosting(self, view_index, view_first_voter_index, formatted_y):
        self.view_chosen_columns_[view_index].append(
            view_first_voter_index[view_index])
        self.view_new_voter[view_index] = np.array(
            self.view_classification_matrix[view_index][:,
            view_first_voter_index[view_index]].reshape(
                (self.n_total_examples, 1)),
            copy=True)

        self.view_previous_vote[view_index] = self.view_new_voter[view_index]
        self.view_norm[view_index].append(
            np.linalg.norm(self.view_previous_vote[view_index]) ** 2)
        self.view_q[view_index] = 1
        self.view_weights_[view_index].append(self.view_q[view_index])

        self.view_previous_margins.append(
            np.sum(np.multiply(formatted_y,
                               self.view_previous_vote[view_index])) / float(
                self.n_total_examples))
        self.view_selected_margins[view_index].append(
            np.sum(
                np.multiply(formatted_y, self.view_previous_vote[view_index])))
        self.view_tau[view_index].append(
            np.sum(np.multiply(self.view_previous_vote[view_index],
                               self.view_new_voter[view_index])) / float(
                self.n_total_examples))

        train_metric = self.plotted_metric.score(formatted_y, np.sign(
            self.view_previous_vote[view_index]))
        self.view_train_metrics[view_index].append(train_metric)

    def get_first_voter(self, view_index, view_first_voter_index, view_y_kernel_matrices):
        if self.random_start:
            view_first_voter_index[view_index] = self.random_state.choice(
                np.where(
                    np.sum(view_y_kernel_matrices[view_index], axis=0) > 0)[0])
            margin = np.sum(view_y_kernel_matrices[view_index][:, view_first_voter_index[view_index]] * self.sample_weighting[view_index])
        else:
            pseudo_h_values = ma.array(
                np.sum(view_y_kernel_matrices[view_index] * self.sample_weighting[view_index], axis=0),
                fill_value=-np.inf)
            view_first_voter_index[view_index] = np.argmax(pseudo_h_values)

            margin = pseudo_h_values[view_first_voter_index[view_index]]
        self.view_decisions[view_index] = (view_y_kernel_matrices[view_index][:,view_first_voter_index[view_index]] > 0).reshape((self.n_total_examples, 1))
        return view_first_voter_index, margin

    def init_estimator_generator(self, view_index):
        if self.estimators_generator is "Stumps":
            self.view_estimators_generator[
                view_index] = BoostUtils.StumpsClassifiersGenerator(
                n_stumps_per_attribute=self.n_stumps,
                self_complemented=self.self_complemented)
        if self.estimators_generator is "Trees":
            self.view_estimators_generator[
                view_index] = BoostUtils.TreeClassifiersGenerator(
                n_trees=self.n_stumps, max_depth=self.max_depth,
                self_complemented=self.self_complemented)

    def get_view_vote(self ,X, sample_indices, view_index,):
        classification_matrix = self.get_classification_matrix(X,
                                                               sample_indices,
                                                               view_index, )

        margins = np.sum(classification_matrix * self.view_weights_[view_index],
                         axis=1)
        signs_array = np.array([int(x) for x in BoostUtils.sign(margins)])
        signs_array[signs_array == -1] = 0
        return signs_array

    def get_classification_matrix(self, X, sample_indices, view_index, ):
        if self.view_estimators_generator[view_index].__class__.__name__ == "TreeClassifiersGenerator":
            probas = np.asarray(
                [clf.predict_proba(
                    X.get_v(view_index, sample_indices)[:, attribute_indices])
                 for
                 clf, attribute_indices in
                 zip(self.view_estimators_generator[view_index].estimators_,
                     self.view_estimators_generator[
                         view_index].attribute_indices)])
        else:
            probas = np.asarray(
                [clf.predict_proba(X.get_v(view_index, sample_indices)) for clf
                 in
                 self.view_estimators_generator[view_index].estimators_])
        predicted_labels = np.argmax(probas, axis=2)
        predicted_labels[predicted_labels == 0] = -1
        values = np.max(probas, axis=2)
        return (predicted_labels * values).T

