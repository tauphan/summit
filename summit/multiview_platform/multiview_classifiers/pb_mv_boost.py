import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from ..multiview.multiview_utils import get_available_monoview_classifiers, \
    BaseMultiviewClassifier, ConfigGenerator
from ..utils.dataset import get_samples_views_indices
from ..utils.multiclass import get_mc_estim, MultiClassWrapper
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.transformations import sign_labels, unsign_labels

"""
This code is implements the C-bound optimization problem for the Multiview learning algorithm PB-MVBoost.

Related Paper:
Multiview Boosting by Controlling the Diversity and the Accuracy of View-specific Voters
by Anil Goyal, Emilie Morvant, Pascal Germain and Massih-Reza Amini

Based on the code of Anil Goyal, 
Link to the paper:
https://arxiv.org/abs/1808.05784
"""

classifier_class_name = "PBMVBoost"


class MV_Cbount_opt(object):
    """
    This class solves the C-bound optimization problem for the Multiview learning algorithm PB-MVBoost.
    It learns the weights for over the views for our algorithm.
    """
    def __init__(self,initial_guess,risk_vector,disagreement_vector):
        """

        :param initial_guess: vector for the initial guess of weights
        :param risk_vector: Risk vector
        :param disagreement_vector: Vector for disagreement values
        """

        self.initial_guess=initial_guess
        self.risk_vector=risk_vector
        self.disagreement_vector=disagreement_vector



    def func(self, x, r,d,sign=1):
        """ Objective function """
        num=1-2 * (sum(x*r))
        den=1-2 * (sum(x*d))

        return sign * ((num)**2 / den)

    def func_deriv(self, x, r,d, sign=1):
        """ Derivative of objective function """
        num = 1 - 2 * (sum(x*r))
        den = 1 - 2 * (sum(x*d))

        dfdx= sign * ((-1 * 4 * r * num * den + 2 * d * (num)**2) / (den ** 2))

        return np.array(dfdx)


    def learn_weights(self):
        """
        Learns weights
        :param self:
        :return:
        """
        x = self.initial_guess
        r = self.risk_vector
        d = self.disagreement_vector
        arguments = (r, d, -1)




        cons = ({'type': 'eq',
                        'fun': lambda x: np.array([sum(x) - 1]),
                         'jac': lambda x: np.array(x)}
                        )

        res = minimize(self.func, x, args=arguments, bounds=tuple((0,None) for i in range(len(x))), jac=self.func_deriv,
                               constraints=cons, method='SLSQP', options={'disp': False})


        if np.isnan(res.x[0]):
            return self.initial_guess
        else:
            return res.x

class PBMVBoost(BaseMultiviewClassifier):

    def __init__(self, random_state=None, num_iterations=100,
                 decision_tree_depth=1):
        BaseMultiviewClassifier.__init__(self, random_state=random_state)
        self.num_iterations = num_iterations
        self.decision_tree_depth = decision_tree_depth
        self.param_names = ["num_iterations", "decision_tree_depth"]
        self.distribs = [CustomRandint(1,100), CustomRandint(1,3)]

        # Variables to store the train and test predictions and weights after each iteration


    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, views_indices = get_samples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        view_names = [X.get_view_name(view_ind)
                      for view_ind in views_indices]
        X_train = dict((X.get_view_name(view_ind), X.get_v(view_ind,
                                                           train_indices))
                      for view_ind in views_indices)
        y_train = dict((X.get_view_name(view_ind),sign_labels(y[train_indices]))
                       for view_ind in views_indices)
        self.train_predictions_classifiers = dict((name, []) for name in view_names)
        self.weak_classifiers = dict((name, []) for name in view_names)
        self.weights_classfiers = dict((name, []) for name in view_names)
        return self.learn(X_train, y_train, all_views=view_names)

    def predict(self, X, sample_indices=None,
                view_indices=None):
        sample_indices, views_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        X = dict((X.get_view_name(view_ind), X.get_v(view_ind,
                                                           sample_indices))
                       for view_ind in views_indices)
        pred = self._compute_majority_vote(X)
        pred[pred==-1]=0
        return pred


    def _compute_weight(self,error,view_index):
        """
        This function is helping function to compute weight of hypothesis based
        on error pased to it.
        It Computes 0.5 * ln (1-error/ error)
        :param error: Error value
        :return: Weight value
        """
        view_weight = self.rho[view_index]
        if view_weight == 0:
            return 0
        else:

            return 0.5 * np.log((1 - error) / (float(error)))
            #return 0.5 * (1 / float(view_weight)) * (np.log((1 - error) / (float(error))) + 2)

    def _learn_classifier(self,X_train, y_train, name_of_view, view_index,
                          example_weights):
        """
        This function learns the weak classifier and returns weight for this learned classifier. Fitting is
        done on weighted samples which is passed as an input parameter.

        Input
        ======
        :param name_of_view : View name for which we need to learn classfier
        :param example_weights : Weight of input training examples

        :return: Weight of Classifier, training data labels, test data labels.

        """

        #learning classifier
        clf = DecisionTreeClassifier(max_depth=self.decision_tree_depth, random_state=1, splitter='random')
        clf.fit(X_train[name_of_view], y_train[name_of_view], sample_weight=example_weights)  # fitting model according to weighted samples.
        self.weak_classifiers[name_of_view].append(clf)
        #predicting lables for training and test data
        predicted_labels_train = clf.predict(X_train[name_of_view])
        # predicted_labels_test = clf.predict(X_test)
        #computing error
        error_t = [int(x) for x in (predicted_labels_train != y_train[name_of_view])]
        error_t_weighted = np.dot(example_weights, error_t) / sum(example_weights)

        # Reweighing the Weights of hypothesis if weighted error is zero to avoid warning at step 7 of algorithm.
        if error_t_weighted == 0:
            error_t_weighted = 0.1 * min(example_weights)

        # Compute hypothesis weight (Line 7 of Algorithm)

        Q_t = self._compute_weight(error_t_weighted,view_index)


        return Q_t, predicted_labels_train

    def _learn_view_weights(self, y_train, initial_guess, example_weights):
        """
        This function learns the weights over views.

        :param initial_guess: initial weights over views.

        :return: rho : weight over views.
        """
        errors_t = []
        disaggrement_t = []

        # Computing View-Specific Error and disagreement on weighted training data.
        for name_of_view in self.all_views:

            classifier_errors = []
            paired_disagreements = []

            # compute view-specific error
            for classifier_output in self.train_predictions_classifiers[name_of_view]:
                error = [int(x) for x in (classifier_output != y_train[name_of_view])]
                weighted_error = np.dot(example_weights, error) / sum(example_weights)
                classifier_errors.append(weighted_error)

            classifier_errors = np.array(classifier_errors)
            classifier_weights = np.array(self.weights_classfiers[name_of_view])
            errors_t.append(sum(classifier_errors * classifier_weights))

            # compute view-specific disagreement
            for index_1, classifier_output_1 in enumerate(self.train_predictions_classifiers[name_of_view]):
                for index_2, classifier_output_2 in enumerate(self.train_predictions_classifiers[name_of_view]):
                    disagreement = [int(x) for x in (classifier_output_1 != classifier_output_2)]
                    weighted_disagreement = np.dot(example_weights, disagreement) / sum(example_weights)

                    classifier_weights = np.array(self.weights_classfiers[name_of_view])

                    weight_1 = classifier_weights[index_1]
                    weight_2 = classifier_weights[index_2]

                    paired_disagreements.append(weighted_disagreement * weight_1 * weight_2)

            disaggrement_t.append(sum(paired_disagreements))

        optimize = MV_Cbount_opt(initial_guess, np.array(errors_t), np.array(disaggrement_t))
        rho = optimize.learn_weights()

        return rho

    def _compute_Cbound(self,risk, disagreement):
        """
        This function computes the C-Bound on the value of gibbs risk and gibbs disagreement.
        :return: C-bound value
        """
        C_bound=1-((1-2*risk)**2 / (1-2*disagreement))
        return C_bound

    def _compute_majority_vote(self, X):
        """More sklearn-ish version of _calculate_majority_vote"""
        predictions = predictions = np.zeros(X[list(X.keys())[0]].shape[0])
        weak_outputs = dict((name,
                      np.array([clf.predict(X[name])
                                for clf in self.weak_classifiers[name]]))
                     for name in self.all_views)
        for view_index, name_of_view in enumerate(self.all_views):
            for t, output in enumerate(weak_outputs[name_of_view]):
                classifier_weights = np.array(self.weights_classfiers[name_of_view])
                predictions = predictions + self.rho[view_index] * classifier_weights[t] * output

        predictions = np.sign(predictions)

        return predictions

    def _calculate_majority_vote(self,data='train'):
        """
        This function calculates the majority vote

        :param data : tells on which data we need to compute the majority vote

        :return: predictions of majority vote
        """
        if data == 'train':
            predictions = np.zeros(self.num_train_examples)
            classifiers_outputs = self.train_predictions_classifiers

        elif data == 'test':
            predictions = np.zeros(self.num_test_examples)
            classifiers_outputs = self.test_predictions_classfiers

        for view_index, name_of_view in enumerate(self.all_views):
            for t, output in enumerate(classifiers_outputs[name_of_view]):
                classifier_weights = np.array(self.weights_classfiers[name_of_view])
                predictions = predictions + self.rho[view_index] * classifier_weights[t] * output

        predictions = np.sign(predictions)

        return predictions

    def _mv_cbound(self, data = 'train', y_train=None):
        """
        This function will compute the 2nd form of multiview c-bound for mv-boost.

        :param data : this parameter will tell on which data we have to compute the c-bound.

        :return: the value of c-bound on input data.
        """

        if data == 'train':
            predictions =  self.train_predictions_classifiers
            labels = y_train

        errors_t = []
        disaggrement_t = []
        # example_weights = np.ones(self.num_train_examples) / self.num_train_examples # to not to consider example weights.
        # Computing View-Specific Error and disagreement on weighted training data.(Line 11-12)
        for name_of_view in self.all_views:

            classifier_errors = []
            paired_disagreements = []

            # compute view-specific error (Line 11)
            for classifier_output in predictions[name_of_view]:
                error = [int(x) for x in (classifier_output != labels[name_of_view])]
                weighted_error = np.mean(error)
                classifier_errors.append(weighted_error)

            classifier_errors = np.array(classifier_errors)
            classifier_weights = np.array(self.weights_classfiers[name_of_view]) / sum(np.array(self.weights_classfiers[name_of_view]))
            errors_t.append(sum(classifier_errors * classifier_weights))

            # compute view-specific disagreement (Line 12)
            for index_1, classifier_output_1 in enumerate(predictions[name_of_view]):
                for index_2, classifier_output_2 in enumerate(predictions[name_of_view]):
                    disagreement = [int(x) for x in (classifier_output_1 != classifier_output_2)]
                    weighted_disagreement = np.mean(disagreement)
                    classifier_weights = np.array(self.weights_classfiers[name_of_view]) / sum(np.array(self.weights_classfiers[name_of_view]))

                    weight_1 = classifier_weights[index_1]
                    weight_2 = classifier_weights[index_2]

                    paired_disagreements.append(weighted_disagreement * weight_1 * weight_2)

            disaggrement_t.append(sum(paired_disagreements))

        rho = np.array(self.rho)
        risk_total = sum(np.array(errors_t) * rho)
        disagreement_total = sum(np.array(disaggrement_t) * rho)
        c_bound = self._compute_Cbound(risk_total,disagreement_total)

        return c_bound

    def _compute_stats(self,predicted_values,true_values):
        """
        This function returns the error and f1-score.
        :param predicted_values:  Predicted labels of any estimator
        :param true_values: True labels
        :return:
        """
        # removing the elements with output zero.
        zero_indices = np.where(predicted_values == 0)[0]
        predicted_values = np.delete(predicted_values, zero_indices)
        true_values = np.delete(true_values, zero_indices)
        error = np.mean(predicted_values * true_values <= 0.0)
        # f1 = f1_score(y_true=true_values, y_pred=predicted_values)

        return error

    def learn(self, X_train, y_train, all_views):
        """
        This function will learn the mvboost model for input multiview learning data.
        :return: Accuracy and F1 Measure on Training and Test Data. Also, Multiview C-Bound value on Training Data
                after T iterations.
        """
        self.num_train_examples = X_train[list(X_train.keys())[0]].shape[0]
        self.all_views=all_views

        #Initializing weights for training data (Line 1 and 2 of Algorithm)
        w=np.ones(self.num_train_examples) / self.num_train_examples

        # T Iterations iterations. (Beginnning of loop at line 4 of Algorithm)
        for t in range(self.num_iterations):
            if t == 0:
                self.rho = np.ones(len(self.all_views)) / len(self.all_views)  # Line 3 of Algorithm

            print("Iteration: " + str(t+1) + "\n")

            #Learning view-specific classifiers and weights over them (Line 5-7)
            for view_index,name_of_view in enumerate(self.all_views):
                    Q_t, predicted_labels_train= self._learn_classifier(X_train, y_train, name_of_view,view_index,example_weights=w)

                    #Storing the view-specific train and test outputs along with hypothesis weights
                    self.train_predictions_classifiers[name_of_view].append(predicted_labels_train)
                    self.weights_classfiers[name_of_view].append(Q_t)

            #Computing weights over views (Line 8)
            if t == 0:
                    self.rho=np.ones(len(self.all_views)) / len(self.all_views) #Line 9 of Algorithm.
                    self.rho_vectors=[]
                    self.rho_vectors.append(self.rho)
            else:
                    initial_guess = np.ones(len(self.all_views)) / len(self.all_views)
                    self.rho = self._learn_view_weights(y_train, initial_guess, w)
                    self.rho_vectors.append(self.rho)


            # Update  weights over training sample (Line 9-10)
            train_predictions=np.zeros(self.num_train_examples)
            for index,name_of_view in enumerate(self.all_views):
                classifier_weights = np.array(self.weights_classfiers[name_of_view])
                predictions=self.rho[index]*classifier_weights[-1]*self.train_predictions_classifiers[name_of_view][-1]
                train_predictions=train_predictions+predictions
            w = w * np.exp(-train_predictions * y_train[name_of_view])
            w=w/sum(w)

            # Computing Majority-vote error and f1-measure at each iteration.
            # test_predictions = self._calculate_majority_vote(data='test')
            train_predictions = self._calculate_majority_vote(data='train')


            # error_test, f1_test = self._compute_stats(predicted_values=test_predictions,true_values=self.y_test[name_of_view])
            error_train = self._compute_stats(predicted_values=train_predictions,true_values=y_train[name_of_view])


            c_bound_train = self._mv_cbound(data='train', y_train=y_train)

            print("Accuracy on Training Data: " + str(1 - np.array(error_train)) + "\n")
            # print("F1 Score on Training Data: " + str( np.array(f1_train)) + "\n")
            print("Multiview C-Bound  Training Data: " + str(np.array(c_bound_train)) + "\n")
            # print("Accuracy on Test Data: " + str(1 - np.array(error_test)) + "\n")
            # print("F1 Score on Test Data: " + str( np.array(f1_test)) + "\n")
            print("=========================================== \n")


        return self