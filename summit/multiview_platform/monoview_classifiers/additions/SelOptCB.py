import logging
import math
import time
import os

import numpy as np
import numpy.ma as ma
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from ...monoview.monoview_utils import BaseMonoviewClassifier

def change_label_to_minus(y):
    """
    Change the label 0 to minus one

    Parameters
    ----------
    y :

    Returns
    -------
    label y with -1 instead of 0

    """
    minus_y = np.copy(y)
    minus_y[np.where(y == 0)] = -1
    return minus_y

class BaseBoost(object):

    def _collect_probas(self, X, sub_sampled=False):
        if self.estimators_generator.__class__.__name__ == "TreeClassifiersGenerator":
            return np.asarray([clf.predict_proba(X[:, attribute_indices]) for
                               clf, attribute_indices in
                               zip(self.estimators_generator.estimators_,
                                   self.estimators_generator.attribute_indices)])
        else:
            return np.asarray([clf.predict_proba(X) for clf in
                               self.estimators_generator.estimators_])

    def _binary_classification_matrix(self, X):
        probas = self._collect_probas(X)
        predicted_labels = np.argmax(probas, axis=2)
        predicted_labels[predicted_labels == 0] = -1
        values = np.max(probas, axis=2)
        return (predicted_labels * values).T

    def _initialize_alphas(self, n_examples):
        raise NotImplementedError(
            "Alpha weights initialization function is not implemented.")

    def check_opposed_voters(self, ):
        nb_opposed = 0
        oppposed = []
        for column in self.classification_matrix[:,
                      self.chosen_columns_].transpose():
            for chosen_col in self.chosen_columns_:
                if (-column.reshape((self.n_total_examples,
                                     1)) == self.classification_matrix[:,
                                            chosen_col].reshape(
                        (self.n_total_examples, 1))).all():
                    nb_opposed += 1
                    break
        return int(nb_opposed / 2)

def sign(array):
    """Computes the elementwise sign of all elements of an array. The sign function returns -1 if x <=0 and 1 if x > 0.
    Note that numpy's sign function can return 0, which is not desirable in most cases in Machine Learning algorithms.

    Parameters
    ----------
    array : array-like
        Input values.

    Returns
    -------
    ndarray
        An array with the signs of input elements.

    """
    signs = np.sign(array)

    signs[array == 0] = -1
    return signs

class CboundStumpBuilder(BaseEstimator, ClassifierMixin):

    def __init__(self, col_ind=0):
        super(CboundStumpBuilder, self).__init__()
        self.col_ind = col_ind

    def fit(self, X, y, base_vote=None, it_ind=0):
        self.n_samples = X.shape[0]
        X = X[:, self.col_ind].reshape((self.n_samples, 1))
        self.sorted_inds = np.argsort(X, axis=0).reshape(self.n_samples)
        ord_X = X[self.sorted_inds, :]
        intervals = np.zeros((self.n_samples-1, 2))
        intervals[:, 0] = ord_X[:-1, 0]
        intervals[:, 1] = ord_X[1:, 0]
        thresholds = np.mean(intervals, axis=1)
        preds = np.array([np.array([(X[i] - th) / np.abs(X[i] - th) for th in thresholds]) for i in range(X.shape[0])])[:,:,0]
        # print(preds.shape)
        gg = np.sum(preds * y, axis=0) / self.n_samples
        t = np.sum(preds * base_vote, axis=0)/self.n_samples
        ng = np.sum(preds * preds, axis=0)/self.n_samples
        base_margin = np.sum(base_vote * y)
        base_norm = np.sum(np.square(base_vote))
        nf = base_norm / self.n_samples
        gf = base_margin / self.n_samples
        weights = (-4 * gf * gg * t + 4 * np.square(
            gg) * nf) / (
                          4 * gf * gg * ng - 4 * np.square(
                      gg) * t)
        cb2 = 1 - (gg * weights + gf) ** 2 / (
                nf + 2 * weights * t + ng * weights ** 2)
        cb2[np.isnan(cb2)] = np.inf
        self.th = thresholds[np.argmin(cb2)]
        self.weight =  weights[np.argmin(cb2)]
        self.cb = cb2[np.argmin(cb2)]

    def predict(self, X, ):
        X = X[:, self.col_ind]
        return np.transpose(np.array([(X[i] - self.th) / np.abs(X[i] - self.th) if X[i]!=self.th else 0 for i in range(X.shape[0])]))


class CboundPseudoStumpBuilder(BaseEstimator, ClassifierMixin):

    def __init__(self, col_ind=0):
        super(CboundPseudoStumpBuilder, self).__init__()
        self.col_ind = col_ind

    def fit(self, X, y, base_vote=None, it_ind=0):
        self.n_samples = X.shape[0]
        X = X[:, self.col_ind].reshape((self.n_samples, 1))
        self.sorted_inds = np.argsort(X, axis=0).reshape(self.n_samples)
        ord_X = X[self.sorted_inds, :]
        intervals = np.zeros((self.n_samples-1, 2))
        intervals[:, 0] = ord_X[:-1, 0]
        intervals[:, 1] = ord_X[1:, 0]
        m1 = np.mean(X) + 2 * np.std(X)
        m2 = np.mean(X) - 2 * np.std(X)
        thresholds = np.mean(intervals, axis=1)
        preds = np.array([np.array([(X[i] - th) / (m1-th) if X[i] > th else (X[i] - th) / (th-m2) for th in thresholds]) for i in range(X.shape[0])])[:,:,0]
        # print(preds.shape)
        gg = np.sum(preds * y, axis=0) / self.n_samples
        t = np.sum(preds * base_vote, axis=0)/self.n_samples
        ng = np.sum(preds * preds, axis=0)/self.n_samples
        base_margin = np.sum(base_vote * y)
        base_norm = np.sum(np.square(base_vote))
        nf = base_norm / self.n_samples
        gf = base_margin / self.n_samples
        weights = (-4 * gf * gg * t + 4 * np.square(
            gg) * nf) / (
                          4 * gf * gg * ng - 4 * np.square(
                      gg) * t)
        cb2 = 1 - (gg * weights + gf) ** 2 / (
                nf + 2 * weights * t + ng * weights ** 2)
        cb2[np.isnan(cb2)] = np.inf
        self.m1 = m1
        self.m2 = m2
        self.th = thresholds[np.argmin(cb2)]
        self.weight =  weights[np.argmin(cb2)]
        self.cb = cb2[np.argmin(cb2)]

    def predict(self, X, ):
        X = X[:, self.col_ind]
        return np.transpose(np.array([(X[i] - self.th) / (self.m1-self.th) if X[i] > self.th else (X[i] - self.th) / (self.th-self.m2) for i in range(X.shape[0])]))


class CBoundThresholdFinder(BaseEstimator, ClassifierMixin):

    def __init__(self, col_ind=0):
        super(CBoundThresholdFinder, self).__init__()
        self.col_ind = col_ind

    def fit(self, X, y, base_vote=None, it_ind=0):
        if len(np.unique(X[:, self.col_ind])) == 1:
            self.th = X[0, self.col_ind]
            self.cb = 1.0
            self.weight = 0.0
            self.no_th=True
            return self

        self.n_samples = X.shape[0]
        X = X[:, self.col_ind].reshape((self.n_samples,1))
        self.sorted_inds = np.argsort(X, axis=0).reshape(self.n_samples)
        self.intervals = np.zeros(self.n_samples + 1)
        self.intervals[0] = -np.inf
        self.intervals[-1] = np.inf
        self.intervals[1:-1] = X[self.sorted_inds[1:], 0]
        ord_X = X[self.sorted_inds, :]
        ord_y = y[self.sorted_inds, :]
        ord_base_vote = base_vote[self.sorted_inds, :]
        infs = np.transpose(np.tri(self.n_samples, dtype=int))
        sups = np.tri(self.n_samples, k=1, dtype=int)
        e11 = np.sum(sups * ord_y, axis=0) / self.n_samples
        e12 = np.sum(ord_y.reshape((self.n_samples, 1)) * infs,
                     axis=0) / self.n_samples
        e21 = np.sum(ord_base_vote * sups, axis=0) / self.n_samples
        e22 = np.sum(ord_base_vote * infs, axis=0) / self.n_samples
        e31 = -np.sum(2 * ord_X * sups, axis=0) / self.n_samples
        e32 = -np.sum(2 * ord_X * infs, axis=0) / self.n_samples
        d11 = np.sum((ord_X * ord_y) * sups, axis=0) / self.n_samples
        d12 = np.sum((ord_X * ord_y) * infs, axis=0) / self.n_samples
        d21 = np.sum((ord_X * ord_base_vote) * sups,
                     axis=0) / self.n_samples
        d22 = np.sum((ord_X * ord_base_vote) * infs,
                     axis=0) / self.n_samples
        d31 = np.sum((ord_X * ord_X) * sups, axis=0) / self.n_samples
        d32 = np.sum((ord_X * ord_X) * infs, axis=0) / self.n_samples

        f31 = np.sum(sups, axis=0) / self.n_samples
        f32 = np.sum(infs, axis=0) / self.n_samples

        base_margin = np.sum(ord_base_vote * ord_y)
        base_norm = np.sum(np.square(ord_base_vote))
        nf = base_norm / self.n_samples
        gf = base_margin / self.n_samples
        m1 = np.mean(X) + 2 * np.std(X)
        m2 = np.mean(X) - 2 * np.std(X)
        # #
        a_ = (-2 * d11 * e21 ** 2 + 4 * d11 * e21 * e22 - 2 * d11 * e22 ** 2 + 2 * d11 * f31 * nf + 2 * d11 * f32 * nf + 2 * d12 * e21 ** 2 - 4 * d12 * e21 * e22 + 2 * d12 * e22 ** 2 - 2 * d12 * f31 * nf - 2 * d12 * f32 * nf + 2 * d21 * e11 * e21 - 2 * d21 * e11 * e22 - 2 * d21 * e12 * e21 + 2 * d21 * e12 * e22 - 2 * d21 * f31 * gf - 2 * d21 * f32 * gf - 2 * d22 * e11 * e21 + 2 * d22 * e11 * e22 + 2 * d22 * e12 * e21 - 2 * d22 * e12 * e22 + 2 * d22 * f31 * gf + 2 * d22 * f32 * gf - 2 * e11 * e21 * e22 * m1 + 2 * e11 * e21 * e22 * m2 + 2 * e11 * e22 ** 2 * m1 - 2 * e11 * e22 ** 2 * m2 + e11 * e31 * nf + e11 * e32 * nf - 2 * e11 * f32 * m1 * nf + 2 * e11 * f32 * m2 * nf + 2 * e12 * e21 ** 2 * m1 - 2 * e12 * e21 ** 2 * m2 - 2 * e12 * e21 * e22 * m1 + 2 * e12 * e21 * e22 * m2 - e12 * e31 * nf - e12 * e32 * nf - 2 * e12 * f31 * m1 * nf + 2 * e12 * f31 * m2 * nf - e21 * e31 * gf - e21 * e32 * gf + 2 * e21 * f32 * gf * m1 - 2 * e21 * f32 * gf * m2 + e22 * e31 * gf + e22 * e32 * gf + 2 * e22 * f31 * gf * m1 - 2 * e22 * f31 * gf * m2)
        b_ = (2 * d11 * d21 * e21 - 2 * d11 * d21 * e22 - 2 * d11 * d22 * e21 + 2 * d11 * d22 * e22 + 6 * d11 * e21 ** 2 * m2 - 2 * d11 * e21 * e22 * m1 - 10 * d11 * e21 * e22 * m2 + 2 * d11 * e22 ** 2 * m1 + 4 * d11 * e22 ** 2 * m2 + d11 * e31 * nf + d11 * e32 * nf - 6 * d11 * f31 * m2 * nf - 2 * d11 * f32 * m1 * nf - 4 * d11 * f32 * m2 * nf - 2 * d12 * d21 * e21 + 2 * d12 * d21 * e22 + 2 * d12 * d22 * e21 - 2 * d12 * d22 * e22 - 4 * d12 * e21 ** 2 * m1 - 2 * d12 * e21 ** 2 * m2 + 10 * d12 * e21 * e22 * m1 + 2 * d12 * e21 * e22 * m2 - 6 * d12 * e22 ** 2 * m1 - d12 * e31 * nf - d12 * e32 * nf + 4 * d12 * f31 * m1 * nf + 2 * d12 * f31 * m2 * nf + 6 * d12 * f32 * m1 * nf - 2 * d21 ** 2 * e11 + 2 * d21 ** 2 * e12 + 4 * d21 * d22 * e11 - 4 * d21 * d22 * e12 - 6 * d21 * e11 * e21 * m2 + 4 * d21 * e11 * e22 * m1 + 2 * d21 * e11 * e22 * m2 - 2 * d21 * e12 * e21 * m1 + 8 * d21 * e12 * e21 * m2 - 2 * d21 * e12 * e22 * m1 - 4 * d21 * e12 * e22 * m2 - d21 * e31 * gf - d21 * e32 * gf + 6 * d21 * f31 * gf * m2 + 2 * d21 * f32 * gf * m1 + 4 * d21 * f32 * gf * m2 - 2 * d22 ** 2 * e11 + 2 * d22 ** 2 * e12 + 4 * d22 * e11 * e21 * m1 + 2 * d22 * e11 * e21 * m2 - 8 * d22 * e11 * e22 * m1 + 2 * d22 * e11 * e22 * m2 - 2 * d22 * e12 * e21 * m1 - 4 * d22 * e12 * e21 * m2 + 6 * d22 * e12 * e22 * m1 + d22 * e31 * gf + d22 * e32 * gf - 4 * d22 * f31 * gf * m1 - 2 * d22 * f31 * gf * m2 - 6 * d22 * f32 * gf * m1 + 2 * d31 * e11 * nf - 2 * d31 * e12 * nf - 2 * d31 * e21 * gf + 2 * d31 * e22 * gf + 2 * d32 * e11 * nf - 2 * d32 * e12 * nf - 2 * d32 * e21 * gf + 2 * d32 * e22 * gf + 2 * e11 * e21 * e22 * m1 * m2 - 2 * e11 * e21 * e22 * m2 ** 2 - 2 * e11 * e22 ** 2 * m1 ** 2 + 2 * e11 * e22 ** 2 * m1 * m2 - 3 * e11 * e31 * m2 * nf - 4 * e11 * e32 * m1 * nf + e11 * e32 * m2 * nf + 2 * e11 * f32 * m1 ** 2 * nf - 2 * e11 * f32 * m1 * m2 * nf - 2 * e12 * e21 ** 2 * m1 * m2 + 2 * e12 * e21 ** 2 * m2 ** 2 + 2 * e12 * e21 * e22 * m1 ** 2 - 2 * e12 * e21 * e22 * m1 * m2 - e12 * e31 * m1 * nf + 4 * e12 * e31 * m2 * nf + 3 * e12 * e32 * m1 * nf + 2 * e12 * f31 * m1 * m2 * nf - 2 * e12 * f31 * m2 ** 2 * nf + 3 * e21 * e31 * gf * m2 + 4 * e21 * e32 * gf * m1 - e21 * e32 * gf * m2 - 2 * e21 * f32 * gf * m1 ** 2 + 2 * e21 * f32 * gf * m1 * m2 + e22 * e31 * gf * m1 - 4 * e22 * e31 * gf * m2 - 3 * e22 * e32 * gf * m1 - 2 * e22 * f31 * gf * m1 * m2 + 2 * e22 * f31 * gf * m2 ** 2)
        c_ = (-6 * d11 * d21 * e21 * m2 + 6 * d11 * d21 * e22 * m2 + 6 * d11 * d22 * e21 * m2 - 6 * d11 * d22 * e22 * m2 - 6 * d11 * e21 ** 2 * m2 ** 2 + 6 * d11 * e21 * e22 * m1 * m2 + 6 * d11 * e21 * e22 * m2 ** 2 - 6 * d11 * e22 ** 2 * m1 * m2 - 3 * d11 * e31 * m2 * nf - 3 * d11 * e32 * m2 * nf + 6 * d11 * f31 * m2 ** 2 * nf + 6 * d11 * f32 * m1 * m2 * nf + 6 * d12 * d21 * e21 * m1 - 6 * d12 * d21 * e22 * m1 - 6 * d12 * d22 * e21 * m1 + 6 * d12 * d22 * e22 * m1 + 6 * d12 * e21 ** 2 * m1 * m2 - 6 * d12 * e21 * e22 * m1 ** 2 - 6 * d12 * e21 * e22 * m1 * m2 + 6 * d12 * e22 ** 2 * m1 ** 2 + 3 * d12 * e31 * m1 * nf + 3 * d12 * e32 * m1 * nf - 6 * d12 * f31 * m1 * m2 * nf - 6 * d12 * f32 * m1 ** 2 * nf + 6 * d21 ** 2 * e11 * m2 - 6 * d21 ** 2 * e12 * m2 - 6 * d21 * d22 * e11 * m1 - 6 * d21 * d22 * e11 * m2 + 6 * d21 * d22 * e12 * m1 + 6 * d21 * d22 * e12 * m2 + 6 * d21 * e11 * e21 * m2 ** 2 - 6 * d21 * e11 * e22 * m1 * m2 - 6 * d21 * e12 * e21 * m2 ** 2 + 6 * d21 * e12 * e22 * m1 * m2 + 3 * d21 * e31 * gf * m2 + 3 * d21 * e32 * gf * m2 - 6 * d21 * f31 * gf * m2 ** 2 - 6 * d21 * f32 * gf * m1 * m2 + 6 * d22 ** 2 * e11 * m1 - 6 * d22 ** 2 * e12 * m1 - 6 * d22 * e11 * e21 * m1 * m2 + 6 * d22 * e11 * e22 * m1 ** 2 + 6 * d22 * e12 * e21 * m1 * m2 - 6 * d22 * e12 * e22 * m1 ** 2 - 3 * d22 * e31 * gf * m1 - 3 * d22 * e32 * gf * m1 + 6 * d22 * f31 * gf * m1 * m2 + 6 * d22 * f32 * gf * m1 ** 2 - 6 * d31 * e11 * m2 * nf + 6 * d31 * e12 * m2 * nf + 6 * d31 * e21 * gf * m2 - 6 * d31 * e22 * gf * m2 - 6 * d32 * e11 * m1 * nf + 6 * d32 * e12 * m1 * nf + 6 * d32 * e21 * gf * m1 - 6 * d32 * e22 * gf * m1 + 3 * e11 * e31 * m2 ** 2 * nf + 3 * e11 * e32 * m1 ** 2 * nf - 3 * e12 * e31 * m2 ** 2 * nf - 3 * e12 * e32 * m1 ** 2 * nf - 3 * e21 * e31 * gf * m2 ** 2 - 3 * e21 * e32 * gf * m1 ** 2 + 3 * e22 * e31 * gf * m2 ** 2 + 3 * e22 * e32 * gf * m1 ** 2)
        d_ = (2 * d11 * d21 * d22 * m1 - 2 * d11 * d21 * d22 * m2 + 6 * d11 * d21 * e21 * m2 ** 2 - 2 * d11 * d21 * e22 * m1 * m2 - 4 * d11 * d21 * e22 * m2 ** 2 - 2 * d11 * d22 ** 2 * m1 + 2 * d11 * d22 ** 2 * m2 - 2 * d11 * d22 * e21 * m1 * m2 - 4 * d11 * d22 * e21 * m2 ** 2 - 2 * d11 * d22 * e22 * m1 ** 2 + 8 * d11 * d22 * e22 * m1 * m2 + 2 * d11 * d32 * m1 * nf - 2 * d11 * d32 * m2 * nf + 2 * d11 * e21 ** 2 * m2 ** 3 - 4 * d11 * e21 * e22 * m1 * m2 ** 2 + 2 * d11 * e22 ** 2 * m1 ** 2 * m2 + 3 * d11 * e31 * m2 ** 2 * nf - d11 * e32 * m1 ** 2 * nf + 4 * d11 * e32 * m1 * m2 * nf - 2 * d11 * f31 * m2 ** 3 * nf - 2 * d11 * f32 * m1 ** 2 * m2 * nf - 2 * d12 * d21 ** 2 * m1 + 2 * d12 * d21 ** 2 * m2 + 2 * d12 * d21 * d22 * m1 - 2 * d12 * d21 * d22 * m2 - 8 * d12 * d21 * e21 * m1 * m2 + 2 * d12 * d21 * e21 * m2 ** 2 + 4 * d12 * d21 * e22 * m1 ** 2 + 2 * d12 * d21 * e22 * m1 * m2 + 4 * d12 * d22 * e21 * m1 ** 2 + 2 * d12 * d22 * e21 * m1 * m2 - 6 * d12 * d22 * e22 * m1 ** 2 + 2 * d12 * d31 * m1 * nf - 2 * d12 * d31 * m2 * nf - 2 * d12 * e21 ** 2 * m1 * m2 ** 2 + 4 * d12 * e21 * e22 * m1 ** 2 * m2 - 2 * d12 * e22 ** 2 * m1 ** 3 - 4 * d12 * e31 * m1 * m2 * nf + d12 * e31 * m2 ** 2 * nf - 3 * d12 * e32 * m1 ** 2 * nf + 2 * d12 * f31 * m1 * m2 ** 2 * nf + 2 * d12 * f32 * m1 ** 3 * nf - 6 * d21 ** 2 * e11 * m2 ** 2 + 2 * d21 ** 2 * e12 * m1 * m2 + 4 * d21 ** 2 * e12 * m2 ** 2 + 10 * d21 * d22 * e11 * m1 * m2 + 2 * d21 * d22 * e11 * m2 ** 2 - 2 * d21 * d22 * e12 * m1 ** 2 - 10 * d21 * d22 * e12 * m1 * m2 - 2 * d21 * d32 * gf * m1 + 2 * d21 * d32 * gf * m2 - 2 * d21 * e11 * e21 * m2 ** 3 + 2 * d21 * e11 * e22 * m1 * m2 ** 2 + 2 * d21 * e12 * e21 * m1 * m2 ** 2 - 2 * d21 * e12 * e22 * m1 ** 2 * m2 - 3 * d21 * e31 * gf * m2 ** 2 + d21 * e32 * gf * m1 ** 2 - 4 * d21 * e32 * gf * m1 * m2 + 2 * d21 * f31 * gf * m2 ** 3 + 2 * d21 * f32 * gf * m1 ** 2 * m2 - 4 * d22 ** 2 * e11 * m1 ** 2 - 2 * d22 ** 2 * e11 * m1 * m2 + 6 * d22 ** 2 * e12 * m1 ** 2 - 2 * d22 * d31 * gf * m1 + 2 * d22 * d31 * gf * m2 + 2 * d22 * e11 * e21 * m1 * m2 ** 2 - 2 * d22 * e11 * e22 * m1 ** 2 * m2 - 2 * d22 * e12 * e21 * m1 ** 2 * m2 + 2 * d22 * e12 * e22 * m1 ** 3 + 4 * d22 * e31 * gf * m1 * m2 - d22 * e31 * gf * m2 ** 2 + 3 * d22 * e32 * gf * m1 ** 2 - 2 * d22 * f31 * gf * m1 * m2 ** 2 - 2 * d22 * f32 * gf * m1 ** 3 + 6 * d31 * e11 * m2 ** 2 * nf - 2 * d31 * e12 * m1 * m2 * nf - 4 * d31 * e12 * m2 ** 2 * nf - 6 * d31 * e21 * gf * m2 ** 2 + 2 * d31 * e22 * gf * m1 * m2 + 4 * d31 * e22 * gf * m2 ** 2 + 4 * d32 * e11 * m1 ** 2 * nf + 2 * d32 * e11 * m1 * m2 * nf - 6 * d32 * e12 * m1 ** 2 * nf - 4 * d32 * e21 * gf * m1 ** 2 - 2 * d32 * e21 * gf * m1 * m2 + 6 * d32 * e22 * gf * m1 ** 2 - e11 * e31 * m2 ** 3 * nf - e11 * e32 * m1 ** 2 * m2 * nf + e12 * e31 * m1 * m2 ** 2 * nf + e12 * e32 * m1 ** 3 * nf + e21 * e31 * gf * m2 ** 3 + e21 * e32 * gf * m1 ** 2 * m2 - e22 * e31 * gf * m1 * m2 ** 2 - e22 * e32 * gf * m1 ** 3)
        e_ = - 2 * d11 * d21 * d22 * m1 * m2 + 2 * d11 * d21 * d22 * m2 ** 2 - 2 * d11 * d21 * e21 * m2 ** 3 + 2 * d11 * d21 * e22 * m1 * m2 ** 2 + 2 * d11 * d22 ** 2 * m1 ** 2 - 2 * d11 * d22 ** 2 * m1 * m2 + 2 * d11 * d22 * e21 * m1 * m2 ** 2 - 2 * d11 * d22 * e22 * m1 ** 2 * m2 - 2 * d11 * d32 * m1 ** 2 * nf + 2 * d11 * d32 * m1 * m2 * nf - d11 * e31 * m2 ** 3 * nf - d11 * e32 * m1 ** 2 * m2 * nf + 2 * d12 * d21 ** 2 * m1 * m2 - 2 * d12 * d21 ** 2 * m2 ** 2 - 2 * d12 * d21 * d22 * m1 ** 2 + 2 * d12 * d21 * d22 * m1 * m2 + 2 * d12 * d21 * e21 * m1 * m2 ** 2 - 2 * d12 * d21 * e22 * m1 ** 2 * m2 - 2 * d12 * d22 * e21 * m1 ** 2 * m2 + 2 * d12 * d22 * e22 * m1 ** 3 - 2 * d12 * d31 * m1 * m2 * nf + 2 * d12 * d31 * m2 ** 2 * nf + d12 * e31 * m1 * m2 ** 2 * nf + d12 * e32 * m1 ** 3 * nf + 2 * d21 ** 2 * e11 * m2 ** 3 - 2 * d21 ** 2 * e12 * m1 * m2 ** 2 - 4 * d21 * d22 * e11 * m1 * m2 ** 2 + 4 * d21 * d22 * e12 * m1 ** 2 * m2 + 2 * d21 * d32 * gf * m1 ** 2 - 2 * d21 * d32 * gf * m1 * m2 + d21 * e31 * gf * m2 ** 3 + d21 * e32 * gf * m1 ** 2 * m2 + 2 * d22 ** 2 * e11 * m1 ** 2 * m2 - 2 * d22 ** 2 * e12 * m1 ** 3 + 2 * d22 * d31 * gf * m1 * m2 - 2 * d22 * d31 * gf * m2 ** 2 - d22 * e31 * gf * m1 * m2 ** 2 - d22 * e32 * gf * m1 ** 3 - 2 * d31 * e11 * m2 ** 3 * nf + 2 * d31 * e12 * m1 * m2 ** 2 * nf + 2 * d31 * e21 * gf * m2 ** 3 - 2 * d31 * e22 * gf * m1 * m2 ** 2 - 2 * d32 * e11 * m1 ** 2 * m2 * nf + 2 * d32 * e12 * m1 ** 3 * nf + 2 * d32 * e21 * gf * m1 ** 2 * m2 - 2 * d32 * e22 * gf * m1 ** 3
        #
        roots = np.zeros((self.n_samples, 4))
        for ind, (c4, c3, c2, c1, c0) in enumerate(zip(a_, b_, c_, d_, e_)):
            poly_roots = np.roots(np.array([c4, c3, c2, c1, c0]))
            if poly_roots.shape[0]==4:
                roots[ind] = np.roots(np.array([c4, c3, c2, c1, c0]))
        ok_roots = []
        poly_inds = []
        for i in range(4):
            is_sup = np.greater(roots[:, i], self.intervals[:-1])
            is_inf = np.less(roots[:, i], self.intervals[1:])
            avail_roots = np.where(is_inf*is_sup)[0]

            # for root, sup, inf, int_inf, int_sup, avail in zip(roots[:,i], is_sup, is_inf, self.intervals[:-1], self.intervals[1:], is_inf*is_sup):
            #     print("Root {} in inter [{}; {}], is {}".format(root, int_inf, int_sup, avail))
            # quit()
            ok_roots += list(roots[avail_roots, i])
            poly_inds+=list(avail_roots)
        ok_roots = np.array(ok_roots)
        self.ths = X[self.sorted_inds, 0]
        gg = (d11[poly_inds] - ok_roots * e11[poly_inds]) / (m1 - ok_roots) + (
                    d12[poly_inds] - ok_roots * e12[poly_inds]) / (ok_roots - m2)
        t = ((d21[poly_inds] - ok_roots * e21[poly_inds]) / (m1 - ok_roots) + (
                    d22[poly_inds] - ok_roots * e22[poly_inds]) / (ok_roots - m2))
        ng = (d31[poly_inds] + e31[poly_inds] * ok_roots + f31[poly_inds] * ok_roots ** 2) / (
                    m1 - ok_roots) ** 2 + (
                         d32[poly_inds] + ok_roots * e32[poly_inds] + ok_roots ** 2 * f32[poly_inds]) / (
                         ok_roots - m2) ** 2

        # ths = np.linspace(m2, m1)
        # preds = np.array([np.array([(X[i] - th) / m1 - th if
        #                                                 X[i] > th else (X[i] - th) / (
        #                               th - m2) for th in ths]) for i in range(X.shape[0])])[:,:,0]
        # gg = np.sum(preds*y, axis=0)/self.n_samples
        # t = np.sum(preds * base_vote, axis=0)/self.n_samples
        # ng = np.sum(preds * preds, axis=0)/self.n_samples
        # gg = (d11 - ths * e11) / (m1 - ths) + (
        #         d12 - ths * e12) / (ths - m2)
        # t = ((d21 - ths * e21) / (m1 - ths) + (
        #         d22 - ths * e22) / (ths - m2))
        # ng = (d31 + e31 * ths + f31 * ths ** 2) / (
        #              m1 - ths) ** 2 + (
        #              d32 + ths * e32 + ths ** 2 * f32) / (
        #              ok_roots - m2) ** 2
        # if it_ind < len(plif) and self.col_ind == plif[it_ind][1]:
        #     self.th = plif[it_ind][0]
        #     self.to_choose=True
        # else:
        #     self.th = 0
        #     self.to_choose=False
        # preds = np.array([(X[i] - self.th) / (m1 - self.th) if
        #                               X[i] > self.th else (X[i] - self.th) / (
        #             self.th - m2) for i in range(X.shape[0])])
        # gg = np.sum(preds[self.sorted_inds]*ord_y)
        # t = np.sum(preds[self.sorted_inds]*base_vote[self.sorted_inds])
        # ng = np.sum(preds[self.sorted_inds]*preds[self.sorted_inds])
        weights = (-4 * base_margin * gg * t + 4 * np.square(
            gg) * base_norm) / (
                          4 * base_margin * gg * ng - 4 * np.square(
                      gg) * t)
        cb2 = 1 - (gg * weights + base_margin / (self.n_samples)) ** 2 / (
                    base_norm / (
                self.n_samples) + 2 * weights * t + ng * weights ** 2)


        self.th = ok_roots[np.argmin(cb2)]
        self.m1 = m1
        self.m2 = m2
        self.no_th = False
        # if self.col_ind ==57:
        #     print("OK roots : {}, cb2 : {}, th: {}".format(ok_roots, cb2, round(self.th, 2)))
            # print(X[self.sorted_inds, :])
        self.cb = cb2[np.argmin(cb2)]
        self.weight = weights[np.argmin(cb2)]
        return self

    def predict(self, X):
        X = X[:, self.col_ind]
        if self.no_th:
            return np.transpose(np.zeros(self.n_samples))
        else:
            return np.sign(np.transpose(np.array([(X[i] - self.th) / (self.m1 - self.th) if
                                      X[i] > self.th else (X[i] - self.th) / (
                    self.th - self.m2) for i in range(X.shape[0])])))


class RandomStump(BaseEstimator, ClassifierMixin):

    def __init__(self, rs=42):
        self.rs = np.random.RandomState(rs)

    def fit(self, X, y):
        self.col_ind = self.rs.choice(np.arange(X.shape[1]), 1)[0]
        self.th = self.rs.uniform(np.min(self.col_ind), np.max(self.col_ind), 1)[0]
        # print(np.sum(np.array([(X[i, self.col_ind] - self.th) / abs(X[i, self.col_ind] - self.th) for i in range(X.shape[0])]) *y))
        if np.sum(np.array([(X[i, self.col_ind] - self.th) / abs(X[i, self.col_ind] - self.th) for i in range(X.shape[0])]) *y) < 0:
            self.reverse=-1
        else:
            self.reverse=1
        return self

    def predict(self, X):
        X = X[:, self.col_ind]
        return self.reverse*np.transpose(np.array([(X[i] - self.th) / abs(X[i] - self.th) for i in range(X.shape[0])]))





# Used for CBBoost

class SelfOptCBBoostClassifier(BaseMonoviewClassifier):
    def __init__(self, n_max_iterations=10, random_state=42, twice_the_same=True,
                 random_start=False, plotted_metric=zero_one_loss, save_train_data=True,
                 test_graph=True, base_estimator="BaseStump"):
        super(SelfOptCBBoostClassifier, self).__init__()
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
        self.twice_the_same = twice_the_same
        self.random_start = random_start
        self.plotted_metric = plotted_metric
        self.save_train_data = save_train_data
        self.test_graph = test_graph
        self.printed_args_name_list = ["n_max_iterations"
                                       "twice_the_same",
                                       "random_start",]
        self.param_names = []
        self.classed_params = []
        self.distribs = []
        self.weird_strings = {}
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.n_features = X.shape[1]
        formatted_X, formatted_y = self.format_X_y(X, y)

        self.init_info_containers()

        # Initialize the weak classifiers ensemble
        m, n = formatted_X.shape

        start = time.time()
        self.n_total_hypotheses_ = n
        self.n_total_examples = m

        # Initialize the majority vote
        self.init_boosting(formatted_X, formatted_y)

        self.break_cause = " the maximum number of iterations was attained."

        for k in range(self.n_max_iterations):# - 1 if self.n_max_iterations is not None else np.inf)):

            # Print dynamically the step and the error of the current classifier
            self.it = k+1

            # Find the best (weight, voter) couple.
            self._find_new_voter(formatted_X, formatted_y)

        end = time.time()
        self.train_time = end - start
        return self

    def predict_proba(self, X):
        start = time.time()
        check_is_fitted(self, 'weights_')
        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())

        votes = np.array([voter.predict(X) for voter in self.voters])
        vote  = np.average(votes, weights=self.weights_, axis=1)
        proba = np.array([np.array([(1 - vote_sample)/2, (1 + vote_sample)/2]) for vote_sample in vote])
        return proba

    def predict(self, X):
        return self._iter_predict(X, self.n_max_iterations)

    def _iter_predict(self, X, iter_index=1):
        start = time.time()
        check_is_fitted(self, 'weights_')
        if scipy.sparse.issparse(X):
            logging.warning('Converting sparse matrix to dense matrix.')
            X = np.array(X.todense())

        votes = np.array(
            [voter.predict(X) for voter in self.voters]).transpose()
        vote = np.sum(votes[:, :iter_index] * self.weights_[:iter_index], axis=1)

        signs_array = np.array([int(x) for x in sign(vote)])
        signs_array[signs_array == -1] = 0

        end = time.time()
        self.predict_time = end - start

        # Predict for each step of the boosting process

        return signs_array


    def init_boosting(self, X, y):
        """THis initialization corressponds to the first round of boosting with equal weights for each examples and the voter chosen by it's margin."""

        if self.random_start:
            voter = RandomStump().fit(X, y)
        else:
            voter = DecisionTreeClassifier(max_depth=1).fit(X, y)
        self.voters.append(voter)

        self.previous_vote = voter.predict(X).astype(np.float64)
        self.q = 1
        self.weights_.append(self.q)


    def format_X_y(self, X, y):
        """Formats the data  : X -the examples- and y -the labels- to be used properly by the algorithm """
        if scipy.sparse.issparse(X):
            logging.info('Converting to dense matrix.')
            X = np.array(X.todense())
        # Initialization
        y_neg = change_label_to_minus(y)
        y_neg = y_neg.reshape((y.shape[0], 1))
        return X, y_neg


    def init_info_containers(self):
        """Initialize the containers that will be collected at each iteration for the analysis"""
        self.weights_ = []
        self.voters = []
        self.chosen_features = []
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




    def _find_new_voter(self, X, y):
        """Here, we solve the two_voters_mincq_problem for each potential new voter,
        and select the one that has the smallest minimum"""

        m, n = X.shape
        prop_cols = np.zeros((m, n*2))
        possible_clfs = []
        for col_ind in range(n):
            if self.base_estimator == "BaseStump":
                clf = CboundStumpBuilder(col_ind=col_ind)
            elif self.base_estimator == "PseudoLinearStump":
                clf = CboundPseudoStumpBuilder(col_ind=col_ind)
            elif self.base_estimator == "LinearStump":
                clf = CBoundThresholdFinder(col_ind=col_ind)
            else:
                raise AttributeError("Wrong base estimator.")
            clf.fit(X, y, self.previous_vote.reshape((m, 1)), it_ind=self.it)
            prop_cols[:, col_ind] = clf.predict(X)
            prop_cols[:, col_ind+n] = -clf.predict(X)
            possible_clfs.append(clf)
        margins = np.sum(prop_cols * y, axis=0)
        # print(margins)
        norms = np.sum(np.square(prop_cols), axis=0)
        base_margin = np.sum(self.previous_vote.reshape((m, 1))*y, axis=0)

        tau = np.sum(prop_cols * self.previous_vote.reshape((m, 1)), axis=0)

        base_norm = np.sum(np.square(self.previous_vote))

        weights = (-4 * base_margin * margins * tau + 4 * np.square(
            margins) * base_norm) / (
                              4 * base_margin * margins * norms - 4 * np.square(
                          margins) * tau)
        print(weights, [clf.weight for clf in possible_clfs])
        # print(weights)
        cbs = 1 - (1/(m))*(margins * weights + base_margin )**2/(base_norm + 2*weights*tau + norms*(weights**2))
        # print(cbs)
        cbs[np.isnan(cbs)] = np.inf
        cbs[np.isnan(weights)] = np.inf
        # quit()
        best_ind = np.argmin(cbs)
        # print(possible_clfs[best_ind].to_choose)
        # print(best_ind)
        if best_ind<n:
            self.weights_.append(weights[best_ind])
            self.voters.append(possible_clfs[best_ind])
        else:
            self.weights_.append(weights[best_ind])
            self.voters.append(possible_clfs[best_ind-n])
            print(weights[best_ind-n], weights[best_ind])
        # print([plif for plif in zip(y, prop_cols[:, best_ind])])
        # print(weights[best_ind])
        # print("Best CB : {} for col : {}, weighted : {}, with th : {}".format(round(cbs[best_ind], 2), best_ind,round(weights[best_ind], 2),  round(possible_clfs[best_ind].th,2)))
        self.previous_vote += weights[best_ind]*prop_cols[:, best_ind]
