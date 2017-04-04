from .covar_base import Covariance
from limix.hcache import cached
import numpy as np

# TODO can we do import LowRankCov ?
import lowrank
import freeform


class Categorical(Covariance):
    """
    Categorical covariance.
    cov_{i,j} depends on the category of i and j only

    Main members of the class:
        - categories: a vector of length the dimension of the cov matrix with the
        categories to which every sample belongs
        - cat_cov, a covariance matrix between categories, whose dimensions are determined
        by the number of categories -> can be free form or low rank
    """
    def __init__(self, categories, rank=None, cats_star=None):
        Covariance.__init__(self)

        self.cats = categories
        self.unique_cats = np.unique(categories)
        self.n_cats = len(self.unique_cats)
        self.rank = rank
        self.dim = len(self.cats)

        # TODO do this cleanly with setters and properties (including the initialise_function)
        # initialisation predictions
        self.cats_star = cats_star
        if self.cats_star is None:
            self._use_to_predict = False
        else:
            self.initialize_cats_star()
            self._use_to_predict = True

        # initialise covariance matrix between categories
        self.initialize_cov()
        # build a categories_num vector where categories are integers from 0 to number of categories
        self.initialize_cats()

    def initialize_cov(self):
        if self.rank is None:
            self.cat_cov = freeform.FreeFormCov(self.n_cats)
        else:
            self.cat_cov = lowrank.LowRankCov(self.n_cats, self.rank)

    def initialize_cats(self):
        self.i_cats = np.zeros(len(self.cats))
        for i in range(self.n_cats):
            self.i_cats += (self.cats == self.unique_cats[i])*i

    def initialize_cats_star(self):
        # check that all categories in cats_star are also found in cats
        cats_star_uq = np.unique(self.cats_star)
        assert all(np.in1d(cats_star_uq, self.unique_cats)), 'all categories used for prediction must be seen during training'

        # build the int category vector
        self.i_cats_star = np.zeros(len(self.cats_star))
        for i in range(self.n_cats):
            self.i_cats_star += (self.cats_star == self.unique_cats[i])*i

    # #####################
    # # properties
    # #####################
    # @property
    # def cat_star(self):
    #     return self.cat_star
    #
    # #####################
    # # Setters
    # #####################
    # @cats_star.setter
    # def cats_star(self, value):
    #     if value is None:
    #         self._use_to_predict = False
    #     else:
    #         self._use_to_predict = True
    #     self.cats_star = value

    #####################
    # Params handling
    #####################
    def setParams(self, params):
        self.cat_cov.setParams(params)

    def getParams(self):
        return self.cat_cov.getParams()

    def getNumberParams(self):
        return self.cat_cov.getNumberParams()

    #####################
    # cov and gradient
    # DO NOT CACHE here (cached in member cat_cov)
    #####################
    def K(self):
        R = self.cat_cov.K()
        return self.expand(R)

    def K_grad_i(self, i):
        R = self.cat_cov.K_grad_i(i)
        return self.expand(R)

    def K_star(self):
        R = self.cat_cov.K()
        return self.expand_star(R)

    # TODO: hessian ? not implemented for lowrank

    #####################
    # Expanding the category * category matrices
    #####################
    # for the covariance or its gradient
    def expand(self, mat):
        R = np.zeros([self.dim, self.dim])
        for i in range(self.n_cats):
            for j in range(self.n_cats):
                tmp_i = (self.i_cats == i)[:, None]
                tmp_j = (self.i_cats == j)[None, :]
                R += tmp_i.dot(tmp_j) * mat[i,j]
        return R

    # for the cross covariance
    def expand_star(self, mat):
        n_star = len(self.cats_star)
        R = np.zeros([n_star, self.dim])
        for i in range(self.n_cats):
            for j in range(self.n_cats):
                tmp_i = (self.i_cats_star == i)[:, None]
                tmp_j = (self.i_cats == j)[None, :]
                R += tmp_i.dot(tmp_j) * mat[i,j]
        return R
