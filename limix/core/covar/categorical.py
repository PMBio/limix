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
    def __init__(self, categories, rank=None):
        Covariance.__init__(self)

        self.categories = categories
        self.unique_categories = np.unique(categories)
        self.n_categories = len(self.unique_categories)
        self.rank = rank
        self.dim = len(self.categories)

        # initialise covariance matrix between categories
        self.initialize_cov()
        # build a categories_num vector where categories are integers from 0 to number of categories
        self.initialize_cats()

    def initialize_cov(self):
        if self.rank is None:
            self.cat_cov = freeform.FreeFormCov(self.n_categories)
        else:
            self.cat_cov = lowrank.LowRankCov(self.n_categories, self.rank)

    def initialize_cats(self):
        self.i_categories = np.zeros(len(self.categories))
        for i in range(self.n_categories):
            self.i_categories += (self.categories == self.unique_categories[i])*i

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
    #####################
    def K(self):
        R = self.cat_cov.K()
        return self.expand(R)

    def K_grad_i(self, i):
        R = self.cat_cov.K_grad_i(i)
        return self.expand(R)

    # TODO: hessian ? not implemented for lowrank

    #####################
    # Expanding the category * category matrices
    # DO NOT CACHE here (cached in member cat_cov)
    #####################
    def expand(self, mat):
        R = np.zeros([self.dim, self.dim])
        for i in range(self.n_categories):
            for j in range(self.n_categories):
                tmp_i = (self.i_categories == i)[:, None]
                tmp_j = (self.i_categories == j)[None, :]
                R += tmp_i.dot(tmp_j) * mat[i,j]
        return R
