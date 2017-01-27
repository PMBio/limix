import sys
from limix.hcache import cached
import scipy as sp
import numpy as np
from .covar_base import Covariance
import scipy.spatial as SS
from . import sqexp


# TODO would be cleaner to inherit from squared exponential ?
class ZKZCov(Covariance):
    """
    squared exponential covariance function
    """
    def __init__(self,X, Kin, remove_diag=True):
        """
        X   dim x d input matrix
        """
        super(ZKZCov, self).__init__()
        self._X = None
        self.se = sqexp.SQExpCov(X)
        self.X = X
        self.Kin = Kin
        self.rm_diag = remove_diag
        # self._initParams()

    def get_input_dim(self):
        return self.X.shape[1]

    #####################
    # Properties
    #####################
    @property
    def scale(self):
        return self.se.scale

    @property
    def length(self):
        return self.se.length

    # TODO not true in the general case -> to change
    @property
    def dim(self):
        return self.se.dim

    @property
    def scale_ste(self):
        # if self.getFIinv() is None:
        #     R = None
        # else:
        #     R = sp.sqrt(self.getFIinv()[0,0])
        # return R
        return self.se.scale_ste

    @property
    def length_ste(self):
        # if self.getFIinv() is None:
        #     R = None
        # else:
        #     R = sp.sqrt(self.getFIinv()[1,1])
        # return R
        return self.se.length_ste

    @property
    def X(self):
        return self._X

    #####################
    # Setters
    #####################
    @scale.setter
    def scale(self, value):
        self.se.scale = value
        self.clear_all()
        self._notify()

    @length.setter
    def length(self, value):
        self.se.length = value
        self.clear_all()
        self._notify()

    @X.setter
    def X(self,value):
        self.se.X = value

    @Covariance.use_to_predict.setter
    def use_to_predict(self, value):
        assert self.Xstar is not None, 'set Xstar!'
        self._use_to_predict = value
        self._notify()

    #####################
    # Params handling
    #####################
    def getNumberParams(self):
        return self.se.getNumberParams()

    def setParams(self, params):
        self.se.setParams(params)
        self.clear_all()

    def getParams(self):
        return self.se.getParams()

    def _calcNumberParams(self):
        self.n_params = 2

    #####################
    # Activation handling
    #####################

    @property
    def act_scale(self):
        return self.se._scale_act

    @act_scale.setter
    def act_scale(self, act):
        self.se._scale_act = bool(act)
        self._notify()

    @property
    def act_length(self):
        return self.se._length_act

    @act_length.setter
    def act_length(self, act):
        self.se._length_act = bool(act)
        self._notify()

    #####################
    # Cached
    #####################
    @cached('covar_base')
    def K(self):
        z = self.se.K()
        if self.rm_diag:
            z -= np.eye(z.shape[0]) * z.diagonal()
        tmp = z.dot(self.Kin.dot(z.transpose()))
        return tmp

    @cached('covar_base')
    def K_grad_i(self, i):
        grad_tmp = self.se.K_grad_i(i)
        se_K_tmp = self.se.K()
        if self.rm_diag:
            grad_tmp -= grad_tmp.diagonal() * np.eye(grad_tmp.shape[0])
            se_K_tmp -= se_K_tmp.diagonal() * np.eye(se_K_tmp.shape[0])
        r = grad_tmp.dot(self.Kin.dot(se_K_tmp.transpose()))
        r += se_K_tmp.dot(self.Kin.dot(grad_tmp.transpose()))
        return r

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return sp.array([self.se.scale, self.se.length])

    # def K_grad_interParam_i(self,i):
    #     if i==0:
    #         r = sp.exp(-self.E()/(2*self.se.length))
    #     else:
    #         A = sp.exp(-self.E()/(2*self.se.length))*self.se.E()
    #         r = self.se.scale * A / (2*self.se.length**2)
    #     return r
