import sys
from limix.hcache import cached
import scipy as sp
import numpy as np
from .covar_base import Covariance
import sqexp


class ZKZCov(Covariance):
    """
    squared exponential covariance function
    """
    def __init__(self, X, Kin, remove_diag=True, interaction_matrix=None, Xstar=None):
        """
        X   dim x d input matrix
        """

        super(ZKZCov, self).__init__()
        self.se = sqexp.SQExpCov(X)
        self.X = X
        self.Kin = Kin
        self.rm_diag = remove_diag
        self.interaction_matrix = interaction_matrix

        self.Xstar = Xstar

        self.penalty_function = None

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
        return self.K().shape[0]
        # return self.se.dim

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

    @property
    def Xstar(self):
        return self._Xstar

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

    # TODO: clear cash, notify etc ?
    @X.setter
    def X(self,value):
        self._X = value
        self.se.X = value

    @Xstar.setter
    def Xstar(self,value):
        # two ways of doing prediction:
        #   - 1 using the test set as an environment with unknown phenotype
        #   - 2 using the test set as an unknown environment and phenotype

        # case 1: X star is a list of boolean, whose value is True for the cells to use for validation
        if value is None:
            self._use_to_predict = False
            self._Xstar = None
            return
        else:
            self._use_to_predict = True

        if value.dtype == bool:
            assert len(value) == self.X.shape[0], 'boolean Xstar must be of length n_cells'
            assert self.Kin.shape[0] == len(value), 'Kin must contain all the cells, validation set included'

        # Case 2: Xstar is a list of positions to use for validation, expression profile X of these positions unknown
        if value.dtype == float:
            self.se.Xstar = value

        self._Xstar = value

    @Covariance.use_to_predict.setter
    def use_to_predict(self, value):
        if value:
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

    # TODO decide how to set this. function or parameters of the function ?
    def setPenalty(self, mu, sigma):
        self.penalty_function = np.zeros(2)
        self.penalty_function[0] = mu
        self.penalty_function[1] = sigma

        # making initialisation consistent with prior
        self.length = mu
        pass

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
        K = self._K()

        # if a test set Xstar is given as a boolean, return only the training set * training set covariance
        if self.Xstar is not None and self.Xstar.dtype == bool:
            return K[~self.Xstar, :][:, ~self.Xstar]

        return K

    # TODO cash ?
    def Kcross(self):
        assert self.Xstar is not None, "Provide test set"

        # if Xstar is a list of positions (test set does not contribute to environment)
        if self.Xstar.dtype == float:
            Zstar = self.se.Kcross()
            ZT = self.se.K().transpose()
            return Zstar.dot(self.Kin.dot(ZT))

        # if Xstar is a list of bool (test set contributes to environment)
        if self.Xstar.dtype == bool:
            # subset the K matrix
            K = self._K()
            return K[self.Xstar, :][:, ~self.Xstar]

    @cached('covar_base')
    def K_grad_i(self, i):
        grad_tmp = self.se.K_grad_i(i)
        se_K_tmp = self.se.K()
        if self.rm_diag:
            grad_tmp -= grad_tmp.diagonal() * np.eye(grad_tmp.shape[0])
            se_K_tmp -= se_K_tmp.diagonal() * np.eye(se_K_tmp.shape[0])
        if self.interaction_matrix is not None:
            grad_tmp *= self.interaction_matrix
            se_K_tmp *= self.interaction_matrix
        r = grad_tmp.dot(self.Kin.dot(se_K_tmp.transpose()))
        r += se_K_tmp.dot(self.Kin.dot(grad_tmp.transpose()))

        # if a test set Xstar is given as a boolean, return only the training set * training set covariance
        if self.Xstar is not None and self.Xstar.dtype == bool:
            return r[~self.Xstar, :][:, ~self.Xstar]
        return r

    @cached('covar_base')
    def _K(self):
        z = self.se.K()
        if self.interaction_matrix is not None:
            z *= self.interaction_matrix
        if self.rm_diag:
            z -= np.eye(z.shape[0]) * z.diagonal()
        tmp = z.dot(self.Kin.dot(z.transpose()))
        return tmp

    @cached('covar_base')
    def penalty(self):
        if self.penalty_function is None:
            return 0
        else:
            return (1/(2*self.penalty_function[1]**2.0)) * (self.length - self.penalty_function[0])**2.0

    @cached('covar_base')
    def penalty_grad(self, i):
        if self.penalty_function is None:
            return 0
        elif i == 0:
            return 0
        elif i == 1:
            # print 'length zkz: '+str(self.length)
            return 2.0*((1/(2*self.penalty_function[1]**2.0)) * (self.length - self.penalty_function[0])) * self.length
        else:
            raise Exception('Index out of range in penalty gradient')

    ####################
    # Interpretable Params
    ####################
    def getInterParams(self):
        return self.se.getInterParams()

    # def K_grad_interParam_i(self,i):
    #     if i==0:
    #         r = sp.exp(-self.E()/(2*self.se.length))
    #     else:
    #         A = sp.exp(-self.E()/(2*self.se.length))*self.se.E()
    #         r = self.se.scale * A / (2*self.se.length**2)
    #     return r
