import unittest
import numpy as np
from limix.core.covar.zkz import ZKZCov
from limix.utils.check_grad import mcheck_grad
import scipy as sp

class TestZKZ(unittest.TestCase):
    def setUp(self):
        np.random.seed()
        print '\n\n\n'
        print np.random.randn(1)
        print '\n\n\n'
        self._X = np.random.randn(10, 5)
        tmp = np.random.randn(10, 20)
        self.Kinship = tmp.dot(tmp.transpose())
        self._cov = ZKZCov(self._X, self.Kinship, remove_diag=True)

    def test_Kgrad(self):

        def func(x, i):
            self._cov.scale = x[i]
            return self._cov.K()

        def grad(x, i):
            self._cov.scale = x[i]
            return self._cov.K_grad_i(0)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)

        np.testing.assert_almost_equal(err, 0, decimal=5)

        def func(x, i):
            self._cov.length = x[i]
            return self._cov.K()

        def grad(x, i):
            self._cov.scale = x[i]
            return self._cov.K_grad_i(1)

        x0 = np.array([self._cov.scale])
        err = mcheck_grad(func, grad, x0)


if __name__ == '__main__':
    unittest.main()
