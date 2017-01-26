__author__ = 'damienarnol1'

import numpy as np
import h5py
import scipy as sp
from limix.utils.preprocess import covar_rescaling_factor


def make_sure_reasonable_conditioning(S):
    max_cond = 1e1
    cond = S.max() / S.min()
    if cond > max_cond:
        eps = (max_cond * S.min() - S.max()) / (1 - max_cond)
        # logger = logging.getLogger(__name__)
        # logger.warn("The covariance matrix's conditioning number" +
        #             " is too high: %e. Summing %e to its eigenvalues and " +
        #             "renormalizing for a better conditioning number.",
        #             cond, eps)
        m = S.mean()
        S += eps
        S *= m / S.mean()
        print('S changed')
    else:
        print("S unchanged ")
    return S


def _generate_SE(N_cells, X=None, off=False):
    if X is None:
        X = np.random.uniform(0, N_cells * 2, [N_cells, 2])
    distance = np.empty([N_cells, N_cells])

    scale = 0 if off else np.random.uniform(0.5, 4, 1)
    length = np.random.uniform(N_cells / 7.0, N_cells / 6.0, 1)

    for i in range(0, N_cells):
        for j in range(0, N_cells):
            tmp = X[i, :] - X[j, :]
            distance[i, j] = tmp[0] ** 2 + tmp[1] ** 2

    cov = scale * sp.exp(-distance / (2 * length))
    # cov *= covar_rescaling_factor(cov)
    return cov, X, scale, length


def normalise_columns(mat):
    mat -= mat.mean(axis=0)
    mat /= mat.std(axis=0)
    mat /= np.sqrt(mat.shape[1])
    return mat


def generate_kinship(N_cells):
    tmp = np.random.randn(N_cells, 2 * N_cells)
    # tmp = normalise_columns(tmp)
    K = tmp.dot(tmp.transpose())
    K *= covar_rescaling_factor(K)
    return K


def generate_zkz(N_cells, rm_diag=False, Kinship=None, X=None, off=False):
    se_all = _generate_SE(N_cells, X=X, off=off)
    se = se_all[0]
    if rm_diag:
        se -= np.eye(se.shape[0]) * se.diagonal()
    se_all = np.delete(se_all, 0)
    if Kinship is None:
        Kinship = generate_kinship(N_cells)
    zkz = se.dot(Kinship.dot(se.transpose()))
    return zkz, se_all, Kinship


def generate_complete_model(N_cells, X=None, Kinship=None,  switch_off=[], rm_diagonal=False):

    # direct effect term
    direct_scale = 0 if 'direct' in switch_off else np.random.uniform(0.5, 4, 1)
    if Kinship is None:
        Kinship = generate_kinship(N_cells)
    direct_cov = direct_scale * Kinship

    # social/environment effect term
    off = 'social' in switch_off
    zkz, social_params, Kinship = generate_zkz(N_cells, rm_diag=rm_diagonal, Kinship=Kinship, X=X, off=off)

    # global noise
    noise_scale = 0 if 'noise' in switch_off else np.random.uniform(0.5, 4, 1)
    noise = noise_scale * np.eye(N_cells)

    # local noise
    off = 'local' in switch_off
    local_noise_cov, X, local_noise_scale, local_noise_length = _generate_SE(N_cells,
                                                                             X=social_params[0],
                                                                             off=off)

    # total cov and mean
    cov = local_noise_cov + noise + direct_cov + zkz
    mean = np.array([0] * N_cells)

    data = np.random.multivariate_normal(mean, cov, 1).transpose()

    return data, {'direct_scale': direct_scale,
                  'noise_scale': noise_scale,
                  'local_noise_scale': local_noise_scale, 'local_noise_length': local_noise_length,
                  'social_scale': social_params[1], 'social_length': social_params[2],
                  'Kinship': Kinship, 'X': X}


def generate_simple(N_cells, rm_zkz_diag=False):
    # one ZKZ matrix
    zkz, se_params, Kinship = generate_zkz(N_cells, rm_diag=rm_zkz_diag)

    # one noise matrix
    sigma_noise = np.random.uniform(0.5, 1.5, 1)
    noise = sigma_noise * np.eye(N_cells)

    # grouping parameters
    parameters = {'X': se_params[0], 'scale': se_params[1], 'length': se_params[2], 'Kinship': Kinship,
                  'noise': sigma_noise}

    # generating data
    covar = zkz + noise
    mean = np.array([0] * N_cells)
    data = np.random.multivariate_normal(mean, covar, 1).transpose()

    return data, parameters


def generate_samples(N_cells):
    # Number of cells
    N_cells = 500

    # model parameters
    sigma_I = 2
    sigma_E = 4
    sigma_eps = 1
    sigma_epsLocal = 1
    length_E = 100
    length_eps = 200

    # covariances structure
    # Kinship
    # TODO come up with a less random kinship matrix
    tmp = np.random.randn(N_cells, 2.0 * N_cells)
    Kinship = tmp.dot(tmp.T)
    Kinship = Kinship / np.mean(Kinship.diagonal())
    # Kinship *= covar_rescaling_factor(Kinship)

    # Z matrix
    X = np.random.uniform(0, N_cells * 2, [N_cells, 2])
    distance = np.empty([N_cells, N_cells])
    for i in range(0, N_cells):
        for j in range(0, N_cells):
            tmp = X[i, :] - X[j, :]
            distance[i, j] = tmp[0] ** 2 + tmp[1] ** 2

    Z_E = sigma_E ** 2 * sp.exp(-distance / 2 * length_E)
    Z_eps = sigma_eps ** 2 * sp.exp(-distance / 2 * length_eps)

    # total covariance
    covar = sigma_I ** 2 * Kinship + \
            Z_E.dot(Kinship.dot(Z_E.transpose())) + \
            Z_eps + \
            sigma_eps ** 2 * np.eye(N_cells)

    # mean
    mean = np.array([0] * N_cells)

    # generate random phenotype
    phenotype = np.random.multivariate_normal(mean, covar, 2)
    phenotype = phenotype.transpose()

    return Kinship, phenotype
