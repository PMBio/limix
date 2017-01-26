__author__ = 'damienarnol1'

import pdb

import numpy as np

from limix.core.covar import SQExpCov
from limix.core.covar import ZKZCov
from limix.core.covar import FixedCov
from limix.core.covar import SumCov
from limix.core.gp import GP
from limix.core.mean import MeanBase
from limix.utils.preprocess import covar_rescaling_factor


def normalise(mat, axis=1):
    mat -= np.reshape(mat.mean(axis=axis), [mat.shape[0], 1])
    mat /= np.reshape(mat.std(axis=axis), [mat.shape[0], 1])
    mat /= np.sqrt(mat.shape[1])
    return mat

rm_diag = True

data_directory = '/Users/damienarnol1/Documents/local/pro/PhD/data/Zurich_data/output/tmp/'

X_file = data_directory + '/cell_locations_1'
phenotype_file = data_directory+'/membrane_1'

# read data
X = np.genfromtxt(X_file, delimiter=',', skiprows=1)
phenotypes = np.genfromtxt(phenotype_file, delimiter=' ', skiprows=1)
phenotypes = np.delete(phenotypes, 0, axis=1)
phenotypes = normalise(phenotypes)
N_cells = phenotypes.shape[0]

Kinship = phenotypes.dot(phenotypes.transpose())
Kinship -= np.linalg.eigvalsh(Kinship).min() * np.eye(N_cells)
Kinship *= covar_rescaling_factor(Kinship)

phenotype = phenotypes[:, 5]
phenotype -= phenotype.mean()
phenotype /= phenotype.std()
phenotype = np.reshape(phenotype, [N_cells, 1])

# create different models and print the result including likelihood
# create all the covariance terms
direct_cov = FixedCov(Kinship)

# noise
noise_cov = FixedCov(np.eye(N_cells))

# local_noise
local_noise_cov = SQExpCov(X)

# environment effect
environment_cov = ZKZCov(X, Kinship, rm_diag)

# mean term
mean = MeanBase(phenotype)

#######################################################################
# MODEL 1: all
#######################################################################
print('...........................................................')
print('model 1 : complete model ')
print('...........................................................')
# total cov and mean
cov = SumCov(direct_cov, noise_cov)
cov = SumCov(cov, local_noise_cov)
cov = SumCov(cov, environment_cov)

# fixing length scale of ZKZ and SE
environment_cov.length = N_cells / 50
environment_cov.act_length = False

# local_noise_cov.length = N_cells/10.0
# local_noise_cov.act_length = False

# define and optimise GP
gp = GP(covar=cov, mean=mean)
gp.optimize()

# show results
print("inferred parameters ")
print("direct_scale = ", " ", direct_cov.scale)
print("noise_scale = ", " ", noise_cov.scale)
print("local_noise_scale = ", " ", local_noise_cov.scale)
print("local_noise_length = ", " ", local_noise_cov.length)
print("environment_scale = ", " ", environment_cov.scale)
print("environment_length = ", " ", environment_cov.length)

#######################################################################
# MODEL 2: no social effect
#######################################################################
print('...........................................................')
print('model 2: no social effect')
print('...........................................................')

direct_cov2 = FixedCov(Kinship)

# noise
noise_cov2 = FixedCov(np.eye(N_cells))

# local_noise
local_noise_cov2 = SQExpCov(X)

# total cov and mean
cov2 = SumCov(direct_cov2, noise_cov2)
cov2 = SumCov(cov2, local_noise_cov2)


# define and optimise GP
gp2 = GP(covar=cov2, mean=mean)
gp2.optimize()

# show results
print("inferred parameters ")
print("direct_scale = ", " ", direct_cov2.scale)
print("noise_scale = ", " ", noise_cov2.scale)
print("local_noise_scale = ", " ", local_noise_cov2.scale)
print("local_noise_length = ", " ", local_noise_cov2.length)

#######################################################################
# MODEL 3: no local moise no social effect
#######################################################################
print('...........................................................')
print('model 3: no local noise, no social effect')
print('...........................................................')
direct_cov3 = FixedCov(Kinship)

# noise
noise_cov3 = FixedCov(np.eye(N_cells))

# total cov and mean
cov3 = SumCov(direct_cov3, noise_cov3)


# define and optimise GP
gp3 = GP(covar=cov3, mean=mean)
gp3.optimize()

# show results
print("inferred parameters ")
print("direct_scale = ", " ", direct_cov3.scale)
print("noise_scale = ", " ", noise_cov3.scale)

#######################################################################
# MODEL 4: no local moise but social effect ON
#######################################################################
print('...........................................................')
print('model 3: no local noise, social effects ON')
print('...........................................................')
direct_cov4 = FixedCov(Kinship)

# noise
noise_cov4 = FixedCov(np.eye(N_cells))

# environment effect
environment_cov4 = ZKZCov(X, Kinship, rm_diag)

cov4 = SumCov(direct_cov4, noise_cov4)
cov4 = SumCov(cov4, environment_cov4)

# define and optimise GP
gp4 = GP(covar=cov4, mean=mean)
gp4.optimize()

# show results
print("inferred parameters ")
print("direct_scale = ", " ", direct_cov4.scale)
print("noise_scale = ", " ", noise_cov4.scale)
print("environment_scale = ", " ", environment_cov4.scale)
print("environment_length = ", " ", environment_cov4.length)