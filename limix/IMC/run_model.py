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
output_dir = '/Users/damienarnol1/Documents/local/pro/PhD/social_effects/real_data/random_4/'

X_file = data_directory + '/all_positions'
phenotype_file = data_directory+'/membrane'

# read data
all_X = np.genfromtxt(X_file, delimiter=',', skiprows=1)
all_phenotypes = np.genfromtxt(phenotype_file, delimiter=' ', skiprows=1)

permutations = True


for image in range(0, 5):
    phenotypes = all_phenotypes[all_phenotypes[:, 0] == image, :]
    phenotypes = np.delete(phenotypes, 0, axis=1)
    phenotypes = normalise(phenotypes)

    X = all_X[all_X[:, 0] == image, :]
    X = np.delete(X, 0, axis=1)

    if permutations:
        X = X[np.random.permutation(X.shape[0]), :]

    if X.shape[0] != phenotypes.shape[0]:
        raise Exception('cell number inconsistent between position and epression levels ')

    N_cells = phenotypes.shape[0]

    parameters = np.zeros([phenotypes.shape[1], 6])

    output_file = output_dir + '_' + str(image)

    for phen in range(0, phenotypes.shape[1]):

        phenotype = phenotypes[:, phen]
        phenotype -= phenotype.mean()
        phenotype /= phenotype.std()
        phenotype = np.reshape(phenotype, [N_cells, 1])

        phenotypes_tmp = np.delete(phenotypes, phen, axis=1)

        Kinship = phenotypes_tmp.dot(phenotypes_tmp.transpose())
        Kinship -= np.linalg.eigvalsh(Kinship).min() * np.eye(N_cells)
        Kinship *= covar_rescaling_factor(Kinship)

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
        # direct_cov.scale = 0

        # local_noise_cov.scale = 0
        # direct_cov.scale = 0
        # cov = SumCov(noise_cov, local_noise_cov)

        cov = SumCov(cov, environment_cov)

        # fixing length scale of ZKZ and SE
        environment_cov.length = N_cells / 50
        # environment_cov.scale=0
        # environment_cov.act_length = False

        # local_noise_cov.length = N_cells/10.0
        # local_noise_cov.act_length = False

        # define and optimise GP
        gp = GP(covar=cov, mean=mean)

        try:
            gp.optimize()
        except:
            print('optimisation', str(phen), 'failed')
            continue

        # rescale each terms to sample variance one
        # direct cov: unnecessary as fixed covariance rescaled before optimisation
        # local noise covariance
        tmp = covar_rescaling_factor(local_noise_cov.K()/local_noise_cov.scale)
        local_noise_cov.scale /= tmp
        # env effect
        tmp = covar_rescaling_factor(environment_cov.K()/environment_cov.scale**2)
        environment_cov.scale = environment_cov.scale**2/tmp


        # show results
        print("inferred parameters ")
        print("direct_scale = ", " ", direct_cov.scale)
        print("noise_scale = ", " ", noise_cov.scale)
        print("local_noise_scale = ", " ", local_noise_cov.scale)
        print("local_noise_length = ", " ", local_noise_cov.length)
        print("environment_scale = ", " ", environment_cov.scale)
        print("environment_length = ", " ", environment_cov.length)

        parameters[phen, :] = [direct_cov.scale,
                               noise_cov.scale,
                               local_noise_cov.scale,
                               local_noise_cov.length,
                               environment_cov.scale,
                               environment_cov.length]

    result_header = 'direct_scale' + ' ' + \
                    'noise_scale' + ' ' + \
                    'local_noise_scale' + ' ' + \
                    'local_noise_length' + ' ' + \
                    'environment_scale' + ' ' + \
                    'environment_length'

    np.savetxt(output_file, parameters, delimiter=' ', header=result_header)


