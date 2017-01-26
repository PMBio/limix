__author__ = 'damienarnol1'

import numpy as np

from limix.core.covar import SQExpCov
from limix.core.covar import ZKZCov
from limix.core.covar import FixedCov
from limix.core.covar import SumCov
from limix.core.gp import GP
from limix.core.mean import MeanBase
from limix.utils.preprocess import covar_rescaling_factor
import sys


def normalise(mat, axis=1):
    mat -= np.reshape(mat.mean(axis=axis), [mat.shape[0], 1])
    mat /= np.reshape(mat.std(axis=axis), [mat.shape[0], 1])
    mat /= np.sqrt(mat.shape[1])
    return mat


def compute_pairwise_interaction(model, expression_file, position_file, output_file,
                                 protein_1, protein_2):
    rm_diag = True

    if model is not 'full' and model is not 'env':
        raise Exception('model not understood. Please specify a model between full and env')

    # read phenotypes data
    with open(expression_file, 'r') as f:
        prot_tmp = f.readline()
    protein_names = prot_tmp.split(' ')
    protein_names[-1] = protein_names[-1][0:-1]  # removing the newline sign at the end of the last protein
    protein_names = np.reshape(protein_names, [len(protein_names), 1])
    phenotypes = np.loadtxt(expression_file, delimiter=' ', skiprows=1)

    # read position data
    X = np.genfromtxt(position_file, delimiter=',')
    if X.shape[0] != phenotypes.shape[0]:
        raise Exception('cell number inconsistent between position and epression levels ')

    # select phenotype and environmental protein
    phen = [i for i in range(0, len(protein_names)) if protein_names[i] == protein_1]
    env_prot_index = [i for i in range(0, len(protein_names)) if protein_names[i] == protein_2]

    N_cells = phenotypes.shape[0]

    # parameters = np.zeros([phenotypes.shape[1], 6])

    # log_lik = np.zeros(phenotypes.shape[1])

    phenotype = phenotypes[:, phen]
    phenotype -= phenotype.mean()
    phenotype /= phenotype.std()
    phenotype = np.reshape(phenotype, [N_cells, 1])

    phenotypes_tmp = phenotypes[:, env_prot_index]
    # phenotypes_tmp = normalise(phenotypes_tmp)

    Kinship = phenotypes_tmp.dot(phenotypes_tmp.transpose())
    # Kinship -= np.linalg.eigvalsh(Kinship).min() * np.eye(N_cells)
    Kinship *= covar_rescaling_factor(Kinship)

    # create different models and print the result including likelihood
    # create all the covariance terms
    direct_cov = FixedCov(Kinship)

    # noise
    noise_cov = FixedCov(np.eye(N_cells))

    # local_noise
    local_noise_cov = SQExpCov(X)
    # local_noise_cov.length = 100
    # local_noise_cov.act_length = False
    # environment effect
    environment_cov = ZKZCov(X, Kinship, rm_diag)

    # mean term
    mean = MeanBase(phenotype)

    #######################################################################
    # defining model
    #######################################################################
    cov = SumCov(noise_cov, local_noise_cov)
    cov = SumCov(cov, environment_cov)
    # if random_start_point:
    #     environment_cov.length = np.random.uniform(10, 300)
    #     environment_cov.scale = np.random.uniform(1, 15)
    #
    # else:
    #     environment_cov.length = 200
    environment_cov.length = 200
    # environment_cov.act_length = False

    if model == 'full':
        cov = SumCov(cov, direct_cov)
    else:
        direct_cov.scale = 0

    # define and optimise GP
    gp = GP(covar=cov, mean=mean)

    try:
        gp.optimize()
    except:
        print('optimisation', str(phen), 'failed')

    # log_lik[phen] = gp.LML()


    # rescale each terms to sample variance one
    # direct cov: unnecessary as fixed covariance rescaled before optimisation
    # local noise covariance
    tmp = covar_rescaling_factor(local_noise_cov.K() / local_noise_cov.scale)
    local_noise_cov.scale /= tmp
    # env effect
    tmp = covar_rescaling_factor(environment_cov.K() / environment_cov.scale ** 2)
    environment_cov.scale = environment_cov.scale ** 2 / tmp

    results = [protein_1,
               protein_2,
               direct_cov.scale,
               noise_cov.scale,
               local_noise_cov.scale,
               local_noise_cov.length,
               environment_cov.scale,
               environment_cov.length]
    results = np.reshape(results, [1, len(results)])

    result_header = 'protein 1' + ' ' + \
                    'protein 2' + ' ' + \
                    'direct_scale' + ' ' + \
                    'noise_scale' + ' ' + \
                    'local_noise_scale' + ' ' + \
                    'local_noise_length' + ' ' + \
                    'environment_scale' + ' ' + \
                    'environment_length'

    with open(output_file, 'w') as f:
        np.savetxt(f,
                   results,
                   delimiter=' ',
                   header=result_header,
                   fmt='%s',
                   comments='')

    # log_lik_file = output_file + '_loglik'
    # with open(log_lik_file, 'w') as f:
    #     np.savetxt(f, log_lik)


##############################################################################################
# running model
##############################################################################################

protein_1 = sys.argv[1]
protein_2 = sys.argv[2]
output_file = sys.argv[3]
image_dir = sys.argv[4]

expression_profile = image_dir+'/'+'expressions.txt'
position_file = image_dir+'/'+'positions.txt'


compute_pairwise_interaction('env', expression_profile, position_file, output_file, protein_1, protein_2)

##############################################################################################
# trial
##############################################################################################
# protein_1 = 'CREB'
# protein_2 = 'pBAD'
# output_file = '/Users/damienarnol1/Documents/local/pro/PhD/random_effects_cluster/tests/Ay10x1/pairwise_res/res.txt'
# image_dir = '/Users/damienarnol1/Documents/local/pro/PhD/random_effects_cluster/tests/Ay10x1/'
#
# expression_profile = image_dir+'/'+'expressions.txt'
# position_file = image_dir+'/'+'positions.txt'
#
#
# compute_pairwise_interaction('env', expression_profile, position_file, output_file, protein_1, protein_2)

