# __author__ = 'damienarnol1'
#
# # THIS IS FAILING. FIX IT!!!!!!!!!!!!!!!!!!!!!!!!!
# 
# import pdb
#
# import numpy as np
#
# from limix.core.covar import SQExpCov
# from limix.core.covar import ZKZCov
# from limix.core.covar import FixedCov
# from limix.core.covar import SumCov
# from limix.core.gp import GP
# from limix.core.mean import MeanBase
# from limix.utils.preprocess import covar_rescaling_factor
#
# from limix.IMC import generate_data
#
#
# def normalise(mat, axis=1):
#     mat -= np.reshape(mat.mean(axis=axis), [mat.shape[0], 1])
#     mat /= np.reshape(mat.std(axis=axis), [mat.shape[0], 1])
#     mat /= np.sqrt(mat.shape[1])
#     return mat
#
# res_true_file = '/Users/damienarnol1/Documents/local/pro/PhD/social_effects/simulations/results_11/true_parameters.csv'
# res_simulated_file = \
#     '/Users/damienarnol1/Documents/local/pro/PhD/social_effects/simulations/results_11/simulated_parameters.csv'
#
# N_tests = 100
#
# results_true = np.zeros([N_tests, 3])
# results_simulated = np.zeros([N_tests, 3])
#
# for i in range(0, N_tests):
#
#     ########################################################################################
#     # simulating data
#     ########################################################################################
#     data_directory = '/Users/damienarnol1/Documents/local/pro/PhD/data/Zurich_data/output/tmp/'
#
#     X_file = data_directory + '/cell_locations_1'
#     phenotype_file = data_directory+'/membrane_1'
#
#     # read data
#     X = np.genfromtxt(X_file, delimiter=',', skiprows=1)
#     phenotypes = np.genfromtxt(phenotype_file, delimiter=' ', skiprows=1)
#     phenotypes = np.delete(phenotypes, 0, axis=1)
#     phenotypes = normalise(phenotypes)
#     N_cells = phenotypes.shape[0]
#
#     Kinship = phenotypes.dot(phenotypes.transpose())
#     Kinship -= np.linalg.eigvalsh(Kinship).min() * np.eye(N_cells)
#     Kinship *= covar_rescaling_factor(Kinship)
#     # Kinship =None
#     rm_diagonal = True
#     # possible values to switch off are local, noise, social, direct
#     switch_off = []
#     # switch_off = []
#
#     data, parameters = generate_data.generate_complete_model(N_cells, X, Kinship, switch_off, rm_diagonal)
#
#     ########################################################################################
#     # inference
#     ########################################################################################
#     # direct effect
#     direct_cov = FixedCov(parameters['Kinship'])
#
#     # noise
#     noise_cov = FixedCov(np.eye(N_cells))
#
#     # local_noise
#     local_noise_cov = SQExpCov(parameters['X'])
#     # local_noise_cov.length = parameters['local_noise_length']
#     local_noise_cov.length = N_cells / 6.0
#     # local_noise_cov.act_length = False
#
#     # environment effect
#     environment_cov = ZKZCov(parameters['X'], parameters['Kinship'], rm_diagonal)
#     # environment_cov.length = parameters['social_length']
#     environment_cov.length = N_cells / 6.0
#     # environment_cov.act_length = False
#
#     # total cov and mean
#     cov = SumCov(direct_cov, noise_cov)
#     cov = SumCov(cov, local_noise_cov)
#     cov = SumCov(cov, environment_cov)
#     # cov = SumCov(direct_cov, noise_cov)
#     # cov = SumCov(cov, local_noise_cov)
#
#     mean = MeanBase(data)
#
#     # define and optimise GP
#     gp = GP(covar=cov, mean=mean)
#
#     try:
#         gp.optimize()
#     except:
#         print('optimisation', str(i), 'failed')
#         continue
#
#     # show results
#     # print "true parameters vs inferred parameters "
#     # print "direct_scale = ", parameters['direct_scale'], " ", direct_cov.scale
#     # print "noise_scale = ", parameters['noise_scale'], " ", noise_cov.scale
#     # print "local_noise_scale = ", parameters['local_noise_scale'], " ", local_noise_cov.scale
#     # print "local_noise_length = ", parameters['local_noise_length'], " ", local_noise_cov.length
#     # print "environment_scale = ", parameters['social_scale'], " ", environment_cov.scale
#     # print "environment_length = ", parameters['social_length'], " ", environment_cov.length
#
#     results_true[i, :] = [parameters['social_scale'], parameters['direct_scale'],
#                           parameters['direct_scale']+parameters['noise_scale']+
#                           parameters['local_noise_scale']+parameters['social_scale']]
#     # results_simulated[i, :] = [direct_cov.scale, environment_cov.scale,
#     #                             environment_cov.scale + direct_cov.scale + noise_cov.scale + local_noise_cov.scale]
#     # results_simulated[i, :] = [direct_cov.scale, environment_cov.scale,
#     #                             environment_cov.scale + direct_cov.scale + noise_cov.scale]
#     # results_simulated[i, :] = [noise_cov.scale, direct_cov.scale,
#     #                             noise_cov.scale + direct_cov.scale + local_noise_cov.scale]
#     results_simulated[i, :] = [environment_cov.scale, direct_cov.scale,
#                                 noise_cov.scale + direct_cov.scale + local_noise_cov.scale + environment_cov.scale]
#     # print 'local noise scale ', local_noise_cov.scale
#
#     print('test', str(i), ' done ')
#
#
# np.savetxt(res_true_file, results_true, delimiter=' ')
# np.savetxt(res_simulated_file, results_simulated, delimiter=' ')
#
#
# #
# # N_cells = 1500
# # cov, se_params, Kinship = generate_data.generate_zkz(N_cells)
# # X = se_params[0]
# # scale = se_params[1]
# # length = se_params[2]
# # data = np.random.multivariate_normal(np.array([0] * N_cells), cov, 1).transpose()
# #
# # se = ZKZCov(X, Kinship)
# # # se.scale = scale
# # # se.length = length
# # gp = GP(MeanBase(data), se)
# # gp.optimize()
# #
# # print "true parameters vs inferred parameters "
# # print "scale = ", scale, " ", se.scale
# # print "length = ", length, " ", se.length
#
# #
# # N_cells = 1500
# # rm_zkz_diag = False
# #
# # data, parameters = generate_data.generate_simple(N_cells, rm_zkz_diag)
# #
# # # Kinship = parameters['Kinship']
# # phenotype = data
# # X = parameters['X']
# # length = parameters['length']
# # scale = parameters['scale']
# # noise_scale = parameters['noise']
# # Kinship = parameters['Kinship']
# # N = N_cells
# #
# # # defining ZKZ covariance term
# # zkz = ZKZCov(X, Kinship, rm_zkz_diag)
# # # zkz = SQExpCov(X)
# # # zkz.length = length
# # # zkz.scale = scale
# #
# # # fixed noise
# # noise = FixedCov(np.eye(N_cells))
# #
# # covar = SumCov(noise, zkz)
# # mean = MeanBase(phenotype)
# #
# # # optimise model
# # gp = GP(covar=covar, mean=mean)
# # # TODO initialise parameters
# # # TODO problem with calc_ste = True -> implement missing functions
# # gp.optimize(calc_ste=False)
# #
# # print "true parameters vs inferred parameters "
# # print "scale = ", scale, " ", zkz.scale
# # print "length = ", length, " ", zkz.length
# # print "noise scale = ", noise_scale, " ", noise.scale
# #
#
#
# # direct effect term
# # tmp = Kinship
# # tmp *= covar_rescaling_factor(Kinship)
# # directCov =FixedCov(tmp)
# #
# # # social effect term
# # tmp = Z.dot(Kinship.dot(Z.transpose()))
# # tmp *= covar_rescaling_factor(tmp)
# # socialCov = FixedCov(tmp)
# #
# # # noise term
# # noise = FixedCov(np.eye(N))
# #
# # # local Noise term
# # localNoise = SQExpCov(X)
# #
# # # total covariance
# # # TODO check syntax
# # covar = SumCov(directCov, socialCov)
# # covar = SumCov(covar, noise)
# # covar = SumCov(covar, localNoise)
# #
# # # mean
# # mean = phenotype
# #
# # gp = GP(covar=covar, mean=mean)
# # # TODO initialise parameters
# # gp.optimize(calc_ste=True)
# #
# #
# # # get results here
