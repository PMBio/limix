__author__ = 'damienarnol1'

import util_functions
from run_individual_model import run_individual_model


def run_all_models(input_directory, overwrite=False):
    # create a results directory in which to write the files
    results_directory = input_directory+'/results/'
    results_directory = util_functions.make_dir(results_directory, overwrite)

    # file_names
    positions = input_directory+'/positions.csv'
    expressions = input_directory+'/expressions.csv'

    # full model
    run_individual_model('full', expressions, positions, results_directory, permute_positions=False)

    # full model plus random
    run_individual_model('full', expressions, positions, results_directory, permute_positions=True)

    # env model
    run_individual_model('env', expressions, positions, results_directory, permute_positions=False)

    # env model plus random
    run_individual_model('env', expressions, positions, results_directory, permute_positions=True)

    return





run_all_models('/Users/damienarnol1/Documents/local/pro/PhD/social_effects/pipeline_test/test_1/', False)
