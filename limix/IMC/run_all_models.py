__author__ = 'damienarnol1'

from . import util_functions
from .run_individual_model import run_individual_model
import sys


def run_all_models(input_directory, replicate_index=None, overwrite=False):
    # create a results directory in which to write the files
    if replicate_index is not None:
        results_directory = input_directory + '/results_' + str(replicate_index) + '/'
        rd_sp = True
    else:
        results_directory = input_directory+'/results/'
        rd_sp =False
    results_directory = util_functions.make_dir(results_directory, overwrite)

    # file_names
    positions = input_directory+'/positions.txt'
    expressions = input_directory+'/expressions.txt'

    # full model
    run_individual_model('full', expressions, positions, results_directory,
                         permute_positions=False, random_start_point=rd_sp)

    # full model plus random
    run_individual_model('full', expressions, positions, results_directory,
                         permute_positions=True, random_start_point=rd_sp)

    # env model
    run_individual_model('env', expressions, positions, results_directory,
                         permute_positions=False, random_start_point=rd_sp)

    # env model plus random
    run_individual_model('env', expressions, positions, results_directory,
                         permute_positions=True, random_start_point=rd_sp)

    return



analysis_directory = sys.argv[1]
replicate_index = sys.argv[2]
run_all_models(analysis_directory, replicate_index, False)
