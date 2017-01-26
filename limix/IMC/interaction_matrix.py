__author__ = 'damienarnol1'

import numpy as np
import os
import glob
from . import util_functions
import shutil



def split_expression_file(expression_file, IMC_dir, output_dir):
    # read protein names
    with open(expression_file, 'r') as f:
        prot_tmp = f.readline()
    protein_names = prot_tmp.split(' ')
    protein_names[-1] = protein_names[-1][0:-1]  # removing the newline sign at the end of the last protein
    protein_names = protein_names[1:]
    protein_names = ' '.join(protein_names)

    # protein_names = np.reshape(protein_names, [len(protein_names), 1])

    phenotypes = np.loadtxt(expression_file, delimiter=' ', skiprows=1)
    image_names = sorted(os.listdir(IMC_dir))

    for image_index in range(0, len(image_names)):
        phenotype = phenotypes[phenotypes[:, 0] == image_index, 1:]
        file_name = image_names[image_index].split('_')[0]
        full_path = output_dir + '/' + file_name

        with open(full_path, 'w') as f:
            np.savetxt(f, phenotype, delimiter=' ', header=protein_names, comments='')


def create_analysis_tree(position_dir, expressions_dir, analysis_dir):
    # name of the images to check positions and expressions match
    position_names = os.listdir(position_dir)
    position_names = [name.split('_')[0] for name in position_names]
    position_names = sorted(position_names)

    expression_names = os.listdir(expressions_dir)
    expression_names = sorted(expression_names)

    # full path of the file for processing
    position_files = glob.glob(position_dir+'/*')
    position_files = sorted(position_files)

    expression_files = glob.glob(expressions_dir+'/*')
    expression_files = sorted(expression_files)

    # reading and creating tree
    for image_index in range(0, len(position_names)):
        if position_names[image_index] != expression_names[image_index]:
            print(position_names[image_index] + 'is not equal to ' + expression_names[image_index])
            raise Exception("Image names dont match for position and expression data")

        # create image directory
        dir_name = analysis_dir+'/'+position_names[image_index]
        dir_name = util_functions.make_dir(dir_name)

        # create file names
        position_file_cp = dir_name + '/' + 'positions.txt'
        expression_file_cp = dir_name + '/' + 'expressions.txt'

        # copy files
        shutil.copy(position_files[image_index], position_file_cp)
        shutil.copy(expression_files[image_index], expression_file_cp)


# 1 - define necessary file names and directory names
# 2 - create analysis tree
# 3 - submit one job per directory of the tree

all_expressions_file = '/homes/arnol/random_effect_model/data/membrane_expressions'
IMC_dir='/hps/nobackup/stegle/users/arnol/data/SampleSet_1/IMC_data/'
expression_dir = '/homes/arnol/random_effect_model/data/expression_levels/'
positions_dir = '/homes/arnol/random_effect_model/data/positions/'
analysis_dir = '/homes/arnol/random_effect_model/analysis_2/'

split_expression_file(all_expressions_file, IMC_dir, expression_dir)
create_analysis_tree(positions_dir, expression_dir, analysis_dir)

# protein list
protein_list = ['pAKT', 'AMPK', 'pBAD', 'bcatenin', 'CAHIX', 'CD20', 'CD3', 'CD44', 'CD45', 'CD68', 'CC3', 'cMyc',
                'CREB', 'Cytokeratin7', 'PanKeratin', 'Ecadherin', 'EGFR', 'EpCAM', 'ERa', 'pERK12', 'Fibronectin',
                'GATA3', 'HER2', 'H3', 'Ki67', 'PRAB', 'S6', 'SHP2', 'Slug', 'SMA', 'Twist', 'Vimentin']

# running model on every directory

# list directories
image_dirs = glob.glob(analysis_dir+'/*')
for image_dir in image_dirs:
    result_directory = image_dir+'/results/'
    util_functions.make_dir(result_directory, overwrite=False)
    for protein_1 in protein_list:
        for protein_2 in protein_list:
            file_name = result_directory+'/'+protein_1+'_'+protein_2
            command_line = \
                'bsub -o /homes/arnol/random_effect_model/log -M 15000 -R "rusage[mem=15000]" python pairwise_interaction.py ' + \
                protein_1 + ' ' + protein_2 + ' ' + file_name + ' ' + image_dir
            os.system(command_line)



