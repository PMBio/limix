__author__ = 'damienarnol1'

import numpy as np
import os
import glob
import util_functions
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

    phenotypes = np.genfromtxt(expression_file, delimiter=' ', skiprows=1)
    image_names = os.listdir(IMC_dir)

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
    expression_names = os.listdir(expressions_dir)

    # full path of the file for processing
    position_files = glob.glob(position_dir+'/*')
    expression_files = glob.glob(expressions_dir+'/*')

    # reading and creating tree
    for image_index in range(0, len(position_names)):
        if position_names[image_index] != expression_names[image_index]:
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
