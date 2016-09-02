__author__ = 'damienarnol1'

import numpy as np


def split_expression_file(expression_file, IMC_dir, output_dir):
    # read protein names
    with open(expression_file, 'r') as f:
        prot_tmp = f.readline()
    protein_names = prot_tmp.split(' ')
    protein_names = np.reshape(protein_names, [len(protein_names), 1])

    phenotypes = np.genfromtxt(expression_file, delimiter=' ', skiprows=1)


def create_analysis_tree(position_dir, expressions_dir):
    pass
