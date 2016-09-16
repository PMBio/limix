__author__ = 'damienarnol1'

import os
import shutil
import glob
import numpy as np
import scipy.spatial as SS



def make_dir(dir_name, overwrite=False):
    while dir_name[-1] == '/':
        dir_name = dir_name[0:-1]

    if overwrite:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    else:
        count = 2
        dir_name_tmp = dir_name
        while os.path.exists(dir_name_tmp):
            dir_name_tmp = dir_name+str(count)
            count += 1
        dir_name = dir_name_tmp

    os.makedirs(dir_name)
    return dir_name+'/'


def compute_all_pairwise_distances(positions_directory, output_directory):
    all_files = sorted(glob.glob(positions_directory+'/*'))

    counter = 0

    for file_name in all_files:
        positions = np.loadtxt(file_name, delimiter=',')
        distances = SS.distance.pdist(positions, 'euclidean')

        image_name = file_name.split('/')[-1]
        output_file_name = output_directory+'/'+image_name
        np.savetxt(output_file_name, distances, delimiter=' ')





