__author__ = 'damienarnol1'

import os
import shutil


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

