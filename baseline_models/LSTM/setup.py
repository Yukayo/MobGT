import sys

sys.path.insert(0, "..")

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

filename = 'algos'
full_filename = 'algos.pyx'

ext_modules = [Extension(filename, [full_filename],
                         language='c++',
                         extra_compile_args=['-O3', '-march=native', '-ffast-math', './openmp'],
                         extra_link_args=['./openmp'])]

setup(
    cmdclass={
        'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[np.get_include()])



import pickle
import torch
import os
path="../data/train"
path1="../data"

# data1,data2=torch.load((path+'.pickle'))
# with open(os.path.join(path1, f'train.pickle'), 'rb') as f:
#     mols = pickle.load(f)
data = pickle.load(open(path + '.pickle', 'rb'))

print(1)