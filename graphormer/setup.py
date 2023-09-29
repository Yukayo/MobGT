#!/usr/bin/python  
#python version: 2.7.3  
#Filename: SetupTestOMP.py  
   
# Run as:    
#    python setup.py build_ext --inplace    
     
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


# ext_module = cythonize("TestOMP.pyx")    
# ext_module = Extension(  
#                         "algos",  
#             ["algos.pyx"],  
#             # extra_compile_args=["/openmp"],  
#             # extra_link_args=["/openmp"],  
#             )  
     
# setup(  
#     cmdclass = {'build_ext': build_ext},  
#         ext_modules = [ext_module],   
# )  