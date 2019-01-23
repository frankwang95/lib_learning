from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

exts = cythonize('learningC.pyx')
setup(ext_modules=exts, include_dirs=[np.get_include()])