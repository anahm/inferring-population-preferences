"""
setup.py

Python script to cythonize the mh_alg.pyx code.

2.16.15
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'mh_alg_3 app',
  ext_modules = cythonize("mh_alg.pyx"),
)
