import os
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
from glob import glob
import subprocess


with open("README.md", "r") as fh:
    long_description = fh.read()


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        try:
            import pybind11
        except ImportError:
            if subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                raise RuntimeError('pybind11 install failed.')
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def get_openmp_flag():
    try:
        flag = os.environ["OPENMP_FLAG"]
        return [flag]
    except KeyError:
        return []


_diffcp = Extension(
        '_diffcp',
        glob("cpp/src/*.cpp"),
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            "cpp/external/eigen",
            "cpp/external/eigen/Eigen",
            "cpp/include",
        ],
        language='c++',
        extra_compile_args=["-O3", "-std=c++11", "-march=native"] + get_openmp_flag()
)

ext_modules = [_diffcp]

setup(
    name='diffcp',
    version="1.0.7",
    author="Akshay Agrawal, Shane Barratt, Stephen Boyd, Enzo Busseti, Walaa Moursi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    setup_requires=['pybind11 >= 2.4'],
    install_requires=[
        "numpy >= 1.15",
        "scs >= 2.1.1",
        "scipy >= 1.1.0",
        "pybind11 >= 2.4",
        "threadpoolctl >= 1.1"],
    url="http://github.com/cvxgrp/diffcp/",
    ext_modules=ext_modules,
    license="Apache License, Version 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
