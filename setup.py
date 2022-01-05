import distutils.sysconfig
import distutils.version
from glob import glob
import os
import platform
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import sys


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


def openmp():
    try:
        flag = os.environ["OPENMP_FLAG"]
        return [flag]
    except KeyError:
        return []


# TODO(akshayka): need a better solution, maybe CFLAGS, but has bad
# interactions with macOS
def march_native():
    try:
        bit = os.environ["MARCH_NATIVE"]
        if bit == '1':
            return ["-march=native"]
        else:
            return []
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
    extra_compile_args=["-O3", "-std=c++11"] + openmp() + march_native()
)


def is_platform_mac():
    return sys.platform == 'darwin'


# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distutils behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py. This behavior is
# motivated by Apple dropping support for libstdc++.
if is_platform_mac():
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = distutils.version.LooseVersion(platform.mac_ver()[0])
        python_target = distutils.version.LooseVersion(
            distutils.sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

ext_modules = [_diffcp]

setup(
    name='diffcp',
    version="1.0.19",
    author="Akshay Agrawal, Shane Barratt, Stephen Boyd, Enzo Busseti, Walaa Moursi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    setup_requires=['pybind11 >= 2.4'],
    install_requires=[
        "numpy >= 1.15",
        "scs >= 2.0.2",  # 2.0.2 is the oldest version on conda forge
        "scipy >= 1.1.0",
        "pybind11 >= 2.4",
        "threadpoolctl >= 1.1",
        "ecos"],
    url="http://github.com/cvxgrp/diffcp/",
    ext_modules=ext_modules,
    license="Apache License, Version 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
