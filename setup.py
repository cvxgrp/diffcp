from setuptools import Extension, setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


_proj = Extension("_proj",
                  sources=["diffcp/proj.c"],
                  extra_compile_args=["-O3"])

setup(
    name="diffcp",
    version="1.0.2",
    author="Akshay Agrawal, Shane Barratt, Stephen Boyd, Enzo Busseti, Walaa Moursi",
    description="Differentiating through cone programs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.15",
        "scs >= 2.0.0",
        "scipy >= 1.1.0"],
    ext_modules=[_proj],
    license="Apache License, Version 2.0",
    url="http://github.com/cvxgrp/diffcp/",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
