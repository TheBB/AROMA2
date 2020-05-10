#!/usr/bin/env python

from pathlib import Path
from setuptools import setup, find_packages
from distutils.extension import Extension

setup(
    name='Aroma',
    version='0.1.0',
    description='Advanced Reduced Order Modelling Applications',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    packages=find_packages(),
    install_requires=[
        'filebacked',
        'numpy',
        'scipy',
        'nutils',
    ],
)
