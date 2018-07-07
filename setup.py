#!/usr/bin/env python
import os

from setuptools import setup, find_packages
from distutils.core import setup

BASE = os.path.dirname(os.path.abspath(__file__))

setup(
    name='polymr',
    version="0.0.1",
    description="Pure MapReduce implementation in Python",
    url="https://www.github.com/Refefer/Polymr",
    packages=find_packages(BASE),
    scripts=[],
    install_requires=[
        "six"
    ],
    author='Andrew Stanton')
