#!/usr/bin/env python
import os

from setuptools import setup, find_packages
from distutils.core import setup

BASE = os.path.dirname(os.path.abspath(__file__))

setup(
    name='polymr',
    version="0.1.2",
    description="Data Processing implementation in Python",
    url="https://www.github.com/Refefer/Polymr",
    packages=find_packages(BASE),
    test_suite="tests",
    scripts=[],
    install_requires=[
        "six"
    ],
    author='Andrew Stanton')
