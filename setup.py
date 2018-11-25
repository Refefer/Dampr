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
    author='Andrew Stanton',
    author_email="refefer@gmail.com",
    classifiers=[
       "Development Status :: 3 - Alpha",
       "License :: OSI Approved :: Apache Software License",
       "Programming Language :: Python :: 2",
       "Programming Language :: Python :: 3",
       "Operating System :: OS Independent"
      ]
    )
