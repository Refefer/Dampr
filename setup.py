#!/usr/bin/env python
import os

from setuptools import setup, find_packages
from distutils.core import setup

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "README.md")) as f:
    long_description = f.read()

setup(
    name='dampr',
    version="0.2.1",
    description="Data Processing implementation in Python",
    long_description=long_description,
    url="https://www.github.com/Refefer/Dampr",
    packages=find_packages(BASE),
    test_suite="tests",
    author='Andrew Stanton',
    author_email="refefer@gmail.com",
    classifiers=[
       "Development Status :: 4 - Beta",
       "License :: OSI Approved :: Apache Software License",
       "Programming Language :: Python :: 2",
       "Programming Language :: Python :: 3",
       "Operating System :: OS Independent"
      ]
    )
