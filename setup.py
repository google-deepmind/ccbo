# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='ccbo',
    version='1.0',
    description=(
        'Code for the paper: Aglietti, Malek, Ktena, Chiappa. "Constrained'
        ' Causal Bayesian Optimization" ICML 2023.'
    ),
    author='DeepMind',
    author_email='aglietti@deepmind.com',
    license='Apache License, Version 2.0',
    url='https://github.com/deepmind/ccbo',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'emukit',
        'GPy',
        'graphviz',
        'matplotlib',
        'numpy',
        'networkx',
        'paramz',
        'pygraphviz',
        'scipy',
        'scikit-learn',
        'typing_extensions',
        'ml_collections'
    ],
    tests_require=['mock'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
