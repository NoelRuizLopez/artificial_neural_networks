#!/usr/bin/python

# Copyright 2017 Altran
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# THIS FILE IS MANAGED BY THE GLOBAL REQUIREMENTS REPO - DO NOT EDIT
# Copyright 2011 OpenStack, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if sys.version_info < (2, 7):
    requirements.append('argparse')
elif sys.version_info < (2, 7):
    raise 'Must use python 2.7 or greater'

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='artififical_neural_network',
    version='0.1b',
    author='Noel Ruiz',
    author_email='',
    description='ANN Demo',
    long_description=long_description,
    install_requires=requirements,
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    data_files=[],
    package_data= {'': ['*.json']},
    license='Apache License 2.0',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Topic :: System"
        "Framework :: Flask"
    ],
    entry_points={
        'console_scripts': [
            'ann_demo=redes_neuronales_numpy.training1:main',
        ],
    },

)
