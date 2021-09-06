# -*- coding: utf-8 -*-
import os
from setuptools import setup


def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()


requirements = [
    'datasets',
    'transformers',
    'torchaudio',
    'soundfile',
    'torch',
    'numpy'
]


setup(
	name='pythaiasr',
	version='0.2',
	packages=['pythaiasr'],
	url='https://github.com/wannaphong/pythaiasr',
	license='Apache Software License 2.0',
	author='Wannaphong Phatthiyaphaibun',
	author_email='wannaphong@yahoo.com',
	keywords = 'asr',
	description='Python Thai ASR',
    install_requires=requirements,
	long_description=(read('README.md')),
    long_description_content_type='text/markdown',
	classifiers= [
		'Development Status :: 1 - Planning',
		'Intended Audience :: Developers',
		'Natural Language :: Thai',
		'License :: OSI Approved :: Apache Software License',
		'Operating System :: OS Independent',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: Implementation :: CPython',
		'Topic :: Scientific/Engineering',
	],
)
