# -*- coding: utf-8 -*-
import os
from setuptools import setup


def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()


requirements = [
    'datasets',
    'transformers<5.0',
    'torchaudio',
    'soundfile',
    'torch',
    'numpy'
]

extras = {
	"lm":[
		"pyctcdecode>=0.4.0",
		# "kenlm @ https://github.com/kpu/kenlm/archive/refs/heads/master.zip"
	]
}


setup(
	name='pythaiasr',
	version='1.3.0',
	packages=['pythaiasr'],
	url='https://github.com/pythainlp/pythaiasr',
	license='Apache Software License 2.0',
	author='Wannaphong Phatthiyaphaibun',
	author_email='wannaphong@yahoo.com',
	test_suite="tests",
	keywords = 'asr',
	description='Python Thai ASR',
    install_requires=requirements,
	extras_require=extras,
	long_description=(read('README.md')),
    long_description_content_type='text/markdown',
	classifiers= [
		'Development Status :: 5 - Production/Stable',
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
