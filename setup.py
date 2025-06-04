#!/usr/bin/env python

from setuptools import setup

setup(name='susan',
	version='0.1',
	description='SUSAN framework for CryoET subtomogram averaging',
	author='Ricardo Miguel Sánchez Loayza',
	author_email='ricardo.sanchez@embl.de',
	url='https://github.com/rkms86/SUSAN',
	packages=['susan','susan.data','susan.io','susan.utils','susan.project','susan.utils.txt_parser'],
	package_dir={'susan': 'susan'},
	package_data={'susan': ['bin/susan*']},
	install_requires=['numpy>=1.20.0','scipy>=1.6.0','numba'],
	zip_safe=False,
)


