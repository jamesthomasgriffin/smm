try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    with open('README.rst') as f:
        LONG_DESCRIPTION = f.read()
except (IOError, ImportError):
    LONG_DESCRIPTION = 'Simplicial mixture models.'

import smm
VERSION = smm.__version__

setup(
    name='SMM',
    version=VERSION,
    author='James Griffin',
    author_email='james.griffin@cantab.net',
    packages=['smm', 'smm.rvs', 'smm.lemm'],
    url='https://github.com/jamesthomasgriffin/smm',
    license='new BSD',
    description='Implementation of simplicial mixture models.',
    long_description=LONG_DESCRIPTION,
    install_requires=[
        "numpy>=1.16.2",
        "numba>=0.43.1",
    ],
)
