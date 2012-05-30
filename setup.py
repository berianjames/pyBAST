from distutils.core import setup

setup(
    name='pyBA',
    version='0.3.0',
    author='Berian James',
    author_email='berian@berkeley.edu',
    packages=['pyBA'],
    scripts=[],
    url='http://pypi.python.org/pypi/pyBA/',
    license='LICENSE.txt',
    description='Python implementation of Bayesian Astrometry framework.',
    long_description=open('README.rst').read(),
)
