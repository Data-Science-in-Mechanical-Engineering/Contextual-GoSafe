from setuptools import setup
from os import path


current_dir = path.abspath(path.dirname(__file__))

with open(path.join(current_dir, 'README.rst'), 'r') as f:
    long_description = f.read()

with open(path.join(current_dir, 'requirements.txt'), 'r') as f:
    install_requires = f.read().split('\n')

setup(
    name='gosafeopt',
    version='0.16',
    author='Bhavya Sukhija',
    author_email='sukhijab@ethz.ch',
    packages=['gosafeopt'],
    url='https://github.com/Data-Science-in-Mechanical-Engineering/Contextual-GoSafe',
    license='MIT',
    description='Safe Bayesian optimization',
    long_description=long_description,
    setup_requires='numpy',
    install_requires=install_requires,
    keywords='Bayesian optimization, Safety',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5'],
)
