from setuptools import find_packages
from setuptools import setup

description = """A synthetic dataset of school-level mathematics questions.

This dataset code generates mathematical question and answer pairs, from a range
of question types (such as in arithmetic, algebra, probability, etc), at roughly
school-level difficulty. This is designed to test the mathematical learning and
reasoning skills of learning models.

Original paper: Analysing Mathematical Reasoning Abilities of Neural Models
(Saxton, Grefenstette, Hill, Kohli) (https://openreview.net/pdf?id=H1gR5iR5FX).
"""

setup(
    name='mathematics_dataset',
    version='1.0.1',
    description='A synthetic dataset of school-level mathematics questions',
    long_description=description,
    author='DeepMind',
    author_email='saxton@google.com',
    license='Apache License, Version 2.0',
    keywords='mathematics dataset',
    url='https://github.com/deepmind/mathematics_dataset',
    packages=find_packages(),
    install_requires=[
        'absl-py>=0.1.0',
        'numpy>=1.10,<1.25',
        'six',
        'sympy>=1.2,<1.10',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
