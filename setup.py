from setuptools import setup, find_packages

exec(open('pysd/_version.py').read())
print(__version__)

test_pckgs = open('dev-requirements.txt').read().strip().split('\n')

setup(
    name='pysd',
    version=__version__,
    python_requires='>=3.7',
    author='James Houghton',
    author_email='james.p.houghton@gmail.com',
    packages=find_packages(exclude=['docs', 'tests', 'dist', 'build']),
    url='https://github.com/JamesPHoughton/pysd',
    license='LICENSE.txt',
    description='System Dynamics Modeling in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords=['System Dynamics', 'Vensim', 'XMILE'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',

        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=open('requirements.txt').read().strip().split('\n'),
    tests_require=test_pckgs,
    extras_require={
        "test": test_pckgs,
        "docs": [
            # pin sphinx to match what RTD uses:
            # https://github.com/readthedocs/readthedocs.org/blob/ecac31de54bbb2c100f933e86eb22b0f4389ba84/requirements/pip.txt#L16
            'sphinx<2',
            'sphinx-rtd-theme<0.5',
            'docutils<0.18'
            ]
        },
    package_data={
        'translation': [
            '*/parsing_grammars/*.peg'
        ]
    },
    include_package_data=True
)
