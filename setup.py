from setuptools import setup, find_packages

exec(open('pysd/_version.py').read())
print(__version__)

setup(
    name='pysd',
    version=__version__,
    python_requires='>=3.7',
    author='James Houghton',
    author_email='james.p.houghton@gmail.com',
    packages=find_packages(exclude=['docs', 'tests', 'dist', 'build']),
    url='https://github.com/JamesPHoughton/pysd',
    license='LICENSE',
    description='System Dynamics Modeling in Python',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords=['System Dynamics', 'Vensim', 'XMILE'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Intended Audience :: Science/Research',

        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=open('requirements.txt').read().strip().split('\n'),
    package_data={
        'translators': [
            '*/parsing_grammars/*.peg'
        ]
    },
    include_package_data=True
)
