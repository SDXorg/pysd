from distutils.core import setup

setup(
    name='pysd',
    version='0.1.1',
    author='James Houghton',
    author_email='james.p.houghton@gmail.com',
    packages=['pysd', 'pysd.translators', 'pysd.functions'],
    url='https://github.com/JamesPHoughton/pysd',
    download_url='https://github.com/JamesPHoughton/pysd/archive/v0.1.1.tar.gz',
    license='LICENSE.txt',
    description='System Dynamics Modeling in Python',
    long_description='see https://github.com/JamesPHoughton/pysd',
    keywords=['System Dynamics', 'XMILE', 'Vensim'],
    install_requires=[
        'pandas >= 0.14.1',
        'numpy >= 1.8.1',
        'scipy >= 0.13.3',
        'parsimonious >= 0.5',
        'networkx >= 1.9',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)