from distutils.core import setup

setup(
    name='pysd',
    version='0.1.0',
    author='James Houghton',
    author_email='james.p.houghton@gmail.com',
    packages=['pysd', 'pysd.translators', 'pysd.functions'],
    url='https://github.com/JamesPHoughton/pysd',
    download_url='https://github.com/JamesPHoughton/pysd/tarball/0.1.0',
    license='LICENSE.txt',
    description='System Dynamics Modeling in Python',
    long_description=open('README.md').read(),
    keywords=['System Dynamics', 'XMILE', 'Vensim'],
    install_requires=[
        'pandas >= 0.14.1',
        'numpy >= 1.8.1',
        'scipy >= 0.13.3',
        'parsimonious >= 0.5',
        'networkx >= 1.9',
    ],
)