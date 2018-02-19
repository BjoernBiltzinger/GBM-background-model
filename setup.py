from setuptools import setup, find_packages
import os


# Create list of data files
def find_data_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            paths.append(os.path.join('..', path, filename))

    return paths

extra_files = find_data_files('gbmbkgpy/data')

setup(

    name="gbmbkgpy",
    packages=find_packages(),
    version='v0.1',
    description=' GBM Background ',
    author='MPE GRB Team',
    author_email='jburgess@mpe.mpg.de',
    package_data={'': extra_files, },
    include_package_data=True,
    requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'scipy'
    ]

)