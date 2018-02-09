import pkg_resources
import os



def get_path_of_data_file(source_type, data_file):
    file_path = pkg_resources.resource_filename("gbmbkgpy", 'data/{0}/{1}'.format(source_type, data_file))

    return file_path



def get_path_of_data_dir():
    file_path = pkg_resources.resource_filename("gbmbkgpy", 'data')

    return file_path

def get_path_of_external_data_dir():

    file_path = os.environ['GBMDATA']

    return file_path