import glob
import os

def FileParser(data_dir, data_type, data_format = 'txt'):
    """
    Returns the list of paths of all measurement data in the subdirectories of data_dir corresponding to the given data_type and data_format.

    Parameters
    ----------
    data_dir: path
        Directory for the measurement data. Note that all data must be stored in the direct subdirectory of data_dir.
    data_type: "app" or "res"
        Corresponds to the type of the measurement. "app" for approach curves and "res" for resonance curves.
    data_format: "str"
        The file extension of the desired data. "txt" by default. 

    Returns
    -------
    files: list
        A list of filepaths corresponding to the desired data_type and data_format
    """

    type_string = {'res': '*Resonance*.', 'app': '*Approach*.'}

    path = os.path.join(data_dir, '*', type_string[data_type] + data_format)

    files = glob.glob(path, recursive = True)

    print('{} files loaded'.format(len(files)))
    
    return files

if __name__ == "__main__":
    files = FileParser('./Data/Experiment/Tapping', 'app', 'json')
    for file_name in files:
        print(file_name)