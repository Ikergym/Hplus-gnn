# Script to convert some pickle files into csv files
import sys
from pathlib import Path

SRC_PATH = str(Path(__file__).parents[1].absolute())
sys.path.append(SRC_PATH)  # Add source directory to PYTHONPATH

from utils.utils import pickle_to_csv


# Define the location of the .pkl files
DATA_PATH = SRC_PATH + '/ATLASMCDATA/'

# Specify the files to convert to .csv
files = ['Hpluscb_60.pkl']

for file in files:
    path_input = DATA_PATH + file
    path_output = file[:-3] + 'csv'
    print(f'Converting {file} to {path_output}')
    pickle_to_csv(path_input)
    print(f'Sucessfully converted {file} to {path_output}')

