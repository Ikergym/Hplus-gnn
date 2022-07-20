import numpy as np
import pandas as pd


def pickle_to_csv(filename):
    """
    Read a pickle dataframe and
    save it in a csv file.
    """

    print(f'Converting {filename}')
    df = pd.read_pickle(filename)
    file_out = filename[:-3] + 'csv'
    df.to_csv(file_out)
    print(f'File converted to {file_out}')


def create_rows_json(filename):
    """
    Create a json file containing the number of
    rows for each file used for the training.
    """

    import json

    # Count of rows from ROOT original files
    Hpluscb_60_size = 47055
    Hpluscb_110_size = 52371
    Hpluscb_160_size = 14710
    ttbarlight_size = 791515
    ttbarcc_size = 1754895
    #ttbarbb_size = 7933973
    ttbarbb_split1_size = 3966986
    ttbarbb_split2_size = 3966987

    files = ['Hpluscb_60.csv', 'Hpluscb_110.csv', 'Hpluscb_160.csv',
             'ttbarlight.csv', 'ttbarcc.csv', 'ttbarbb_split_1.csv',
             'ttbarbb_split_2.csv']
    
    sizes = [Hpluscb_60_size, Hpluscb_110_size, Hpluscb_160_size, 
             ttbarlight_size, ttbarcc_size, ttbarbb_split1_size,
             ttbarbb_split2_size]
    
    # Construct dictionary
    nrows_dict = {}
    for key, value in zip(files, sizes):
        nrows_dict[key] = value

    # Save to json
    with open(filename, 'w') as f:
        json.dump(nrows_dict, f, indent=4)

    print(f'Created json file at: {filename}')