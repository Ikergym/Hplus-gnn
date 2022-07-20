# Import third-party libraries
import pickle5
import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

# Import custom libraries
sys.path.append(str(Path().absolute()))  # Add source directory to PYTHONPATH
import gnn_tools.data as gnn_data


# Define global features to include in graphs
global_features= ['mjb_leading_bjet', 'mjb_maxdr', 'mjb_mindr',
                  'jets_n', 'bjets_n'
                 ]

# Define scaling of global features: scaled feature[i] = feature[i] / scaling[i]
global_scale = np.asarray([1,1,1,1,1])

# Define scaling of node variables
node_scale=np.asarray([1,1,1,1,1,1,1])

# Define path for data fies 
parent_dir = Path('.').resolve().parents[0] / 'ATLASMCDATA'

# Define data files to use
files = ['Hpluscb_60.csv', 'Hpluscb_110.csv', 'Hpluscb_160.csv',
        'ttbarlight.csv', 'ttbarcc.csv', 'ttbarbb_split_1.csv',
        'ttbarbb_split_2.csv']

is_signals = [1, 1, 1, 0, 0, 0, 0]  # Define if files are signal or background

# Read number of rows
with open('nrows.json') as json_file:
    sizes_dict = json.load(json_file)

file_indices_dict = {}

with open('log.log', 'w') as log:

    starting_index = 0

    for file, is_signal in zip(files, is_signals):
        print(f'Processing file {file}...', file=log)
        filepath = parent_dir / file
        reader = pd.read_csv(filepath, chunksize=10000)
        size = sizes_dict[file]
        expected_nchunks = int(size / 10000)

        for ichunk, chunk in enumerate(reader):

            # Report local and global indices
            global_index = starting_index + ichunk
            print(f'Processing chunk {ichunk} out of {expected_nchunks}', file=log)
            print(f'Global chunk index: {global_index}', file=log)
                    
            # Define and process dataframe
            data_df = chunk  # Refactor variable
            data_df['IsSig'] = is_signal  ## Add IsSig column
            data_df['IsSig']=data_df['IsSig'].astype(int)  # Ensure type match
            data_df['eventNumber']=data_df['event_number'].astype(int)  # Ensure type match
            gnn_data.generate_pseudoMass(data_df)  # Generate pseudo mass

            # Create and save training graphs with even events
            graphs, booking = gnn_data.CreateTorchGraphs(data_df.query('eventNumber%2==0'), global_features, global_scale, node_scale)
            with open(f'Geometric_Data_Even/graphs_{global_index}.pkl', 'wb') as f:
                pickle.dump(graphs, f)
            booking.to_pickle(f'Geometric_Data_Even/booking_{global_index}.pkl')

            # Create and save testing graphs with odd events
            graphs, booking = gnn_data.CreateTorchGraphs(data_df.query('eventNumber%2==1'), global_features, global_scale, node_scale)
            with open(f'Geometric_Data_Odd/graphs_{global_index}.pkl', 'wb') as f:
                pickle.dump(graphs, f)
            booking.to_pickle(f'Geometric_Data_Odd/booking_{global_index}.pkl')

        # Recalculate and store indices for current file 
        ending_index = global_index
        file_indices_dict[file] = (starting_index, ending_index)
        starting_index = ending_index + 1


# Save indices dictionary as json
with open('geometric_indices.json', 'w') as f_ind:
    json.dump(file_indices_dict, f_ind)