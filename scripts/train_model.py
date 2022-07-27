import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json


# Import custom libraries
SRC_PATH = Path(__file__).parents[1].absolute()
sys.path.append(str(SRC_PATH))  # Add source directory to PYTHONPATH

import gnn_tools.data as gnn_data


# Read indices for each file
with open(SRC_PATH / 'utils/geometric_indices.json') as json_file:
    file_indices = json.load(json_file)

# Define files used for training:
files = ['Hpluscb_160.csv', 'ttbarlight.csv']

## Hard coded workaround
file_indices['ttbarlight.csv'] = [13, 14]

# Create array of indices to use
indices = np.array([], dtype=int)

for file in files:
    start, stop = tuple(file_indices[file])
    indices = np.append(indices, np.arange(start, stop + 1, dtype=int))

# Generate loaders for train and test sets
path_train = SRC_PATH / 'Geometric_Data_Even'
path_test = SRC_PATH / 'Geometric_Data_Odd'
loader_train = gnn_data.Loader(path_train, indices)
loader_test = gnn_data.Loader(path_test, indices)


# Read one file to get number of node, edge and global features
for data_set in loader_train:
    node_feats_in=data_set[0].x.shape[1]
    edge_feats_in=data_set[0].edge_attr.shape[1]
    global_feats_in=data_set[0].u.shape[1]+1 # +1 for pseudo mass as additional global
    break

loader_train.index = 0  # Reset loader index


import gnn_tools.model as gnn_model
import torch

#cuda = torch.device('cuda') # gpu 
cuda = torch.device('cpu') # cpu Using CPU instead of cuda on debugging
model = gnn_model.GeneralMPGNN(node_feats_in, edge_feats_in, global_feats_in)
model.to(cuda) # move model onto gpu


import gnn_tools.train as gnn_train

# Gradient descent method
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False) 
# Loss function
criterion = torch.nn.BCELoss(reduction='none')

# Run full training
model = gnn_train.runTraining(model, loader_train, loader_test, 10, 50, "bestModel.pt", cuda, criterion, optimizer)