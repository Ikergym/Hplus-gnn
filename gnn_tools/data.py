from __future__ import print_function
from ast import For
import torch_geometric
import torch_geometric.utils as utils
from torch_geometric.data import Data
import time
import networkx as nx
import matplotlib.pyplot as plt
import pickle5
import numpy as np
import pandas as pd
from itertools import combinations, permutations
import torch
import matplotlib.cm as cm
from typing import List, Tuple


def generate_pseudoMass(df):
    #df['pseudo_mH']=df['mH_label']
    #df.loc[df['IsSig']==0, 'pseudo_mH'] =  np.random.randint(4, 11, (df.query('IsSig==0')).shape[0])*100
    df['pseudo_mH'] = 0.5  # Dummy pseudo_mH to keep code running
    df['sample_weight'] = 1.0
    return df


# Get sample weights from weight_rw and number of occurances of discrete feature e.g. mH_label or IsSig
def balanceFeatures(df, feature):
    df['sample_weight']=df['weight_rw']
    occurances=df[feature].value_counts()
    # divide weights by average weight of each class
    for occ in occurances.keys():
        sumWeights=sum(df.query('{}=={}'.format(feature,occ))['weight_rw'].to_numpy())
        df.loc[df[feature]==occ, 'sample_weight'] = df.loc[df[feature]==occ, 'sample_weight']*1/sumWeights
    df['sample_weight']=df['sample_weight']*1/np.mean(df['sample_weight'].to_numpy())
    return df


def eval_csv_array(str_array: str) -> np.ndarray:
  return np.fromstring(str_array[2:-2], dtype='float', sep=' ')


def loadData(file):
  print('Loading in datafile..')
  with open(file, "rb") as fh:
    data = pickle5.load(fh)
  cut = '(mcChannelNumber!=312444 | mH_label!=600)'
  data = data.query(cut)
  # remove duplicate cols
  data=data.loc[:,~data.columns.duplicated()]
  # generate pseudo_mass
  print('Generating pseudo-mass...')
  data=generate_pseudoMass(data)
  # balance features so masses are balanced
  print('Balancing features...')
  data = balanceFeatures(data, 'mH_label')
  # balance sig/bkg frac
  data.loc[data['IsSig']==0, 'sample_weight']=data.loc[data['IsSig']==0, 'sample_weight']*7
  print('Done')
  return data


def event2networkx(event, global_features, global_scale,
                   node_scale, include_node_qg_bdt=True):
  
    """
    Create a NetworkX Graph object from an event.
    Global features are treated as graph attributes.
    
    Nodes represent leptons/jets. Node attributes
    are pt, eta, phi and e for each node. All node
    attributes are introduced in a 'features' key
    of the node, and the values are represented in
    a formatted list as:
    [pt, phi, eta, 0, 0, encoding].

    The encodings is a special number that codifies
    the nature of each particle:
     0: met (missing transverse energy)
    -1: electron
    -2: muon
     1: jet
    
    Edges represent difference in those attributes
    between any pair of events.
    """
            

    # Create empty graph
    G = nx.DiGraph()
    
    # Define book-keeping features
    bookkeeping_features=['eventNumber','run_number',
                          'channel',
                          'mH','pseudo_mH', 
                          'scores_70','jets_n']
    
    # Assign global features as graph attributes
    G.graph['features']=np.asarray(event[global_features].tolist())/global_scale
    G.graph['book_keeping']=dict(zip(bookkeeping_features,event[bookkeeping_features].tolist()))
    G.graph['sample_weight']=event['sample_weight']
    G.graph['IsSig']=event['IsSig']
    G.graph['pseudo_mH']=event['pseudo_mH']/1000
    
    # Add nodes
    index=0  #The index acts as a label for each node
    
    # Add the met object with eta=0
    met_encoding = [0]
    met_features=np.asarray(event[['met','met_phi']].tolist()+[0]+[0,0]+met_encoding)
    G.add_node(index, features=met_features)#/node_scale) 
    index=index+1
    
    # Add electrons
    electron_features=['el1_pt','el1_phi', 'el1_eta']#,'el_e']
    nElectrons=event['el_n']
    nfeatures=len(electron_features)
    electrons=np.vstack(event[electron_features])
    electron_encoding=[-1]

    for i in range(0, nElectrons):
        electron=list((electrons[0:nfeatures, i]).flatten())+[0,0]+electron_encoding
        G.add_node(index, features=np.asarray(electron))#/node_scale)
        index=index+1

    # Add muons   
    muon_features=['mu1_pt','mu1_phi', 'mu1_eta']#, 'mu_e']
    nMuons=event['mu_n']
    nfeatures=len(muon_features)
    muons=np.vstack(event[muon_features])
    muon_encoding=[-2]

    for i in range(0, nMuons):
        muon=list((muons[0:nfeatures, i]).flatten())+[0,0]+muon_encoding
        G.add_node(index, features=np.asarray(muon))#/node_scale)
        index=index+1

    # Add jets
    jet_features=['jets_pt', 'jets_phi', 'jets_eta']#, 'jet_e', 'jet_tagWeightBin_DL1r_Continuous','jet_qg_BDT_calibrated']
    nJets=event['jets_n']
    nfeatures=len(jet_features)

    # Process string type arrays (from csv) as numpy arrays
    for feature in jet_features:
      if isinstance(event[feature], str):
        event[feature] = eval_csv_array(event[feature])

    # Create 2D jets array for easy iteration
    jets=np.vstack(event[jet_features])
    jet_encoding=[1]

    for i in range(0, nJets):
        jet=list((jets[0:nfeatures, i]).flatten())+ [0,0] +jet_encoding
        G.add_node(index, features=np.asarray(jet))#/node_scale)
        index=index+1


    # Add edges
    objects=list(G.nodes)
    pairs=permutations(objects, 2)  # calculate all pairs of nodes
    node_scale = [1, 1, 1]  # dummy node_scale

    # Calculate delta_phi, delta_eta and delta_r for all pairs
    for pair in pairs:
          delta_phi=np.arctan2(np.sin(node_scale[1]*(G.nodes[pair[0]]['features'][1]-G.nodes[pair[1]]['features'][1])), np.cos(node_scale[1]*(G.nodes[pair[0]]['features'][1]-G.nodes[pair[1]]['features'][1])))/node_scale[1]# this is computationally expensive but give correct angle and sign
          delta_eta=G.nodes[pair[0]]['features'][2]-G.nodes[pair[1]]['features'][2]
          delta_R = np.sqrt((node_scale[1]*delta_phi)**2+(node_scale[2]*delta_eta)**2)/(node_scale[1]+node_scale[2])
          G.add_edge(pair[0], pair[1], features=np.asarray([delta_phi, delta_eta, delta_R]))

    return G


def getNodeFeatures(G):
  """
  Return the node features as an array.
  
  Args:
  G: NetworkX Graph object
  
  Returns:
  torch tensor of dimension (n_of_nodes, n_of_features)
  """
  x=np.zeros(shape=(G.number_of_nodes(), len(G.nodes[0]['features'])))
  for node in G.nodes:
    x[int(node)]=np.asarray(G.nodes[node]['features'])    
  return torch.from_numpy(x)


def getEdgeFeatures(G):
  """
  Return the edge attributes as an array
  
  Args:
  G: NetworkX Graph object
  
  Returns:
  torch tensor of dimension (n_of_edges, n_of_features)
  """
  e=np.zeros(shape=(G.number_of_edges(), len(G.edges[0,1]['features'])))
  i=0
  for edge in G.edges:
    e[i]=np.asarray(G.edges[edge]['features'])    
    i=i+1
  return torch.from_numpy(e)


def getEdgeList(G):
  """
  Return the edge list in a Torch Geometric format
  The consists in a tensor of dimension (2, n_of_edges)
  in which each `column` encodes the two nodes that
  constitute an edge
  
  Args:
  G: networkX Graph object
  
  Returns
  torch tensor of dimension (2, n_of_edges)
  """
  return torch.from_numpy(np.asarray(G.edges())).t().contiguous().long()


def CreateTorchGraphs(data: pd.DataFrame,
                      global_features: List[str],
                      global_scale: np.ndarray,
                      node_scale: np.ndarray
) -> Tuple[List[Data], pd.DataFrame]:
    """
    Create a torch geometric Data object
    from a Pandas DataFrame
    
    Args:
    data: Data to use
    global_features: list of global features
    global_scale: scale of global features
    node_scaling: scale of node features

    Returns:
    graphs: List of torch Data objects
    DataFrame with only the needed data
    """
    print('Creating graph data...')  # Echo start of Graph creation

    # Empty lists to store data for each event
    graphs=[]
    globals=[]
    labels=[]
    weights=[]
    booking=[]
    i=0
    start = time.time() 

    # Iterate through each event
    for index, event in data.reset_index(drop=True).iterrows():
        i+=1

        # Echo status each 100 events
        if i%100==0 or i==len(data):
            elapsed = time.time()-start
            graphs_per_second = i/elapsed
            graphs_remaining = len(data)-i
            seconds_remaining = graphs_remaining/graphs_per_second
            print('\r{}/{} complete. Time elapsed: {:.1f}s,   Estimated time remaining: {:.1f}s'.format(i, len(data), elapsed, seconds_remaining), end='')

        # Create NetworkX Graph object from DataFrame row    
        G_nx = event2networkx(event, global_features, global_scale, node_scale)

        # Create PyTorch geometric Data from Networkx Graph
        x = getNodeFeatures(G_nx)
        edge_attr = getEdgeFeatures(G_nx)
        edge_index = getEdgeList(G_nx)
        G_geo = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                     y=torch.from_numpy(np.asarray(G_nx.graph['IsSig'])).float(), 
                     w=torch.from_numpy(np.asarray(G_nx.graph['sample_weight'])).float(),
                     u=torch.from_numpy(np.asarray(G_nx.graph['features'])).float().view(-1,len(G_nx.graph['features'])),
                     pseudo_mH=torch.tensor(G_nx.graph['pseudo_mH'])
                     );
        graphs.append(G_geo);
        booking.append(G_nx.graph['book_keeping']);
    print('\nDone')
    return graphs, pd.DataFrame(booking).reset_index(drop=True)



def plotGraph(G, node_scale, global_features):   
    color_map=[]
    pos={}
    pt_max=0
    #sum_pt=np.asarray([0,0])
    for node in G:
        pt=G.nodes[node]['features'][0]
        btag=G.nodes[node]['features'][4]
        if node==0:
            color_map.append(cm.bone(0))
        #else:
            #sum_pt=sum_pt+np.asarray([pt*np.cos(G.nodes[node]['pos'][1]), pt*np.sin(G.nodes[node]['pos'][1])])
        if node==1:
            color_map.append(cm.bwr(0))
        if node>1:
            color_map.append(cm.summer(btag))

    
        pt_max=max(pt,pt_max)
        pos[node]=(pt*np.cos(node_scale[1]*G.nodes[node]['features'][1]), pt*np.sin(node_scale[1]*G.nodes[node]['features'][1]))
        
    ax = plt.figure(figsize=(10, 10)).gca()
    
    t=2*np.pi*np.linspace(0,1,1000)
    
    plt.plot(pt_max*np.cos(t), pt_max*np.sin(t))
    plt.plot(.01*pt_max*np.cos(t), .01*pt_max*np.sin(t), 'red')
    
    nx.draw(G, ax=ax, pos=pos, node_color=color_map, alpha=0.7)


    print('Book-keeping variables:\n')
    for key in G.graph['book_keeping'].keys():
        print('{}: '.format(key),G.graph['book_keeping'][key])
    print('\n')

    global_dict=dict(zip(global_features, G.graph['features']))

    print('Global variables:\n')
    for key in global_dict.keys():
        print('{}: '.format(key), global_dict[key])
    print('\n')

    print('Sample weight: ', G.graph['sample_weight'])
    print('Pseudo-mass:', G.graph['pseudo_mH'])
    print('Target: ', G.graph['IsSig'])
    print('\n')
    for node in G:
        print(
              'Node: {}'.format(node),
              '\tpT: {:.4f}'.format(G.nodes[node]['features'][0]), 
              '\tphi: {:.4f}'.format(G.nodes[node]['features'][1]),
              '\teta: {:.4f}'.format(G.nodes[node]['features'][2]),
              '\tE: {:.4f}'.format(G.nodes[node]['features'][3]),
              '\tbtag: {:.1f}'.format(G.nodes[node]['features'][4]),
              '\tQ/G-BDT: {:.4f}'.format(G.nodes[node]['features'][5]),
              '\ttype: {}'.format(G.nodes[node]['features'][6]),
             )


from torch_geometric.data import Dataset
import pickle
import os

class customDataset(Dataset):
    def __init__(self, graphs=[], booking=[]):
        super(customDataset, self).__init__()
        self.graphs = graphs
        self.booking = booking
        self.n=10

    def save_to(self, path):
      if not os.path.exists(path):
          os.makedirs(path)
      remainder_n = len(self.booking) %  self.n
      len_int = int(len(self.booking)-remainder_n)
      for i in range(self.n):
        if i==self.n-1:
          end=len(self.booking)
        else:
          end=int((i*len_int+len_int)/self.n)
        start=int(i*len_int/self.n)
        self.booking.iloc[start:end].to_pickle('{}/booking_{}.pkl'.format(path,i))
        with open('{}/graphs_{}.pkl'.format(path, i), 'wb') as f:
            pickle.dump(self.graphs[start:end], f)
      
    def download_from(self, path):
      self.booking=pd.DataFrame()
      self.graphs=[]
      for i in range(self.n):
        print("Downloading file {}/{}...".format(i+1,self.n))
        self.booking=self.booking.append(pd.read_pickle('{}/booking_{}.pkl'.format(path, i)), ignore_index=True)
        with open('{}/graphs_{}.pkl'.format(path, i), 'rb') as f:
          self.graphs = self.graphs + pickle.load(f)
      print('Done')

    def len(self):
        return len(self.booking)

    def get(self, idx):
      return self.graphs[idx]
      
    def get_booking(self, idx):
      return self.booking.iloc[idx]


class Loader():

  def __init__(self, path, indices):
    self.path = path
    self.indices = indices
    self.index = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self.index >= len(self.indices):
      raise StopIteration
    else:
      index = self.index
      self.index += 1
      print(f'Opening fie {self.path}/graphs_{self.indices[index]}.pkl:')
      with open(f'{self.path}/graphs_{self.indices[index]}.pkl', 'rb') as f:
        data_set = pickle.load(f)
        return data_set