from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Dropout, Tanh
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import degree
import torch


class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, n_global_feats, n_edge_feats_out):
        super(EdgeModel, self).__init__()

        # update edges
        self.edge_mlp = Seq(Lin(2*n_node_feats+n_edge_feats+n_global_feats, 32),
                            #ReLU(), 
                            #Dropout(p=0.1),
                            #Lin(32, 32),
                            ReLU(),
                            Dropout(p=0.1),
                            Lin(32, n_edge_feats_out),
                            ReLU())

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1).float()
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, n_global_feats, message_feats, n_node_feats_out):
        super(NodeModel, self).__init__()

        # message creation
        self.node_mlp_1 = Seq(Lin(2*n_node_feats+n_edge_feats, 32), 
                              #ReLU(), 
                              #Dropout(p=0.1),
                              #Lin(32, 32),
                              ReLU(),
                              Dropout(p=0.1),
                              Lin(32, message_feats),
                              ReLU())
        
        # update node with message
        self.node_mlp_2 = Seq(Lin(2*message_feats+n_node_feats+n_global_feats, 32), 
                              #ReLU(),
                              #Dropout(p=0.1),
                              #Lin(32, 32),
                              ReLU(),
                              Dropout(p=0.1), 
                              Lin(32, n_node_feats_out),
                              ReLU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], 1).float()
        out = self.node_mlp_1(out) # message
        agg_mean = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        agg_max = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out = torch.cat([x, agg_mean, agg_max, u[batch]], dim=1).float()
        return self.node_mlp_2(out) # update node with message

class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_feats, n_global_feats, n_global_feats_out):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(n_global_feats+n_node_feats*2, 32), 
                              ReLU(), 
                              Dropout(p=0.1),
                              #Lin(32, 32),
                              #ReLU(),
                              #Dropout(p=0.1),
                              Lin(32, n_global_feats_out),
                              ReLU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0), scatter_max(x, batch, dim=0)[0]], dim=1).float()
        return self.global_mlp(out)


class GeneralMPGNN(torch.nn.Module):
    def __init__(self, node_feats_in, edge_feats_in, global_feats_in):
      super(GeneralMPGNN, self).__init__()
      

      self.mp_layer1 = MetaLayer(EdgeModel(n_node_feats=node_feats_in, n_edge_feats=edge_feats_in, n_global_feats=global_feats_in, n_edge_feats_out=6), 
                                NodeModel(n_node_feats=node_feats_in, n_edge_feats=6, n_global_feats=global_feats_in, message_feats=12, n_node_feats_out=12), 
                                GlobalModel(n_node_feats=12,  n_global_feats=global_feats_in, n_global_feats_out=9))
      
      self.mp_layer2 = MetaLayer(EdgeModel(n_node_feats=12, n_edge_feats=6, n_global_feats=9, n_edge_feats_out=6), 
                                NodeModel(n_node_feats=12, n_edge_feats=6, n_global_feats=9, message_feats=12, n_node_feats_out=12), 
                                GlobalModel(n_node_feats=12,  n_global_feats=9, n_global_feats_out=9))
      
      self.mp_layer3 = MetaLayer(EdgeModel(n_node_feats=12, n_edge_feats=6, n_global_feats=9, n_edge_feats_out=6), 
                                NodeModel(n_node_feats=12, n_edge_feats=6, n_global_feats=9, message_feats=12, n_node_feats_out=12), 
                                GlobalModel(n_node_feats=12,  n_global_feats=9, n_global_feats_out=9))
      

      self.fc = Seq(Lin(9+2*12,32), 
                    ReLU(),
                    Dropout(p=0.1),
                    #Lin(32,32), 
                    #ReLU(),
                    #Dropout(p=0.1),
                    Lin(32, 1),
                    Sigmoid())

    def forward(self, x, edge_index, edge_attr, u, param, batch):
      u_new = torch.cat([u, param.view(-1,1)], axis=1)
      x_new, edge_attr_new, u_new = self.mp_layer1(x, edge_index, edge_attr, u_new, batch)
      x_new, edge_attr_new, u_new = self.mp_layer2(x_new, edge_index, edge_attr_new, u_new, batch)
      x_new, edge_attr_new, u_new = self.mp_layer3(x_new, edge_index, edge_attr_new, u_new, batch)
      x_mean_pool = global_mean_pool(x_new, batch)
      x_max_pool = global_max_pool(x_new, batch)
      out = torch.cat([u_new, x_mean_pool, x_max_pool], dim=1).float()
      out = self.fc(out)
      return out
