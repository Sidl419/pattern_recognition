import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TopKPooling, GINConv, Sequential
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class FlexGCN(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_layers=2, classifier_width_ratio=4, num_classes=2):
        super(FlexGCN, self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        output_feat_dim = input_feat_dim // 2
        self.gc = GINConv(nn.Linear(input_feat_dim, output_feat_dim))
        self.act = nn.ReLU(True)
        input_feat_dim = output_feat_dim

        layers = []

        for _ in range(num_layers):
            output_feat_dim = output_feat_dim // 2
            layers.append((GINConv(nn.Linear(input_feat_dim, output_feat_dim)), 'x, edge_index -> x'))
            layers.append(nn.ReLU(True))
            layers.append((TopKPooling(output_feat_dim, 0.5), 'x, edge_index, None, batch -> x, edge_index, _, batch, _, _'))
            input_feat_dim = output_feat_dim

        self.body = Sequential('x, edge_index, batch', layers)

        self.hook = nn.Identity()
        self.classifier_input_size = output_feat_dim
        self.classifier = nn.Linear(self.classifier_input_size, num_classes, bias=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.act(self.gc(x, edge_index))
        
        #x = self.act3(self.gc2(x, edge_index))
        #x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x, edge_index, _, batch, _, _ = self.body(x, edge_index, batch)

        x = self.hook(gap(x, batch))
        x = self.classifier(x)
        
        return x


class BigGIN(nn.Module):
    def __init__(self, in_features, out_features, hidden, eps=0.):
        super(BigGIN, self).__init__()
        self.eps = eps
        #self.h = nn.Linear(in_features, out_features, bias=True)
        self.h = nn.Sequential(
                    nn.Linear(in_features, hidden, bias=True),
                    nn.ReLU(True),
                    nn.Linear(hidden, out_features, bias=True),
                    nn.ReLU(True)
                )
        #nn.Linear(hidden, out_features, bias=True),
        #nn.ReLU(True),

    def forward(self, input, adj):
        adj = adj + (1 + self.eps) * torch.eye(adj.shape[0], device=adj.device)
        output = torch.matmul(adj, input)
        output = self.h(output)
        return output


class SmallGIN(nn.Module):
    def __init__(self, in_features, out_features, adj, eps=0.):
        super(SmallGIN, self).__init__()
        self.eps = eps
        self.adj = adj
        self.h = nn.Linear(in_features, out_features, bias=True)

    def forward(self, input):
        adj = self.adj + (1 + self.eps) * torch.eye(self.adj.shape[0], device=input.device)
        output = torch.matmul(adj, input)
        output = self.h(output)
        return output


class TemporalConv(nn.Module):
    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * Residual Connection *
    #        |                                |
    #        |    |--->--- CasualConv2d ----- + -------|       
    # -------|----|                                   ⊙ ------>
    #             |--->--- CasualConv2d --- Sigmoid ---|                               
    #
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(x + self.conv1(x))
        output = self.batch_norm(x)
        return output


class STGCN_block(nn.Module):
    def __init__(self, adj, in_channels=42, num_nodes=64):
        super(STGCN_block, self).__init__()
        self.adj = adj

        self.temp1 = TemporalConv(num_nodes, num_nodes)
        self.gc = BigGIN(in_channels, in_channels, in_channels // 4)
        self.temp2 = TemporalConv(num_nodes, num_nodes)
        self.batch_norm = nn.BatchNorm1d(num_nodes)

    def forward(self, x):
        x = self.temp1(x)
        x = self.gc(x, self.adj)
        output = self.temp2(x)
        output =  self.batch_norm(x)
        return output


class STGCN(nn.Module):
    def __init__(self, adj, in_channels=42, num_nodes=64, num_classes=1):
        super(STGCN, self).__init__()

        self.block1 = STGCN_block(adj, in_channels, num_nodes)
        self.block2 = STGCN_block(adj, in_channels, num_nodes)

        self.temp = TemporalConv(num_nodes, num_nodes)
        self.hook = nn.Identity()
        self.classifier = nn.Linear(in_channels * num_nodes, num_classes, bias=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = self.temp(x)

        x = self.hook(torch.flatten(x, 1))
        x = self.classifier(x)

        return x


class BaseGNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels, adj, time_kernel=13, num_classes=2, channel_filters=1):
        super(BaseGNN, self).__init__()

        self.num_classes = num_classes

        self.gc = SmallGIN(input_feat_dim, input_feat_dim, adj)
        self.linear_channel = nn.Conv1d(n_channels, channel_filters, kernel_size=1, bias=True)
        self.conv = nn.Conv1d(channel_filters, 1, kernel_size=time_kernel, padding='same')
        self.bn1 = nn.BatchNorm1d(1)
        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(input_feat_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.gc(x)
        x = self.linear_channel(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.linear_output(x)
        
        return x
    

class BaseGNNBig(nn.Module):
    def __init__(self, input_feat_dim, n_channels, adj, time_kernel=13, num_classes=2, channel_filters=1):
        super(BaseGNNBig, self).__init__()

        self.num_classes = num_classes

        self.gc1 = SmallGIN(input_feat_dim, input_feat_dim, adj)
        self.gc2 = SmallGIN(input_feat_dim, input_feat_dim, adj)

        self.linear_channel = nn.Conv1d(n_channels, channel_filters, kernel_size=1, bias=True)
        
        self.conv = nn.Conv1d(channel_filters, 1, kernel_size=time_kernel, padding='same')
        self.bn1 = nn.BatchNorm1d(1)
        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(input_feat_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.gc1(x)
        x = self.gc2(x)
        x = self.linear_channel(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.linear_output(x)
        
        return x
