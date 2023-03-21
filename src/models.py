import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch_geometric.nn import TopKPooling, GINConv, Sequential
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCNModel(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_layers=2, classifier_width_ratio=4, num_classes=2):
        super(GCNModel, self).__init__()

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


class Small_CNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_layers=2, num_classes=2):
        super(Small_CNN, self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        input_channels = n_channels
        output_channels = input_channels * 2
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(True)
        input_channels = output_channels
        output_feat_dim = input_feat_dim

        layers = []

        for _ in range(num_layers):
            output_channels = input_channels * 2
            layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool1d(2, stride=2))
            input_channels = output_channels
            output_feat_dim = output_feat_dim // 2

        self.body = nn.Sequential(*layers)
        self.hook = nn.Identity()

        self.classifier_input_size = output_channels * output_feat_dim
        self.classifier = nn.Linear(self.classifier_input_size, num_classes, bias=True)

    def forward(self, x):
        x = self.act(self.conv(x))

        x = self.body(x)

        x = self.hook(torch.flatten(x, 1))
        x = self.classifier(x)
        
        return x


class MyGIN(nn.Module):
    def __init__(self, in_features, out_features, hidden, eps=0.):
        super(MyGIN, self).__init__()
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
        self.gc = MyGIN(in_channels, in_channels, in_channels // 4)
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


class EEGNet(nn.Module):
    def __init__(self, in_channels=64, sample_len=72, num_classes=2):
        super(EEGNet, self).__init__()
        self.sample_len = sample_len
        self.in_channels = in_channels

        F1 = 8
        F2 = 16
        D = 2
        
        self.conv1 = nn.Conv2d(1, F1, (1, sample_len // 2), padding = 'same')
        self.bn1 = nn.BatchNorm2d(F1, False)

        self.conv2 = nn.Conv2d(F1, D * F1, (in_channels, 1), groups=F1)
        self.bn2 = nn.BatchNorm2d(D * F1, False)
        self.pool1 = nn.AvgPool2d(1, 4)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        
        self.conv3 = nn.Conv2d(D * F1, F2, (1, D), groups=D * F1, padding = 'same')
        self.bn3 = nn.BatchNorm2d(D * F1, False)
        self.pool2 = nn.AvgPool2d(1, 8)
        self.hook = nn.Dropout(p=0.5, inplace=False)
        
        self.classifier_input_size = 72 // 24 * F2
        self.classifier = nn.Linear(self.classifier_input_size, num_classes, bias=True)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.elu(self.bn2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = F.elu(self.bn3(x))
        x = torch.flatten(self.pool2(x), 1)
        x = self.hook(x)
        
        x = self.classifier(x)
        return x

def scaled_tanh(z):
    return 1.7159 * torch.tanh((2.0 / 3.0) * z)

class CNN1(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_layers=2, num_classes=2):
        super(CNN1, self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(n_channels, 10, kernel_size=1)
        self.conv2 = nn.Conv1d(10, 50, kernel_size=13, padding = 'same')

        self.classifier_input_size = input_feat_dim * 50
        self.hook = nn.Linear(self.classifier_input_size, 100, bias=True)
        self.classifier = nn.Linear(100, num_classes, bias=True)

    def forward(self, x):
        x = scaled_tanh(self.conv1(x))
        x = scaled_tanh(self.conv2(x))

        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.classifier(x)
        
        return x

class base_CNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, time_kernel=13, num_classes=2, channel_filters=1):
        super(base_CNN, self).__init__()

        self.num_classes = num_classes

        self.linear_channel = nn.Conv1d(n_channels, channel_filters, kernel_size=1, bias=True)
        #self.bn1 = nn.BatchNorm1d(1)

        self.conv = nn.Conv1d(channel_filters, 1, kernel_size=time_kernel, padding='same')
        self.bn2 = nn.BatchNorm1d(1)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(input_feat_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.linear_channel(x)
        #x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.linear_output(x)
        
        return x

class base_GIN(nn.Module):
    def __init__(self, in_features, out_features, adj, eps=0.):
        super(base_GIN, self).__init__()
        self.eps = eps
        self.adj = adj
        self.h = nn.Linear(in_features, out_features, bias=True)

    def forward(self, input):
        adj = self.adj + (1 + self.eps) * torch.eye(self.adj.shape[0], device=input.device)
        output = torch.matmul(adj, input)
        output = self.h(output)
        return output

class base_GNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels, adj, time_kernel=13, num_classes=2, channel_filters=1):
        super(base_GNN, self).__init__()

        self.num_classes = num_classes

        self.gc = base_GIN(input_feat_dim, input_feat_dim, adj)

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
