import torch
from torch import nn
from torch.nn import functional as F


class GraphLoss(nn.Module):
    def __init__(self, base, alpha=1, beta=1, gamma=1):
        super(GraphLoss, self).__init__()
        self.base = base
        self.alpha = alpha 
        self.beta = beta
        self.gamma = gamma

    def forward(self, predictios, adj_matrix, labels, inputs):
        predictios = predictios.view(-1)
        labels = labels.view(-1)
        n = adj_matrix.size(2)

        base_loss = self.base(predictios, labels) # F.mse_loss
        
        #degree_matrix = adj_matrix.sum(dim=1)
        #laplacian = degree_matrix - adj_matrix
        #laplacian
        smoothness_term = torch.mean(torch.cdist(inputs, inputs, p=2).pow(2) * adj_matrix, dim=(1, 2)) / 2

        connectivity_term = - torch.log(adj_matrix.sum(dim=1)).mean(dim=1)

        sparsity_term = torch.norm(adj_matrix, dim=(1, 2)) / (n*n)

        graph_loss = self.alpha * smoothness_term + self.beta * connectivity_term + self.gamma * sparsity_term
        
        return base_loss + graph_loss.mean()
