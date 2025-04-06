import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FraudDetectionGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(FraudDetectionGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)