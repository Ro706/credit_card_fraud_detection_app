import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    return df

def create_graph(df, k=10):
    # Extract features and labels
    features = df.drop(columns=['Class', 'Time']).values  # Exclude 'Time' and 'Class'
    labels = df['Class'].values

    # Normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    # Convert to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Compute k-nearest neighbors (k-NN) to create edges
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Create edge_index from k-NN
    edge_index = []
    for i in range(len(indices)):
        for j in indices[i]:
            if i != j:  # Avoid self-loops
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create PyTorch Geometric data object
    graph = Data(x=x, edge_index=edge_index, y=y)
    return graph

def split_data(graph, train_ratio=0.8):
    # Split data into training and testing sets
    num_nodes = graph.x.shape[0]
    num_train = int(train_ratio * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[num_train:] = True

    graph.train_mask = train_mask
    graph.test_mask = test_mask
    return graph