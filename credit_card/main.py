import torch
import torch.optim as optim
from utils.data_loader import load_data, create_graph, split_data
from models.gnn_model import FraudDetectionGNN
from utils.train import train, evaluate

def main():
    # Step 1: Load and preprocess the dataset
    print("Loading dataset...")
    df = load_data("data/creditcard.csv")  # Load the dataset
    print("Dataset loaded successfully!")

    # Step 2: Create the graph
    print("Creating graph...")
    graph = create_graph(df)  # Construct the graph
    graph = split_data(graph)  # Split into training and testing sets
    print("Graph created successfully!")

    # Step 3: Initialize the GNN model
    print("Initializing model...")
    model = FraudDetectionGNN(num_features=graph.x.shape[1], hidden_channels=16, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("Model initialized successfully!")

    # Step 4: Train the model
    print("Training model...")
    train(model, graph, optimizer, epochs=100)
    print("Training completed!")

    # Step 5: Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), "models/fraud_detection_model.pth")
    print("Model saved successfully!")

    # Step 6: Evaluate the model
    print("Evaluating model...")
    evaluate(model, graph)
    print("Evaluation completed!")

if __name__ == "__main__":
    main()