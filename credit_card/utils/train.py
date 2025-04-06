import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

def train(model, graph, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def evaluate(model, graph):
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        pred = out.argmax(dim=1)
        print(classification_report(graph.y[graph.test_mask].numpy(), pred[graph.test_mask].numpy()))
        print("AUC-ROC:", roc_auc_score(graph.y[graph.test_mask].numpy(), out[graph.test_mask, 1].exp().numpy()))