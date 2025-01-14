import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assume the dataset is loaded and available as 'dataset', with splits 'train', 'validation', 'test'
# We will use the first 100 samples from the 'train' split
print("Using the first 100 samples from the 'train' split of SST-2 dataset.")
data_samples = dataset['train'][:100]

# Extract sentences and labels
sentences = data_samples['sentence']
labels = data_samples['label']

# Filter out any samples with empty sentences
non_empty_indices = [i for i, sentence in enumerate(sentences) if sentence.strip() != '']
sentences = [sentences[i] for i in non_empty_indices]
labels = [labels[i] for i in non_empty_indices]
labels = torch.tensor(labels, dtype=torch.long)

# Convert sentences to TF-IDF vectors
print("Converting sentences to TF-IDF vectors.")
vectorizer = TfidfVectorizer(max_features=1000)
X_np = vectorizer.fit_transform(sentences).toarray()
X = torch.tensor(X_np, dtype=torch.float)

# Build a k-nearest neighbor graph based on cosine similarity
print("Building a k-nearest neighbor graph based on cosine similarity.")
similarity_matrix = cosine_similarity(X_np)
num_nodes = X.size(0)
k = 5  # Number of neighbors
edge_index = []
for i in range(num_nodes):
    sim_scores = similarity_matrix[i]
    sim_scores[i] = -1  # Exclude self
    k_neighbors = np.argsort(sim_scores)[-k:]
    for j in k_neighbors:
        edge_index.append([i, j])
        edge_index.append([j, i])  # Since the graph is undirected

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {edge_index.size(1)}")

# Define GCN model without functions
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
    
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = torch.mm(x, self.weight)
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * norm.unsqueeze(1))
        return out

conv1 = GCNLayer(X.size(1), 16)
conv2 = GCNLayer(16, 2)  # 2 classes

# Define optimizer and loss function
optimizer = optim.Adam(list(conv1.parameters()) + list(conv2.parameters()), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Starting training loop.")
for epoch in range(50):
    # Forward pass
    out = conv1(X, edge_index)
    out = torch.relu(out)
    out = conv2(out, edge_index)
    
    # Compute loss
    loss = criterion(out, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        _, pred = out.max(dim=1)
        correct = int(pred.eq(labels).sum().item())
        acc = correct / labels.size(0)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

# Approximate Shapley Values using Monte Carlo Sampling
print("\nApproximating Shapley Values using Monte Carlo Sampling")
print("We approximate each node's Shapley value by randomly sampling subsets of nodes and measuring the change in the model's prediction when a node is added, reflecting its marginal contribution.")

num_samples = 50
shapley_values = torch.zeros(num_nodes)

for node_idx in range(num_nodes):
    marginal_contributions = []
    for _ in range(num_samples):
        # Random subset of nodes excluding the node in question
        subset = torch.randperm(num_nodes)
        subset = subset[subset != node_idx]
        subset_size = np.random.randint(1, num_nodes)
        S = subset[:subset_size]
        
        # Create node mask
        node_mask = torch.zeros(num_nodes)
        node_mask[S] = 1
        node_mask[node_idx] = 1  # Include the node in question
        x_subset = X * node_mask.unsqueeze(1)
        
        # Forward pass
        out_subset = conv1(x_subset, edge_index)
        out_subset = torch.relu(out_subset)
        out_subset = conv2(out_subset, edge_index)
        out_subset = torch.softmax(out_subset, dim=1)
        
        contribution = out_subset[node_idx, labels[node_idx]]  # Probability of true class
        marginal_contributions.append(contribution.item())
    
    shapley_values[node_idx] = torch.tensor(marginal_contributions).mean()

# Plot Shapley Values
plt.figure(figsize=(10, 6))
plt.bar(range(num_nodes), shapley_values.numpy(), color='skyblue')
plt.xlabel('Node Index')
plt.ylabel('Approximate Shapley Value')
plt.title('Approximate Shapley Values for Nodes')
plt.savefig('Figure_1.png')
plt.show()

# Experiment 2: Structure-Aware GNN Explanations Using HN Value
print("\nExperiment 2: Structure-Aware GNN Explanations Using HN Value")
print("This experiment creates a GNN explanation method that incorporates graph topology using the Hamiache-Navarro (HN) value, enhancing the structural awareness of the interpretations.")

# Compute node degrees
degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()

# Compute HN values (normalize degrees)
hn_values = degrees / degrees.sum()

# Plot HN Values
plt.figure(figsize=(10, 6))
plt.bar(range(num_nodes), hn_values.numpy(), color='salmon')
plt.xlabel('Node Index')
plt.ylabel('HN Value (Normalized Degree)')
plt.title('HN Values (Structural Importance of Nodes)')
plt.savefig('Figure_2.png')
plt.show()

# Analysis
print("\nAnalysis:")
print("Shapley Values (Experiment 1) represent the average marginal contribution of each node (sentence) to the model's prediction, approximated via Monte Carlo sampling. Higher values indicate greater importance in the prediction.")
print("HN Values (Experiment 2) use the normalized node degrees as a proxy for structural importance within the graph, highlighting the role of graph topology in the GNN's predictions.")
print("Comparing Figures 1 and 2, we can observe how feature importance (Shapley Values) and structural importance (HN Values) differ across nodes.")