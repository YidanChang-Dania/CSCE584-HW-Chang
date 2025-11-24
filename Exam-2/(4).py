import argparse
import copy
import time
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import networkx as nx


from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx

warnings.filterwarnings('ignore')

# -----------------------------
# Reproducibility & Device
# -----------------------------
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
print(f"Using device: {device}")

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


# -----------------------------
# Dataset Info
# -----------------------------
class CiteseerDatasetInfo:
    @staticmethod
    def print_info():
        info = """
        ================================================================================
        CITESEER DATASET INFORMATION
        ================================================================================

        Dataset Type: Citation Network
        Task: Node Classification (Semi-supervised)

        GRAPH STRUCTURE:
        - Nodes: 3,327 scientific papers
        - Edges: 9,104 stored (bidirectional) edges 
  (≈4,552 unique undirected edges; treated as undirected, no self-loops)
        - Average degree: ~2.84

        NODE FEATURES:
        - Dimension: 3,703
        - Type: Sparse bag-of-words representation
        - Each feature represents presence/absence of specific words

        CLASSES (6):
        1. Agents - Multi-agent systems
        2. AI - Artificial Intelligence
        3. DB - Databases
        4. IR - Information Retrieval
        5. ML - Machine Learning  
        6. HCI - Human Computer Interaction

        DATA SPLIT (fixed):
        - Train: 120 nodes (20 per class)
        - Validation: 500 nodes
        - Test: 1,000 nodes

        TASK:
        - Semi-supervised transductive learning
        - Given: Full graph structure + features, only 120 labeled nodes
        - Goal: Predict labels for remaining nodes
        ================================================================================
        """
        print(info)


# -----------------------------
# FIXED Manual SGD with Momentum
# -----------------------------
class ManualSGDMomentum:
    """
    Manual implementation of SGD with momentum for better convergence.
    This is crucial for training GNNs without adaptive optimizers.
    """

    def __init__(self, model_parameters, learning_rate=0.2, weight_decay=5e-4, momentum=0.9):
        """
        Initialize manual SGD with momentum

        Args:
            model_parameters: Model parameters to optimize
            learning_rate: Learning rate (needs to be higher for plain SGD)
            weight_decay: L2 regularization coefficient
            momentum: Momentum coefficient for smoother updates
        """
        self.parameters = list(model_parameters)
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Initialize velocity (momentum) buffers for each parameter
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(torch.zeros_like(param.data))

    def zero_grad(self):
        """Zero out all gradients"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self):
        """
        Perform optimization step with momentum
        v = momentum * v - lr * (grad + weight_decay * param)
        param = param + v
        """
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue

                # Get gradient
                grad = param.grad.data

                # Add weight decay (L2 regularization)
                if self.weight_decay != 0:
                    grad = grad.add(param.data, alpha=self.weight_decay)

                # Update velocity with momentum
                v = self.velocities[i]
                v.mul_(self.momentum).add_(grad, alpha=-self.lr)

                # Update parameters
                param.data.add_(v)


# -----------------------------
# GNN Model Architecture
# -----------------------------
class GNNModel(nn.Module):
    """
    2-layer Graph Convolutional Network
    """

    def __init__(self, num_features: int, num_classes: int,
                 hidden_dim: int = 16, dropout_rate: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        print("\n" + "=" * 80)
        print("GNN MODEL ARCHITECTURE (2-layer GCN)")
        print("=" * 80)
        print(f"  Layer 1 (GCNConv): {num_features} -> {hidden_dim}")
        print(f"  Activation: ReLU")
        print(f"  Dropout: {dropout_rate}")
        print(f"  Layer 2 (GCNConv): {hidden_dim} -> {num_classes}")
        print(f"  Output: LogSoftmax")

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total Parameters: {total_params:,}")
        print("=" * 80 + "\n")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# -----------------------------
# Trainer with Fixed Manual Optimization
# -----------------------------
class Trainer:
    """
    Training class using manual gradient descent with momentum
    """

    def __init__(self, model, data, device, learning_rate=0.05, weight_decay=5e-4, momentum=0.9):
        """
        Initialize trainer with properly tuned hyperparameters for manual SGD

        Args:
            model: GNN model
            data: Graph data
            device: Computing device
            learning_rate: Higher learning rate needed for SGD (0.2 vs 0.01 for Adam)
            weight_decay: L2 regularization
            momentum: Momentum for better convergence
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device

        # Use manual SGD with momentum
        self.optimizer = ManualSGDMomentum(
            self.model.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum
        )

        self.train_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, mask):
        """Evaluate accuracy on specified nodes"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            pred = logits.argmax(dim=1)
            correct = pred[mask] == self.data.y[mask]
            acc = float(correct.sum()) / float(mask.sum())
        return acc

    def train(self, epochs=600, patience=200, verbose=True, ckpt_path="best_gnn.pt"):
        """
        Full training loop with early stopping
        Note: Reduced epochs and patience for faster convergence with momentum SGD
        """
        best_val_acc = 0
        best_state = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print("\n" + "=" * 80)
        print("TRAINING PROGRESS (Manual SGD with Momentum)")
        print("=" * 80)

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            self.train_losses.append(loss)

            train_acc = self.evaluate(self.data.train_mask)
            val_acc = self.evaluate(self.data.val_mask)
            self.val_accuracies.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Loss={loss:.4f}, "
                      f"Train={train_acc:.4f}, Val={val_acc:.4f}")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Load best model
        self.model.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved to {ckpt_path}")
        print("=" * 80 + "\n")

        return best_val_acc


# -----------------------------
# Visualization Functions
# -----------------------------
def visualize_training_history(train_losses, val_accuracies):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("Saved: training_history.png")


def visualize_graph_subset(data, num_nodes=150):
    """Visualize graph structure"""
    G = to_networkx(data, to_undirected=True)

    # Get non-isolated nodes with highest degree
    non_isolated = [(n, d) for n, d in G.degree() if d > 0]
    non_isolated.sort(key=lambda x: x[1], reverse=True)
    nodes = [n for n, _ in non_isolated[:num_nodes]]

    G_sub = G.subgraph(nodes)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_sub, seed=42, k=0.5, iterations=50)
    colors = [data.y[n].item() for n in G_sub.nodes()]

    nx.draw_networkx_nodes(G_sub, pos, node_color=colors,
                           cmap='tab10', node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G_sub, pos, edge_color='gray',
                           alpha=0.3, width=0.5)

    plt.title(f'Citeseer subgraph: top-{num_nodes} non-isolated nodes (by degree)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_structure.png', dpi=150)
    plt.show()
    print("Saved: graph_structure.png")


def visualize_embeddings(model, data, device):
    """t-SNE visualization of learned embeddings"""
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        # Get embeddings from first layer
        h = model.conv1(x, edge_index)
        h = F.relu(h)
        embeddings = h.cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot test nodes
    plt.figure(figsize=(10, 8))
    test_mask = data.test_mask.cpu().numpy()
    labels = data.y[test_mask].cpu().numpy()

    scatter = plt.scatter(emb_2d[test_mask, 0], emb_2d[test_mask, 1],
                          c=labels, cmap='tab10', s=30, alpha=0.7)
    plt.colorbar(scatter, ticks=range(6), label='Class')

    plt.title('t-SNE of Learned Node Embeddings (Test Set)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig('node_embeddings.png', dpi=150)
    plt.show()
    print("Saved: node_embeddings.png")


def plot_confusion_matrix(model, data, device):
    """Generate confusion matrix (force everything on CPU to be safe)."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        pred = logits.argmax(dim=1).cpu()  # CPU
    y_all = data.y.cpu()  # CPU
    test_mask = data.test_mask.cpu().numpy()  # numpy bool mask

    y_true = y_all.numpy()[test_mask]
    y_pred = pred.numpy()[test_mask]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    class_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label');
    plt.xlabel('Predicted Label')
    plt.tight_layout();
    plt.savefig('confusion_matrix.png', dpi=150);
    plt.show()
    print("Saved: confusion_matrix.png")


def run_all_visualizations(model, data, device, trainer):
    """Run all visualizations"""
    print("\nGenerating visualizations...")
    visualize_training_history(trainer.train_losses, trainer.val_accuracies)
    visualize_graph_subset(data, num_nodes=150)
    visualize_embeddings(model, data, device)
    plot_confusion_matrix(model, data, device)
    print("All visualizations complete!\n")


# -----------------------------
# Main Pipeline
# -----------------------------
def main(mode='all', checkpoint='best_gnn_citeseer.pt'):
    """
    Main execution pipeline

    Args:
        mode: 'train', 'eval', or 'all'
        checkpoint: Path for model checkpoint
    """
    # Print dataset info
    CiteseerDatasetInfo.print_info()

    # Load dataset
    print("Loading Citeseer dataset...")
    dataset = Planetoid(root="./data/Citeseer", name="Citeseer",
                        transform=NormalizeFeatures())
    data = dataset[0]

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_features:,}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Train/Val/Test: {int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")
    print(f"  Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"  Is undirected: {data.is_undirected()}")

    # Initialize model
    print("\nInitializing GNN model...")
    model = GNNModel(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=16,
        dropout_rate=0.5
    )

    # Training configuration (tuned for manual SGD)
    config = {
        'learning_rate': 0.05,  # from 0.2 ↓ to 0.05 (necessary)
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'epochs': 600,  # from 200 ↑ to 600 (SGD needs longer)
        'patience': 200,  # from 50 ↑ to 200 (avoid premature stop)
    }

    print("\nTraining Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        data=data,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        momentum=config['momentum']
    )

    # Training or evaluation
    if mode in ['train', 'all']:
        print("\nStarting training...")
        start_time = time.time()

        best_val = trainer.train(
            epochs=config['epochs'],
            patience=config['patience'],
            verbose=True,
            ckpt_path=checkpoint
        )

        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")

    elif mode == 'eval':
        print(f"\nLoading model from {checkpoint}...")
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.to(device)

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    train_acc = trainer.evaluate(data.train_mask)
    val_acc = trainer.evaluate(data.val_mask)
    test_acc = trainer.evaluate(data.test_mask)

    print(f"Training Accuracy:   {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Test Accuracy:       {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Per-class accuracy
    print("\nPer-Class Test Accuracy:")
    class_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']

    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        pred = logits.argmax(dim=1).cpu()

    y = data.y.cpu()
    test_mask = data.test_mask.cpu()

    for i, name in enumerate(class_names):
        mask = ((y == i) & test_mask)
        if mask.sum().item() > 0:
            acc = (pred[mask] == y[mask]).float().mean().item()
            count = mask.sum().item()
            print(f"  {name:6s}: {acc:.4f} ({acc * 100:.2f}%) - {count:3d} samples")

    print("=" * 80 + "\n")

    # Visualizations
    if mode == 'all':
        run_all_visualizations(model, data, device, trainer)

    return model, data, test_acc


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Citeseer GNN with Manual Optimization (Fixed Version)")
    parser.add_argument("--mode", choices=["train", "eval", "all"],
                        default="all",
                        help="Mode: train only, eval only, or train+eval+viz")
    parser.add_argument("--checkpoint", default="best_gnn_citeseer.pt",
                        help="Checkpoint file path")

    args = parser.parse_args()

    print("=" * 80)
    print("CITESEER GRAPH NEURAL NETWORK")
    print("Manual SGD Implementation (No torch.optim)")
    print("=" * 80)

    model, data, test_acc = main(mode=args.mode, checkpoint=args.checkpoint)

    print(f"\n{'=' * 80}")
    print(f"FINAL TEST ACCURACY: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"{'=' * 80}\n")
