"""
Graph Convolutional Network (GCN) for Link Prediction — Complete Implementation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os
import matplotlib.pyplot as plt
from typing import Tuple

# ============================================================================
# Part 1: Model parameter definitions
# ============================================================================

class GCNLayer(nn.Module):
    """
    Single GCN layer

    Equation: H^(l+1) = σ(Ã H^(l) W^(l) + b^(l))

    Parameters:
        W^(l) ∈ R^(d_in × d_out): weight matrix
        b^(l) ∈ R^(d_out): bias vector
    """
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix W
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Bias vector b
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: node features [n, d_in]
            adj_norm: normalized adjacency Ã [n, n]

        Returns:
            output: [n, d_out]
        """
        # H^(l) W^(l)
        support = torch.mm(x, self.weight)
        # Ã H^(l) W^(l)
        output = torch.sparse.mm(adj_norm, support)
        # Add bias
        output = output + self.bias
        return output


class GCNEncoder(nn.Module):
    """
    Two-layer GCN encoder

    Architecture:
        Input (d_in) → GCN + ReLU (d_1) → GCN (d_2) → Output

    Chosen dimensions:
        d_in: input feature dimension
        d_1 = 64: hidden dimension (layer 1)
        d_2 = 32: embedding dimension (layer 2)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32):
        super(GCNEncoder, self).__init__()

        # Layer 1: d_in → d_1
        self.gc1 = GCNLayer(input_dim, hidden_dim)
        # Layer 2: d_1 → d_2
        self.gc2 = GCNLayer(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Equations:
            Z^(1) = ReLU(Ã X W^(1) + b^(1))
            Z^(2) = Ã Z^(1) W^(2) + b^(2)

        Args:
            x: node features [n, d_in]
            adj_norm: normalized adjacency [n, n]

        Returns:
            Z^(2): node embeddings [n, d_2]
        """
        # First layer
        z1 = self.gc1(x, adj_norm)
        z1 = F.relu(z1)

        # Second layer
        z2 = self.gc2(z1, adj_norm)

        return z2


class InnerProductDecoder(nn.Module):
    """
    Inner-product decoder for link prediction

    Equation: score_uv = z_u^T z_v

    Note: sigmoid is applied inside the loss (BCEWithLogitsLoss), not here.
    """
    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Vectorized decoder (computes all edge scores in one shot)

        Args:
            z: node embeddings [n, d_2]
            edge_index: edge indices [2, num_edges]

        Returns:
            scores: edge logits [num_edges]
        """
        # Source and target indices
        src, dst = edge_index[0], edge_index[1]

        # Vectorized inner product: z_u^T z_v = sum(z_u * z_v)
        scores = (z[src] * z[dst]).sum(dim=1)

        return scores


class GCNLinkPredictor(nn.Module):
    """
    Full GCN model for link prediction

    Parameter summary:
        - W^(1) ∈ R^(d_in × 64)
        - b^(1) ∈ R^64
        - W^(2) ∈ R^(64 × 32)
        - b^(2) ∈ R^32
    """

    # ==============================
    # Handwritten SGD + Momentum (no torch.optim)
    # ==============================
    class ManualSGDMomentum:
        def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
            # Keep weight_decay=0.0 here; explicit L2 is added in compute_loss
            self.params = list(params)
            self.lr = lr
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.v = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]

        def zero_grad(self):
            for p in self.params:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

        def step(self):
            with torch.no_grad():
                for i, p in enumerate(self.params):
                    if p.grad is None:
                        continue
                    g = p.grad
                    # If L2 is desired here, use: g = g + self.weight_decay * p
                    self.v[i].mul_(self.momentum).add_(g, alpha=1.0)
                    p.add_(self.v[i], alpha=-self.lr)


    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = 32):
        super(GCNLinkPredictor, self).__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim)
        self.decoder = InnerProductDecoder()

    def encode(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Return node embeddings"""
        return self.encoder(x, adj_norm)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return edge logits"""
        return self.decoder(z, edge_index)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass

        Returns:
            scores: edge logits (pre-sigmoid)
        """
        z = self.encode(x, adj_norm)
        scores = self.decode(z, edge_index)
        return scores


# ============================================================================
# Part 2: Data setup (link prediction task)
# ============================================================================
# ==============================
# Handwritten SGD + Momentum (no torch.optim)
# ==============================
class ManualSGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        # L2 is handled explicitly in compute_loss; keep 0 here to avoid duplication
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                # Optional L2 here: g = g + self.weight_decay * p
                self.v[i].mul_(self.momentum).add_(g, alpha=1.0)
                p.add_(self.v[i], alpha=-self.lr)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Normalize the adjacency matrix

    Equation: Ã = D^(-1/2) (A + I) D^(-1/2)

    Where:
        A: original adjacency
        I: identity (self-loops)
        D: degree matrix

    Args:
        adj: adjacency [n, n]

    Returns:
        adj_norm: normalized adjacency (sparse tensor)
    """
    # Add self-loops: Â = A + I
    adj = adj + torch.eye(adj.size(0), device=adj.device)

    # Degree: d_i = Σ_j Â_ij
    degree = adj.sum(1)

    # D^(-1/2)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

    # D^(-1/2) Â D^(-1/2)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # Convert to sparse tensor to save memory
    adj_normalized = adj_normalized.to_sparse()

    return adj_normalized


def negative_sampling(pos_edge_index: torch.Tensor, num_nodes: int,
                      num_neg_samples: int = None) -> torch.Tensor:
    """
    Negative sampling: generate nonexistent edges as negatives

    For link prediction, negatives are node pairs without edges.
    Balanced positives/negatives help avoid biased training.

    Args:
        pos_edge_index: positive edges [2, num_pos]
        num_nodes: number of nodes
        num_neg_samples: number of negatives (defaults to number of positives)

    Returns:
        neg_edge_index: negative edges [2, num_neg]
    """
    if num_neg_samples is None:
        num_neg_samples = pos_edge_index.size(1)

    # Use a set for O(1) membership checks
    pos_edges_set = set(map(tuple, pos_edge_index.t().tolist()))

    neg_edges = []
    attempts = 0
    max_attempts = num_neg_samples * 100  # avoid infinite loops

    while len(neg_edges) < num_neg_samples and attempts < max_attempts:
        # Sample two nodes
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)

        # Conditions: (1) not a self-loop, (2) not a positive edge in either order
        if src != dst and (src, dst) not in pos_edges_set and (dst, src) not in pos_edges_set:
            neg_edges.append([src, dst])

        attempts += 1

    if len(neg_edges) < num_neg_samples:
        print(f"Warning: generated only {len(neg_edges)}/{num_neg_samples} negative samples")

    return torch.tensor(neg_edges, dtype=torch.long).t()


def create_link_prediction_data(num_nodes: int = 100, num_features: int = 10,
                                edge_prob: float = 0.15) -> Tuple:
    """
    Create synthetic data for link prediction

    Includes:
        - Graph G = (V, E)
        - Node feature matrix X ∈ R^(n × d_in)
        - Train/validation/test edge splits
        - Positive and negative samples

    Args:
        num_nodes: number of nodes n
        num_features: feature dimension d_in
        edge_prob: edge probability

    Returns:
        x, adj, train_data, val_data, test_data
    """
    # Node feature matrix X (requires_grad not needed)
    x = torch.randn(num_nodes, num_features)

    # Adjacency and edge list
    adj = torch.zeros(num_nodes, num_nodes)
    edges = []

    # Random undirected graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])

    if len(edges) == 0:
        raise ValueError("No edges were generated; increase edge_prob")

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    num_edges = edge_index.size(1)

    # 80% / 10% / 10% split
    perm = torch.randperm(num_edges)
    num_train = int(0.8 * num_edges)
    num_val = int(0.1 * num_edges)

    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]

    train_pos = edge_index[:, train_idx]
    val_pos = edge_index[:, val_idx]
    test_pos = edge_index[:, test_idx]

    # Negative samples for each split
    train_neg = negative_sampling(train_pos, num_nodes, 2 * train_pos.size(1))
    val_neg = negative_sampling(val_pos, num_nodes, val_pos.size(1))
    test_neg = negative_sampling(test_pos, num_nodes, test_pos.size(1))

    train_data = (train_pos, train_neg)
    val_data = (val_pos, val_neg)
    test_data = (test_pos, test_neg)

    return x, adj, train_data, val_data, test_data


# ============================================================================
# Part 3: Loss definition
# ============================================================================

def compute_loss(model: nn.Module, x: torch.Tensor, adj_norm: torch.Tensor,
                pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor,
                lambda_reg: float = 5e-4) -> torch.Tensor:
    """
    Loss = BCEWithLogits + explicit L2 weight decay

    Objective:
        L = L_BCE + λ/2 * (||W^(1)||² + ||W^(2)||²)

    Where:
        L_BCE = -1/|E| Σ [y_e log(σ(score_e)) + (1−y_e) log(1−σ(score_e))]
        λ: weight decay coefficient

    Why BCEWithLogitsLoss:
        • Numerically stable (combines sigmoid and log)
        • Avoids manual sigmoid under/overflow issues

    Args:
        model: GCN model
        x: node features
        adj_norm: normalized adjacency
        pos_edge_index: positive edges
        neg_edge_index: negative edges
        lambda_reg: L2 regularization coefficient

    Returns:
        total_loss: scalar loss
    """
    # Merge positive and negative edges
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_labels = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).to(x.device)

    # Forward (logits)
    edge_scores = model(x, adj_norm, edge_index)

    # BCE loss with logits
    bce_loss = F.binary_cross_entropy_with_logits(edge_scores, edge_labels)

    # Explicit L2 on weights (bias excluded)
    l2_reg = torch.tensor(0., device=x.device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg += torch.sum(param ** 2)

    total_loss = bce_loss + (lambda_reg / 2) * l2_reg
    return total_loss


# ============================================================================
# Part 4: Training and evaluation
# ============================================================================

def train_epoch(model: nn.Module, optimizer,
               x: torch.Tensor, adj_norm: torch.Tensor,
               pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor,
               lambda_reg: float = 5e-4) -> float:
    """
    One training epoch with handwritten SGD + Momentum
      1) optimizer.zero_grad()
      2) loss.backward()
      3) optimizer.step()
    """
    model.train()
    optimizer.zero_grad()
    loss = compute_loss(model, x, adj_norm, pos_edge_index, neg_edge_index, lambda_reg)
    loss.backward()
    optimizer.step()
    return loss.item()




@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, adj_norm: torch.Tensor,
            pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> Tuple[float, float]:
    """
    Evaluation (logits → probabilities via sigmoid)

    Metrics:
        - ROC-AUC
        - Average Precision (AP)

    Args:
        model: GCN model
        x: node features
        adj_norm: normalized adjacency
        pos_edge_index: positive edges
        neg_edge_index: negative edges

    Returns:
        (auc, ap)
    """
    model.eval()

    # Merge edges and labels
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_labels = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ]).numpy()

    # Predict
    edge_scores = model(x, adj_norm, edge_index)
    edge_probs = torch.sigmoid(edge_scores).cpu().numpy()

    auc_score = roc_auc_score(edge_labels, edge_probs)
    ap_score = average_precision_score(edge_labels, edge_probs)

    return auc_score, ap_score


# ============================================================================
# Part 5: Visualize gradients (backprop view)
# ============================================================================

def visualize_gradients(model: nn.Module):
    """
    Show gradient norm and mean for each parameter.
    This reflects ∂L/∂θ after backprop and guides the update scale.
    """
    print("\n" + "=" * 80)
    print("Backpropagation Gradient Summary")
    print("=" * 80)
    print(f"{'Parameter':<35} {'Shape':<20} {'Grad-Norm':<15} {'Grad-Mean':<15}")
    print("-" * 80)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            print(f"{name:<35} {str(list(param.shape)):<20} "
                  f"{grad_norm:<15.6f} {grad_mean:<15.6e}")
        else:
            print(f"{name:<35} {str(list(param.shape)):<20} {'<no grad>':<15}")

    print("=" * 80)
    print("Notes:")
    print("- Grad-Norm ≈ ||∂L/∂θ|| indicates update magnitude")
    print("- Grad-Mean is the average sign/scale of gradients")
    print("- These gradients are used by the handwritten SGD+Momentum updates")
    print("=" * 80)


# ============================================================================
# Part 6: Main training pipeline
# ============================================================================

def main():
    """
    Main pipeline

    Walkthrough:
        1) Prepare data
        2) Initialize model
        3) Train
        4) Inspect gradients
        5) Evaluate
    """
    print("=" * 80)
    print("Graph Convolutional Network (GCN) for Link Prediction — Complete Implementation")
    print("=" * 80)
    print("\nThis script covers all required items:")
    print("✓ 1) Parameter definitions with dimensions")
    print("✓ 2) Data setup for link prediction")
    print("✓ 3) Forward equations (two-layer GCN)")
    print("✓ 4) Loss (BCE with Logits + L2)")
    print("✓ 5) Backprop via autograd with gradient display")
    print("✓ 6) Parameter updates via handwritten SGD + Momentum")
    print("\nFixes made:")
    print("✓ Negative sampling (essential)")
    print("✓ Vectorized decoder (efficiency)")
    print("✓ Removed requires_grad on X (correct)")
    print("✓ Explicit L2 regularization (matches derivation)")
    print("✓ BCEWithLogitsLoss (stable)")
    print("=" * 80)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # ========================================================================
    # 1) Data preparation
    # ========================================================================
    print("\n[Step 1] Data preparation")
    print("-" * 80)

    num_nodes = 100
    num_features = 10

    x, adj, train_data, val_data, test_data = create_link_prediction_data(
        num_nodes=num_nodes,
        num_features=num_features,
        edge_prob=0.15
    )

    train_pos, train_neg = train_data
    val_pos, val_neg = val_data
    test_pos, test_neg = test_data

    print(f"Nodes: {num_nodes}")
    print(f"Feature dimension: {num_features}")
    print(f"Total edges (pos only): {train_pos.size(1) + val_pos.size(1) + test_pos.size(1)}")
    print("\nSplit:")
    print(f"  Train  — positives: {train_pos.size(1)}, negatives: {train_neg.size(1)}")
    print(f"  Val    — positives: {val_pos.size(1)}, negatives: {val_neg.size(1)}")
    print(f"  Test   — positives: {test_pos.size(1)}, negatives: {test_neg.size(1)}")

    # Normalized adjacency
    adj_norm = normalize_adjacency(adj)
    print("\nNormalized adjacency Ã = D^(-1/2)(A+I)D^(-1/2)")
    print(f"  Shape: {adj_norm.size()}")
    print(f"  Sparsity: {adj_norm._nnz() / (num_nodes ** 2) * 100:.2f}%")

    # ========================================================================
    # 2) Model initialization
    # ========================================================================
    print("\n[Step 2] Model initialization")
    print("-" * 80)

    hidden_dim = 64
    embedding_dim = 32

    print("Architecture:")
    print(f"  Input dimension:   d_in = {num_features}")
    print(f"  Hidden dimension:  d_1  = {hidden_dim}")
    print(f"  Embedding dim:     d_2  = {embedding_dim}")
    print(f"  Activation:        ReLU (first layer)")
    print("\nParameter definitions:")
    print(f"  W^(1) ∈ R^({num_features}×{hidden_dim})")
    print(f"  b^(1) ∈ R^{hidden_dim}")
    print(f"  W^(2) ∈ R^({hidden_dim}×{embedding_dim})")
    print(f"  b^(2) ∈ R^{embedding_dim}")

    model = GCNLinkPredictor(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel stats:")
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")

    # ========================================================================
    # 3) Optimizer setup
    # ========================================================================
    print("\n[Step 3] Optimizer setup")
    print("-" * 80)

    learning_rate = 0.01
    weight_decay = 5e-4  # used in compute_loss (explicit L2)
    momentum = 0.9

    optimizer = ManualSGDMomentum(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=0.0  # avoid double-count with explicit L2 in loss
    )

    print("Optimizer: handwritten SGD + Momentum (no torch.optim)")
    print(f"  Learning rate α:    {learning_rate}")
    print(f"  Momentum μ:         {momentum}")
    print(f"  (Weight decay in loss: λ = {weight_decay})")
    print("\nLoss: BCE with Logits + explicit L2")
    print("  L = L_BCE + λ/2 * (||W^(1)||² + ||W^(2)||²)")

    # ========================================================================
    # 4) Training
    # ========================================================================
    print("\n[Step 4] Training")
    print("-" * 80)

    num_epochs = 200
    best_val_auc = 0
    patience = 20
    patience_counter = 0

    train_losses = []
    val_aucs = []
    val_aps = []

    for epoch in range(1, num_epochs + 1):
        # Train
        loss = train_epoch(model, optimizer, x, adj_norm,
                           train_pos, train_neg, weight_decay)
        train_losses.append(loss)

        # Validate
        val_auc, val_ap = evaluate(model, x, adj_norm, val_pos, val_neg)
        val_aucs.append(val_auc)
        val_aps.append(val_ap)

        # Progress
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Val AP: {val_ap:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ========================================================================
    # 5) Gradient view (backprop demo)
    # ========================================================================
    print("\n[Step 5] Gradient inspection")
    print("-" * 80)
    print("Running one forward/backward pass to display gradients...")

    model.train()
    optimizer.zero_grad()
    loss = compute_loss(model, x, adj_norm, train_pos, train_neg, weight_decay)
    loss.backward()

    visualize_gradients(model)

    print("\nBackpropagation sketch")
    print("Gradients propagate from output to input via chain rule:")
    print("1) Loss to decoder: ∂L/∂score")
    print("2) Decoder to embeddings: ∂score/∂Z^(2)")
    print("3) Second GCN layer: ∂L/∂W^(2), ∂L/∂b^(2)")
    print("4) First GCN layer (with ReLU): ∂L/∂W^(1), ∂L/∂b^(1)")
    print("\nSGD+Momentum updates:")
    print("  v ← μ·v + g")
    print("  θ ← θ − α·v")

    # ========================================================================
    # 6) Test evaluation
    # ========================================================================
    print("\n[Step 6] Test evaluation")
    print("-" * 80)

    model.load_state_dict(best_model_state)

    test_auc, test_ap = evaluate(model, x, adj_norm, test_pos, test_neg)

    print("Test set:")
    print(f"  ROC-AUC: {test_auc:.4f}")
    print(f"  Average Precision: {test_ap:.4f}")

    # Optional: curves and distributions on test set
    test_edge_index = torch.cat([test_pos, test_neg], dim=1)
    test_labels = torch.cat([
        torch.ones(test_pos.size(1)),
        torch.zeros(test_neg.size(1))
    ]).cpu().numpy()

    model.eval()
    with torch.no_grad():
        test_scores = model(x, adj_norm, test_edge_index)
        test_probs = torch.sigmoid(test_scores).cpu().numpy()

    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(test_labels, test_probs)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4))

    # ROC
    axes2[0].plot(fpr, tpr, linewidth=2)
    axes2[0].plot([0, 1], [0, 1], linestyle='--')
    axes2[0].set_xlabel('False Positive Rate')
    axes2[0].set_ylabel('True Positive Rate')
    axes2[0].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    axes2[0].grid(True, alpha=0.3)

    # PR
    axes2[1].plot(rec, prec, linewidth=2)
    axes2[1].set_xlabel('Recall')
    axes2[1].set_ylabel('Precision')
    axes2[1].set_title(f'Precision–Recall (AP = {test_ap:.3f})')
    axes2[1].grid(True, alpha=0.3)

    # Score histograms
    pos_probs = test_probs[:test_pos.size(1)]
    neg_probs = test_probs[test_pos.size(1):]
    axes2[2].hist(pos_probs, bins=30, alpha=0.6, label='Positive edges')
    axes2[2].hist(neg_probs, bins=30, alpha=0.6, label='Negative edges')
    axes2[2].set_xlabel('Predicted probability')
    axes2[2].set_ylabel('Count')
    axes2[2].set_title('Score Distribution (Test Set)')
    axes2[2].legend()

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out_path2 = os.path.join("outputs", "gcn_test_roc_pr_hist.png")
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    print(f"Saved test visualizations to: {out_path2}")
    plt.show()

    # ========================================================================
    # 7) Training curves
    # ========================================================================
    print("\n[Step 7] Training curves")
    print("-" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(train_losses, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Val AUC
    axes[1].plot(val_aucs, linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation AUC', fontsize=12)
    axes[1].set_title('Validation AUC over Epochs', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Val AP
    axes[2].plot(val_aps, linewidth=2, color='green')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Validation AP', fontsize=12)
    axes[2].set_title('Validation AP over Epochs', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "gcn_training_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {out_path}")
    plt.show()

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"✓ Data: {num_nodes} nodes, {num_features}-dim features")
    print(f"✓ Model: two-layer GCN ({num_features} → {hidden_dim} → {embedding_dim})")
    print(f"✓ Training: {len(train_losses)} epochs, handwritten SGD+Momentum")
    print(f"✓ Test performance: AUC={test_auc:.4f}, AP={test_ap:.4f}")
    print(f"✓ Regularization: L2 weight decay λ={weight_decay}")
    print(f"✓ Negative sampling: {train_neg.size(1)} negatives in training")
    print("=" * 80)

    return model, (test_auc, test_ap)


# ============================================================================
# Run main
# ============================================================================

if __name__ == "__main__":
    model, (test_auc, test_ap) = main()

    print("\n" + "=" * 80)
    print("Program finished.")
    print("=" * 80)
    print("\nAll required items are completed:")
    print("1) ✓ Parameter definitions (see Part 1)")
    print("2) ✓ Data setup with negative sampling (see Part 2)")
    print("3) ✓ Forward pass (GCNEncoder + InnerProductDecoder)")
    print("4) ✓ Loss (BCE with Logits + L2)")
    print("5) ✓ Backprop via autograd with gradient display")
    print("6) ✓ Parameter updates via handwritten SGD+Momentum (no torch.optim)")
    print("\nKey fixes:")
    print("✓ Negative sampling (necessary for link prediction)")
    print("✓ Vectorized decoder (efficient)")
    print("✓ Removed requires_grad on X (correct)")
    print("✓ Explicit L2 regularization (aligned with theory)")
    print("✓ BCEWithLogitsLoss (numerically stable)")
    print("=" * 80)
