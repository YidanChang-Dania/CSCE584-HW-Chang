
"""
Neural Network Diversity Project - Multi-Task Training Script
==================================================================
Author: Chang Yidan

"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Use the Agg backend before importing pyplot (headless rendering)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge

# XGBoost baseline
try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

from scipy.stats import pearsonr, ttest_rel

# Optional libs
try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from scipy.stats import gaussian_kde

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# Globals & Utilities
# ============================================================================

CATEGORICAL_FEATURE_ORDER = ["zipcode", "category", "brand", "platform"]

# Human-readable names for feature-importance plots
FEATURE_NAME_MAPPING = {
    "zipcode": "Zipcode",
    "category": "Category",
    "brand": "Brand",
    "platform": "Platform",
    "num_0": "Log Purchase Freq.",
    "num_1": "Avg Basket Size",
    "num_2": "Category Diversity",
    "num_3": "Feature Price",
    "num_4": "Day of Week",
    "num_5": "Part of Day",
    "num_6": "Historical Spend",
    "num_7": "Item Count",
    "num_8": "Distinct Categories",
    "num_9": "Avg Unit Price",
}

TODAY_STR = datetime.now().strftime("%Y%m%d")
FIGURE_OUTPUT_DIR = Path.home() / "Desktop"
FIGURE_SIZE = (6.4, 3.6)
FIGURE_DPI = 300
VOCAB_MIN_SIZE = 2

EXPORTED_FIGURES: List[Tuple[str, str]] = []


def figure_path(platform: str, phase: str, name: str) -> Path:
    """Build standardized image path."""
    filename = f"{TODAY_STR}_{platform.upper()}_{phase}_{name}.png"
    return FIGURE_OUTPUT_DIR / filename


def save_figure(fig, platform: str, phase: str, name: str, description: Optional[str] = None):
    """Save the figure with consistent settings and register manifest."""
    filepath = figure_path(platform, phase, name)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white", edgecolor="none", format="png")
    plt.close(fig)
    if description is None:
        default_scope = "TEST split (VAL used only for selection)."
        if name in {"Loss_Curves", "LR_Schedule", "Alpha_Scan_Validation"}:
            default_scope = "Train/VAL diagnostic."
        description = f"{name.replace('_', ' ')} for {platform.upper()} [{phase}]. {default_scope}"
    EXPORTED_FIGURES.append((str(filepath), description))
    return filepath


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enforce_vocab_health(vocabs: Dict[str, Dict[str, Dict]], logger: Optional[logging.Logger] = None,
                         min_size: int = VOCAB_MIN_SIZE, strict_features: Optional[List[str]] = None):
    """Fail-fast if any categorical vocab collapses."""
    if strict_features is None:
        strict_features = CATEGORICAL_FEATURE_ORDER
    issues = []
    for plat, plat_vocab in vocabs.items():
        for feat in strict_features:
            size = len(plat_vocab.get(feat, {}))
            if size < min_size:
                issues.append((plat, feat, size))
    if issues:
        lines = [f"[FATAL] Collapsed vocabularies detected (size < {min_size}). Aborting.", "Issues:"]
        for plat, feat, size in issues:
            lines.append(f"  - platform={plat}, feature={feat}, vocab_size={size}")
        msg = "\n".join(lines)
        if logger is not None:
            logger.error(msg)
        raise RuntimeError(msg)


def compute_business_kpi(metrics: Dict[str, float]) -> float:
    """Business KPI, smaller is better."""
    keys = ["rmse_h", "rmse_hhi", "mae_h", "mae_hhi", "kl_divergence", "val_loss_b"]
    values = [float(metrics[k]) for k in keys if k in metrics and metrics[k] is not None]
    return float(np.mean(values)) if values else 0.0


def get_peak_memory_mb() -> float:
    """Peak GPU/MPS memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def pvalue_to_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ============================================================================
# Config & Logging
# ============================================================================

@dataclass
class Config:
    """Training configuration and pipeline options."""
    data_root: str = str((Path.home() / "nn-diversity-project" / "data" / "processed").resolve())
    output_root: str = str((Path.home() / "nn-diversity-project" / "runs").resolve())

    # Platforms included in the experiments
    platforms: List[str] = field(default_factory=lambda: ["walmart", "kroger", "target"])

    # NN model hyperparams
    embedding_dim: int = 32
    shared_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    head_hidden_dim: int = 32
    dropout: float = 0.3

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5

    # Alpha values scanned on the validation set (including α=0.0 single-task baseline)
    alpha_options: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Fractions of target training data used in few-shot experiments
    few_shot_ratios: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.5])
    few_shot_epochs: int = 50
    few_shot_lr: float = 5e-4
    few_shot_freeze_encoder_epochs: int = 10

    # Multi-seed
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    seed: int = 42
    device: str = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.output_root) / self.timestamp


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("DiversityNN")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(run_dir / "training.log")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
# Data
# ============================================================================

class DiversityDataset(Dataset):
    """Tensor dataset: categorical indices, standardized numerics, targets."""

    def __init__(self, data_dir: Path, platform: str, split: str):
        tensor_dir = data_dir / platform / "tensors" / split
        self.X_cat = torch.LongTensor(np.load(tensor_dir / "X_cat.npy"))
        self.X_num = torch.FloatTensor(np.load(tensor_dir / "X_num.npy"))
        self.y_div = torch.FloatTensor(np.load(tensor_dir / "y_div.npy"))
        self.y_dist = torch.FloatTensor(np.load(tensor_dir / "y_dist.npy"))
        with open(tensor_dir / "meta.json", "r") as f:
            self.meta = json.load(f)
        self.num_categorical = self.X_cat.shape[1]
        self.num_numerical = self.X_num.shape[1]
        self.dist_output_dim = self.y_dist.shape[1]

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        return {"X_cat": self.X_cat[idx], "X_num": self.X_num[idx],
                "y_div": self.y_div[idx], "y_dist": self.y_dist[idx]}


def load_vocab_and_scaler(data_dir: Path, platform: str) -> Tuple[Dict, Dict]:
    vocab_path = data_dir / platform / f"vocab_{platform}.json"
    scaler_path = data_dir / platform / f"scaler_{platform}.json"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    return vocab, scaler


def get_vocab_sizes(vocabs: Dict[str, Dict]) -> Dict[str, int]:
    """Maximum vocab size per categorical feature across platforms."""
    max_sizes = {}
    for feat in CATEGORICAL_FEATURE_ORDER:
        sizes = [len(vocabs[plat].get(feat, {})) for plat in vocabs]
        max_sizes[feat] = max(sizes) if sizes else 100
    return max_sizes


def create_dataloaders(data_dir: Path, platform: str, batch_size: int, num_workers: int = 0):
    train_dataset = DiversityDataset(data_dir, platform, "train")
    val_dataset = DiversityDataset(data_dir, platform, "val")
    test_dataset = DiversityDataset(data_dir, platform, "test")

    dims = {"num_numerical": train_dataset.num_numerical,
            "dist_output_dim": train_dataset.dist_output_dim}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, dims


# ============================================================================
# Model
# ============================================================================

class EmbeddingEncoder(nn.Module):
    """Embedding encoder for categorical features."""

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dim: int):
        super().__init__()
        self.feature_order = CATEGORICAL_FEATURE_ORDER
        self.vocab_sizes = vocab_sizes
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleDict()
        for feat in self.feature_order:
            size = vocab_sizes.get(feat, 100)
            self.embeddings[feat] = nn.Embedding(size + 1, embedding_dim, padding_idx=0)
        self.output_dim = len(self.feature_order) * embedding_dim

    def forward(self, X_cat: torch.Tensor) -> torch.Tensor:
        emb_list = [self.embeddings[feat](X_cat[:, i]) for i, feat in enumerate(self.feature_order)]
        return torch.cat(emb_list, dim=-1)


class SharedEncoder(nn.Module):
    """Shared MLP over concatenated embeddings + normalized numerics."""

    def __init__(self, vocab_sizes: Dict[str, int], num_numerical: int,
                 embedding_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        self.cat_encoder = EmbeddingEncoder(vocab_sizes, embedding_dim)
        self.num_bn = nn.BatchNorm1d(num_numerical)
        input_dim = self.cat_encoder.output_dim + num_numerical

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, X_cat: torch.Tensor, X_num: torch.Tensor) -> torch.Tensor:
        cat_emb = self.cat_encoder(X_cat)
        num_norm = self.num_bn(X_num)
        return self.mlp(torch.cat([cat_emb, num_norm], dim=-1))

    def set_bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()


class HeadA(nn.Module):
    """Distribution head logits."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeadB(nn.Module):
    """Regression head for H & HHI."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskDiversityNet(nn.Module):
    """Full multi-task network."""

    def __init__(self, vocab_sizes: Dict[str, int], num_numerical: int, dist_output_dim: int,
                 embedding_dim: int = 32, shared_hidden_dims: List[int] = [256, 128, 64],
                 head_hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.encoder = SharedEncoder(vocab_sizes, num_numerical, embedding_dim, shared_hidden_dims, dropout)
        self.head_a = HeadA(self.encoder.output_dim, head_hidden_dim, dist_output_dim, dropout)
        self.head_b = HeadB(self.encoder.output_dim, head_hidden_dim, dropout)
        self._config = {"vocab_sizes": vocab_sizes, "num_numerical": num_numerical, "dist_output_dim": dist_output_dim,
                        "embedding_dim": embedding_dim, "shared_hidden_dims": shared_hidden_dims,
                        "head_hidden_dim": head_hidden_dim, "dropout": dropout}

    def forward(self, X_cat: torch.Tensor, X_num: torch.Tensor, return_embeddings: bool = False):
        enc = self.encoder(X_cat, X_num)
        dist_logits = self.head_a(enc)
        div_pred = self.head_b(enc)
        out = {"dist_logits": dist_logits, "div_pred": div_pred}
        if return_embeddings:
            out["embeddings"] = enc
        return out

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def set_encoder_bn_eval(self):
        self.encoder.set_bn_eval()


# ============================================================================
# Loss & Evaluation
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Loss: alpha * KL (A) + (1 - alpha) * MSE (B)."""

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def forward(self, dist_logits, div_pred, y_dist, y_div):
        loss_b = self.mse(div_pred, y_div)
        if self.alpha > 0 and dist_logits.shape[-1] == y_dist.shape[-1]:
            p_log = F.log_softmax(dist_logits, dim=-1)
            q = y_dist / (y_dist.sum(dim=-1, keepdim=True) + 1e-8)
            loss_a = self.kl(p_log, q)
        else:
            loss_a = torch.tensor(0.0, device=div_pred.device)
        total = self.alpha * loss_a + (1 - self.alpha) * loss_b
        return {"total": total, "head_a": loss_a, "head_b": loss_b}


class EarlyStopping:
    """Classic early stopping on scalar (smaller is better)."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


class TrainingRecorder:
    """Track training/val losses, LR, epoch time, and peak memory."""

    def __init__(self):
        self.history = {"train_loss": [], "val_loss": [], "train_loss_a": [], "val_loss_a": [],
                        "train_loss_b": [], "val_loss_b": [], "learning_rate": [], "epoch_time": []}
        self.best_epoch = 0
        self.early_stop_epoch = None
        self.total_time = 0.0
        self.peak_memory_mb = 0.0
        self.lr_drop_epochs = []

    def record_epoch(self, train_losses, val_losses, lr, epoch_time, epoch, is_best):
        self.history["train_loss"].append(train_losses["total"])
        self.history["val_loss"].append(val_losses["total"])
        self.history["train_loss_a"].append(train_losses["head_a"])
        self.history["val_loss_a"].append(val_losses["head_a"])
        self.history["train_loss_b"].append(train_losses["head_b"])
        self.history["val_loss_b"].append(val_losses["head_b"])
        self.history["learning_rate"].append(lr)
        self.history["epoch_time"].append(epoch_time)
        if is_best:
            self.best_epoch = epoch
        if len(self.history["learning_rate"]) > 1 and lr < self.history["learning_rate"][-2]:
            self.lr_drop_epochs.append(epoch)

    def set_early_stop(self, epoch: int):
        self.early_stop_epoch = epoch

    def finalize(self, total_time: float, peak_memory_mb: float):
        self.total_time = total_time
        self.peak_memory_mb = peak_memory_mb


def train_epoch(model, loader, optimizer, criterion, device, freeze_encoder_bn=False):
    model.train()
    if freeze_encoder_bn:
        model.encoder.set_bn_eval()

    total, la, lb, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        X_cat = batch["X_cat"].to(device)
        X_num = batch["X_num"].to(device)
        y_div = batch["y_div"].to(device)
        y_dist = batch["y_dist"].to(device)

        optimizer.zero_grad()
        out = model(X_cat, X_num)
        losses = criterion(out["dist_logits"], out["div_pred"], y_dist, y_div)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += losses["total"].item()
        la += losses["head_a"].item()
        lb += losses["head_b"].item()
        n += 1
    return {"total": total / n, "head_a": la / n, "head_b": lb / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False, head_b_only=False):
    model.eval()
    tot, la, lb, n = 0.0, 0.0, 0.0, 0
    div_pred_all, div_true_all, dist_pred_all, dist_true_all, emb_all = [], [], [], [], []
    X_cat_all, X_num_all = [], []

    for batch in loader:
        X_cat = batch["X_cat"].to(device)
        X_num = batch["X_num"].to(device)
        y_div = batch["y_div"].to(device)
        y_dist = batch["y_dist"].to(device)

        out = model(X_cat, X_num, return_embeddings=return_predictions)
        losses = criterion(out["dist_logits"], out["div_pred"], y_dist, y_div)
        tot += losses["total"].item()
        la += losses["head_a"].item()
        lb += losses["head_b"].item()
        n += 1

        if return_predictions:
            div_pred_all.append(out["div_pred"].cpu().numpy())
            div_true_all.append(y_div.cpu().numpy())
            X_cat_all.append(X_cat.cpu().numpy())
            X_num_all.append(X_num.cpu().numpy())
            if not head_b_only and out["dist_logits"].shape[-1] == y_dist.shape[-1]:
                dist_pred_all.append(F.softmax(out["dist_logits"], dim=-1).cpu().numpy())
                dist_true_all.append(y_dist.cpu().numpy())
            emb_all.append(out["embeddings"].cpu().numpy())

    res = {"loss": {"total": tot / n, "head_a": la / n, "head_b": lb / n}}
    if return_predictions:
        div_pred = np.concatenate(div_pred_all, axis=0)
        div_true = np.concatenate(div_true_all, axis=0)
        emb = np.concatenate(emb_all, axis=0)
        X_cat_np = np.concatenate(X_cat_all, axis=0)
        X_num_np = np.concatenate(X_num_all, axis=0)

        rmse_h = float(np.sqrt(mean_squared_error(div_true[:, 0], div_pred[:, 0])))
        rmse_hhi = float(np.sqrt(mean_squared_error(div_true[:, 1], div_pred[:, 1])))
        mae_h = float(mean_absolute_error(div_true[:, 0], div_pred[:, 0]))
        mae_hhi = float(mean_absolute_error(div_true[:, 1], div_pred[:, 1]))

        res["metrics"] = {"rmse_h": rmse_h, "rmse_hhi": rmse_hhi, "mae_h": mae_h, "mae_hhi": mae_hhi}
        res["predictions"] = {"div_pred": div_pred, "div_true": div_true,
                              "embeddings": emb, "X_cat": X_cat_np, "X_num": X_num_np}

        if dist_pred_all:
            dist_pred = np.concatenate(dist_pred_all, axis=0)
            dist_true = np.concatenate(dist_true_all, axis=0)
            dist_true_norm = dist_true / (dist_true.sum(axis=-1, keepdims=True) + 1e-8)
            kl = float(
                np.mean(np.sum(dist_true_norm * (np.log(dist_true_norm + 1e-8) - np.log(dist_pred + 1e-8)), axis=-1)))
            perplexity = float(np.exp(kl))
            res["metrics"]["kl_divergence"] = kl
            res["metrics"]["perplexity"] = perplexity
            res["predictions"]["dist_pred"] = dist_pred
            res["predictions"]["dist_true"] = dist_true
    return res


def train_model(model, train_loader, val_loader, config: Config, alpha, logger, save_dir, model_name="model"):
    """Train NN with ReduceLROnPlateau + early stopping on VAL head_b."""
    device = config.device
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    criterion = MultiTaskLoss(alpha=alpha)
    es = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    rec = TrainingRecorder()

    best_state = None
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(config.max_epochs):
        t0 = time.time()
        train_losses = train_epoch(model, train_loader, opt, criterion, device)
        val_res = evaluate(model, val_loader, criterion, device)
        val_losses = val_res["loss"]
        sch.step(val_losses["head_b"])
        lr = opt.param_groups[0]["lr"]
        ep_time = time.time() - t0

        is_best = es(val_losses["head_b"], epoch)
        rec.record_epoch(train_losses, val_losses, lr, ep_time, epoch, is_best)

        if is_best:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state_dict": best_state, "val_loss": val_losses["head_b"],
                        "alpha": alpha, "config": model._config},
                       save_dir / f"{model_name}_best.pt")

        if epoch % 10 == 0 or is_best:
            logger.info(f"Epoch {epoch:3d} | Train {train_losses['total']:.4f} | "
                        f"Val {val_losses['total']:.4f} | LR {lr:.2e} {'*' if is_best else ''}")
        if es.early_stop:
            rec.set_early_stop(epoch)
            logger.info(f"Early stopping at epoch {epoch}")
            break

    total = time.time() - start
    rec.finalize(total, get_peak_memory_mb())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rec


# ============================================================================
# Visualization (Corrected to match paper figures)
# ============================================================================

class Visualizer:
    """Utility class for plotting training and evaluation figures."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg = config
        self.logger = logger
        self.figsize = FIGURE_SIZE
        self.dpi = FIGURE_DPI

    # ---------- Phase 0: Data checks ----------

    def plot_vocab_sizes(self, vocabs: Dict[str, Dict], platform_dims: Dict):
        fig, ax = plt.subplots(figsize=self.figsize)
        features = CATEGORICAL_FEATURE_ORDER
        platforms = list(vocabs.keys())
        x = np.arange(len(features))
        width = 0.25
        for i, plat in enumerate(platforms):
            sizes = [len(vocabs[plat].get(f, {})) for f in features]
            ax.bar(x + i * width, sizes, width, label=plat.capitalize(), alpha=0.8)
            for j, s in enumerate(sizes):
                if s <= 1:
                    ax.annotate(f"⚠ {s}", (x[j] + i * width, s), ha="center", va="bottom",
                                fontsize=9, color="red", fontweight="bold")
                else:
                    ax.annotate(str(s), (x[j] + i * width, s), ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Categorical Feature", fontsize=11)
        ax.set_ylabel("Vocabulary Size", fontsize=11)
        ax.set_title("Vocabulary Sizes by Feature and Platform", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(features, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, "ALL", "phase0", "Vocab_Size_Bar")

    # --------#4: Combined Training Curve with Log Scale (Figure 7) ----------

    def plot_training_curves_combined(self, recorder, platform: str, alpha: float, peak_memory_mb: float = 0.0):
        """
        Figure 7: Combined training curve with:
        - Head B validation loss on log scale
        - LR drop vertical lines
        - Best epoch marker (red star)
        - Early stop vertical line
        - Text box with stats
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        epochs = range(1, len(recorder.history["val_loss_b"]) + 1)

        # Plot Head B validation loss (log scale)
        ax.plot(epochs, recorder.history["val_loss_b"], "b-", linewidth=2, label="Val Loss (Head B)")
        ax.plot(epochs, recorder.history["train_loss_b"], "b--", linewidth=1, alpha=0.6, label="Train Loss (Head B)")

        # Set log scale before querying y-limits to position annotations correctly
        ax.set_yscale("log")

        # Query limits after the scale change
        y_min, y_max = ax.get_ylim()

        # LR drop vertical lines (position annotations correctly for log scale)
        for e in recorder.lr_drop_epochs:
            ax.axvline(e + 1, color="orange", linestyle="--", alpha=0.7, linewidth=1.5)
            # Use geometric mean for log scale positioning
            annotation_y = np.sqrt(y_min * y_max) * 1.5
            ax.annotate(f"LR×0.5", (e + 1, annotation_y), fontsize=7, color="orange", rotation=90, va="bottom")

        # Best epoch marker (red star)
        best_val_loss = recorder.history["val_loss_b"][recorder.best_epoch]
        ax.plot(recorder.best_epoch + 1, best_val_loss, "r*", markersize=15, label=f"Best Epoch {recorder.best_epoch}")

        # Early stop vertical line
        if recorder.early_stop_epoch is not None:
            ax.axvline(recorder.early_stop_epoch + 1, color="red", linestyle=":", linewidth=2, label="Early Stop")

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Head B Validation Loss (log scale)", fontsize=11)
        ax.set_title(f"{platform.capitalize()} Training Progress (α={alpha})", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        # Text box with stats
        textstr = "\n".join([
            f"Best val loss: {best_val_loss:.4f}",
            f"Total train time: {recorder.total_time / 60:.1f} min",
            f"Early stop saved: {max(0, len(list(epochs)) - recorder.best_epoch - 1)} epochs",
            f"Peak VRAM: {peak_memory_mb:.1f} MB"
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

        save_figure(fig, platform, f"phase1_alpha{alpha}", "Training_Curve_Combined",
                    "Figure 7: Combined training curve with log scale, LR drops, best epoch, and stats.")

    # ---------- Alpha Scan with α=0 Single-Task Baseline (Figure 3) ----------

    def plot_alpha_scan_validation(self, alpha_results: Dict, platform: str):
        """
        Figure 3: U-shaped alpha scan including α=0 single-task baseline.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        alphas = sorted(alpha_results.keys())

        kpi_means = [alpha_results[a]["business_kpi_mean"] for a in alphas]
        kpi_stds = [alpha_results[a].get("business_kpi_std", 0) for a in alphas]

        # Plot with error bars
        ax.errorbar(alphas, kpi_means, yerr=kpi_stds, fmt="o-", linewidth=2, markersize=8,
                    capsize=5, label="Validation KPI (mean ± std)", color="#2196F3")

        # Highlight α=0 (single-task baseline)
        if 0.0 in alphas:
            idx_0 = alphas.index(0.0)
            ax.annotate("α=0: Single-task\nBaseline", (0.0, kpi_means[idx_0]),
                        xytext=(0.1, kpi_means[idx_0] + 0.02), fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        ha="left", color="gray")

        # Best alpha
        best_alpha = min(alphas, key=lambda a: alpha_results[a]["business_kpi_mean"])
        best_idx = alphas.index(best_alpha)
        ax.plot(best_alpha, kpi_means[best_idx], "r*", markersize=15)
        ax.annotate(f"Best α={best_alpha}", (best_alpha, kpi_means[best_idx]),
                    xytext=(best_alpha + 0.05, kpi_means[best_idx] - 0.01), fontsize=9,
                    color="red", fontweight="bold")

        ax.set_xlabel("Multi-task Weight α", fontsize=11)
        ax.set_ylabel("Validation KPI (lower is better)", fontsize=11)
        ax.set_title(f"{platform.capitalize()} α Sensitivity Analysis (VAL)", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        save_figure(fig, platform, "phase1", "Alpha_Scan_Validation",
                    "Figure 3: U-shaped alpha scan with α=0 single-task baseline highlighted.")

    # ---------- Scatter with Constant Baseline and % Improvement (Figure 2) ----------

    def plot_true_vs_pred_with_baseline(self, predictions: Dict, platform: str, constant_means: Dict):
        """
        Figure 2: True vs Predicted scatter with:
        - Constant baseline horizontal line
        - r, R², RMSE, p-value in text box
        - % improvement over constant prediction
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for j, (which, color) in enumerate([("H", "#2196F3"), ("HHI", "#4CAF50")]):
            ax = axes[j]
            idx = 0 if which == "H" else 1
            y_true = predictions["div_true"][:, idx]
            y_pred = predictions["div_pred"][:, idx]

            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.35, s=10, c=color)

            # y=x line
            minv, maxv = y_true.min(), y_true.max()
            ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=2, label="y=x (Perfect)")

            # Constant baseline horizontal line
            const_mean = constant_means[which.lower()]
            ax.axhline(const_mean, color="gray", linestyle=":", linewidth=2,
                       label=f"Constant Prediction ({const_mean:.3f})")

            # Statistics
            r, p_val = pearsonr(y_true, y_pred)
            r_squared = r ** 2
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            # RMSE of constant predictor
            rmse_const = np.sqrt(mean_squared_error(y_true, np.full_like(y_true, const_mean)))
            improvement_pct = (rmse_const - rmse) / rmse_const * 100

            ax.set_xlabel(f"True {which}", fontsize=10)
            ax.set_ylabel(f"Predicted {which}", fontsize=10)
            ax.set_title(f"{which} Prediction (TEST)", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.3)

            if which == "HHI":
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            # Text box with stats
            textstr = "\n".join([
                f"r = {r:.4f}",
                f"R² = {r_squared:.4f}",
                f"RMSE = {rmse:.4f}",
                f"p-value < 0.001" if p_val < 0.001 else f"p-value = {p_val:.3f}",
                f"",
                f"Model improves {improvement_pct:.1f}%",
                f"over constant prediction"
            ])
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)

        fig.suptitle(f"{platform.capitalize()} True vs Predicted (TEST)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_figure(fig, platform, "phase1", "True_vs_Pred_With_Baseline",
                    "Figure 2: Scatter plots with constant baseline, r, R², and % improvement.")

    # ---------- Feature Importance with Semantic Names (Figure 6) ----------

    def plot_feature_importance_permutation(self, model: nn.Module, predictions: Dict, platform: str,
                                            num_repeats: int = 5, max_features: int = 15):
        """
        Figure 6: Permutation importance with:
        - Semantic feature names
        - 0.01 threshold line for weak influence
        """
        if "X_num" not in predictions or "X_cat" not in predictions:
            self.logger.warning(f"[{platform}] Missing X_num/X_cat; skip permutation importance.")
            return

        model.eval()
        device = self.cfg.device
        Xc = predictions["X_cat"]
        Xn = predictions["X_num"]
        yt = predictions["div_true"]

        with torch.no_grad():
            out = model(torch.LongTensor(Xc).to(device), torch.FloatTensor(Xn).to(device))
            base = out["div_pred"].cpu().numpy()
        base_rmse_h = np.sqrt(mean_squared_error(yt[:, 0], base[:, 0]))
        base_rmse_hhi = np.sqrt(mean_squared_error(yt[:, 1], base[:, 1]))

        num_cat = Xc.shape[1]
        num_num = Xn.shape[1]
        cat_names = CATEGORICAL_FEATURE_ORDER[:num_cat]
        num_names = [f"num_{i}" for i in range(num_num)]

        feats, imp_h, imp_hhi = [], [], []

        # Categorical features
        for idx, name in enumerate(cat_names):
            dh, dhhi = [], []
            for _ in range(num_repeats):
                Xc_perm = Xc.copy()
                perm = np.random.permutation(len(Xc_perm))
                Xc_perm[:, idx] = Xc_perm[perm, idx]
                with torch.no_grad():
                    out = model(torch.LongTensor(Xc_perm).to(device), torch.FloatTensor(Xn).to(device))
                pred = out["div_pred"].cpu().numpy()
                dh.append(np.sqrt(mean_squared_error(yt[:, 0], pred[:, 0])) - base_rmse_h)
                dhhi.append(np.sqrt(mean_squared_error(yt[:, 1], pred[:, 1])) - base_rmse_hhi)
            feats.append(name)
            imp_h.append(float(np.mean(dh)))
            imp_hhi.append(float(np.mean(dhhi)))

        # Numeric features
        for idx, name in enumerate(num_names):
            dh, dhhi = [], []
            for _ in range(num_repeats):
                Xn_perm = Xn.copy()
                perm = np.random.permutation(len(Xn_perm))
                Xn_perm[:, idx] = Xn_perm[perm, idx]
                with torch.no_grad():
                    out = model(torch.LongTensor(Xc).to(device), torch.FloatTensor(Xn_perm).to(device))
                pred = out["div_pred"].cpu().numpy()
                dh.append(np.sqrt(mean_squared_error(yt[:, 0], pred[:, 0])) - base_rmse_h)
                dhhi.append(np.sqrt(mean_squared_error(yt[:, 1], pred[:, 1])) - base_rmse_hhi)
            feats.append(name)
            imp_h.append(float(np.mean(dh)))
            imp_hhi.append(float(np.mean(dhhi)))

        # Sort by total importance
        total_imp = [abs(h) + abs(hhi) for h, hhi in zip(imp_h, imp_hhi)]
        order = np.argsort(total_imp)[::-1][:max_features]
        feat_top = [feats[i] for i in order]
        imp_h_top = [imp_h[i] for i in order]
        imp_hhi_top = [imp_hhi[i] for i in order]

        # Map to semantic names
        feat_display = [FEATURE_NAME_MAPPING.get(f, f) for f in feat_top]

        y = np.arange(len(feat_display))
        fig, ax = plt.subplots(figsize=self.figsize)
        h = 0.35
        ax.barh(y - h / 2, imp_h_top, height=h, alpha=0.8, label="ΔRMSE (H)", color="#2196F3")
        ax.barh(y + h / 2, imp_hhi_top, height=h, alpha=0.8, label="ΔRMSE (HHI)", color="#4CAF50")

        # Zero line
        ax.axvline(0, color="black", linewidth=1)

        # Add 0.01 threshold line
        ax.axvline(0.01, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(0.011, len(feat_display) - 0.5, "≈0.01 threshold\n(weak influence)",
                fontsize=7, color="red", va="top")

        ax.set_yticks(y)
        ax.set_yticklabels(feat_display, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Permutation Importance (ΔRMSE)", fontsize=10)
        ax.set_title(f"{platform.capitalize()} Feature Importance (TEST)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="x")

        save_figure(fig, platform, "phase1", "Feature_Importance_Permutation",
                    "Figure 6: Permutation importance with semantic names and 0.01 threshold.")

    # ---------- Zero-shot Transfer Waterfall (Figure 4) ----------

    def plot_transfer_waterfall(self, results: Dict, config: Config):
        """
        Figure 4: Zero-shot transfer with:
        - ΔRMSE = RMSE(Zero-shot) − RMSE(In-platform)
        - Positive (red) = zero-shot worse
        - Negative (green) = zero-shot better
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        deltas, labels, colors_list = [], [], []

        for src in config.platforms:
            for trg in config.platforms:
                if src == trg:
                    continue
                if src in results and "cross_platform" in results[src] and trg in results[src]["cross_platform"]:
                    zs = results[src]["cross_platform"][trg]["zero_shot"]
                    inplat = results[trg]["in_platform_agg"]["mean"]

                    # ΔRMSE = Zero-shot - In-platform (as per paper)
                    delta_h = zs["rmse_h"] - inplat["rmse_h"]
                    delta_hhi = zs["rmse_hhi"] - inplat["rmse_hhi"]

                    deltas.extend([delta_h, delta_hhi])
                    labels.extend([f"{src[:3].upper()}→{trg[:3].upper()}\nH",
                                   f"{src[:3].upper()}→{trg[:3].upper()}\nHHI"])
                    # Red if positive (worse), green if negative (better)
                    colors_list.extend(["#F44336" if delta_h > 0 else "#4CAF50",
                                        "#F44336" if delta_hhi > 0 else "#4CAF50"])

        if not deltas:
            return

        x = np.arange(len(labels))
        bars = ax.bar(x, deltas, color=colors_list, edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", linewidth=1)

        # Annotations with Δ%
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            sign = "+" if delta > 0 else ""
            ax.annotate(f"{sign}{delta:.3f}", (bar.get_x() + bar.get_width() / 2, delta),
                        xytext=(0, 3 if delta > 0 else -10), textcoords="offset points",
                        ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Transfer Direction & Metric", fontsize=10)
        ax.set_ylabel("ΔRMSE (Zero-shot − In-platform)", fontsize=10)
        ax.set_title("Zero-Shot Cross-Platform Transfer (TEST)\nRed = Negative Transfer, Green = Positive Transfer",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#F44336", label="Negative Transfer (worse)"),
                           Patch(facecolor="#4CAF50", label="Neutral/Positive Transfer")]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

        save_figure(fig, "ALL", "phase2", "CrossPlatform_Transfer_Waterfall",
                    "Figure 4: Zero-shot ΔRMSE = Zero-shot − In-platform. Red = worse, Green = better.")

    # ---------- Few-Shot Learning Curve with Recovery Rate (Figure 5) ----------

    def plot_few_shot_learning_curve(self, few_shot_results: Dict, src: str, trg: str,
                                     zs_metrics: Dict, inplat_mean: Dict):
        """
        Figure 5: Few-shot learning curve with:
        - Y-axis: Performance Recovery Rate (%)
        - Reference lines for 0% (zero-shot) and 100% (in-platform)
        - Annotations for key recovery points
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ratios = sorted(few_shot_results.keys())

        for j, (which, color) in enumerate([("H", "#2196F3"), ("HHI", "#4CAF50")]):
            ax = axes[j]
            metric_key = f"rmse_{which.lower()}"

            zs_rmse = zs_metrics[metric_key]
            inplat_rmse = inplat_mean[metric_key]

            # Recovery rate = (RMSE_zs - RMSE_fewshot) / (RMSE_zs - RMSE_inplat) * 100
            recovery_means = []
            recovery_stds = []

            for ratio in ratios:
                rmse_list = few_shot_results[ratio][metric_key]
                recoveries = [(zs_rmse - rmse) / (zs_rmse - inplat_rmse + 1e-8) * 100 for rmse in rmse_list]
                recovery_means.append(np.mean(recoveries))
                recovery_stds.append(np.std(recoveries))

            x_pct = [f"{int(r * 100)}%" for r in ratios]
            x = np.arange(len(ratios))

            ax.errorbar(x, recovery_means, yerr=recovery_stds, fmt="o-", linewidth=2,
                        markersize=8, capsize=5, color=color, label=f"RMSE_{which} Recovery")

            # Reference lines
            ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, label="0% (Zero-shot)")
            ax.axhline(100, color="gray", linestyle=":", linewidth=1.5, label="100% (In-platform)")

            # Annotations for key points
            for i, (ratio, rec_mean) in enumerate(zip(ratios, recovery_means)):
                if ratio in [0.05, 0.2, 0.5]:  # Key ratios from paper
                    ax.annotate(f"{int(ratio * 100)}% data\n→ {rec_mean:.0f}% recovery",
                                (x[i], rec_mean), xytext=(10, 10 if rec_mean < 80 else -20),
                                textcoords="offset points", fontsize=8,
                                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))

            ax.set_xticks(x)
            ax.set_xticklabels(x_pct, fontsize=9)
            ax.set_xlabel("Target Training Data Used", fontsize=10)
            ax.set_ylabel(f"Performance Recovery Rate (%)", fontsize=10)
            ax.set_title(f"{which} Recovery", fontsize=11, fontweight="bold")
            ax.set_ylim(-10, 110)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")

        fig.suptitle(f"Few-Shot Transfer: {src.upper()} → {trg.upper()} (TEST)\n"
                     f"Recovery = (RMSE_zs − RMSE_ft) / (RMSE_zs − RMSE_inplat) × 100%",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        save_figure(fig, f"{src.upper()}to{trg.upper()}", "phase2", "FewShot_Recovery_Curve",
                    "Figure 5: Few-shot recovery rate (%) with 0% and 100% reference lines.")

    # ---------- Baseline Comparison with NN (Figure 1) ----------

    def plot_baseline_comparison_with_nn(self, baseline_results: Dict, nn_results: Dict, platform: str):
        """
        Figure 1: Baseline comparison including MTNN with:
        - Error bars from 3 seeds (95% CI)
        - Δ% and significance stars vs XGBoost
        - Shows improvement arrows
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Methods to compare (order as in paper)
        methods = ["Constant Pred.", "Linear Reg.", "XGBoost", "Multi-task NN"]

        for j, target in enumerate(["H", "HHI"]):
            ax = axes[j]
            metric_key = target

            means = []
            ci_lows = []
            ci_highs = []
            colors = []

            for method in methods:
                if method == "Multi-task NN":
                    # NN results from seeds
                    res = nn_results[metric_key]
                else:
                    # Baseline results - map names
                    name_map = {
                        "Constant Pred.": "ConstantMean",
                        "Linear Reg.": "Linear",
                        "XGBoost": "XGBoost"
                    }
                    key = name_map.get(method, method)
                    if key not in baseline_results:
                        means.append(0)
                        ci_lows.append(0)
                        ci_highs.append(0)
                        colors.append("gray")
                        continue
                    res = baseline_results[key][metric_key]

                means.append(res["mean"])
                ci_lows.append(res["mean"] - res["ci"][0])
                ci_highs.append(res["ci"][1] - res["mean"])
                colors.append("#4CAF50" if method == "Multi-task NN" else "#2196F3")

            x = np.arange(len(methods))
            bars = ax.bar(x, means, yerr=[ci_lows, ci_highs], capsize=5, alpha=0.85,
                          color=colors, edgecolor="black")

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel(f"RMSE ({target})", fontsize=10)
            ax.set_title(f"{target} Prediction Error", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3, axis="y")

            # Add significance and % improvement vs XGBoost
            if "XGBoost" in baseline_results and len(means) == 4:
                xgb_mean = means[2]  # XGBoost index
                xgb_samples = baseline_results["XGBoost"][metric_key]["samples"]

                # NN vs XGBoost
                nn_mean = means[3]
                nn_samples = nn_results[metric_key]["samples"]

                delta_pct = (nn_mean - xgb_mean) / xgb_mean * 100

                # Paired t-test
                if len(nn_samples) == len(xgb_samples):
                    _, p_val = ttest_rel(nn_samples, xgb_samples)
                else:
                    # Bootstrap comparison
                    diff = nn_samples.mean() - xgb_samples.mean()
                    combined_std = np.sqrt(nn_samples.std() ** 2 + xgb_samples.std() ** 2)
                    t_stat = diff / (combined_std / np.sqrt(min(len(nn_samples), len(xgb_samples))))
                    from scipy.stats import t
                    p_val = 2 * (1 - t.cdf(abs(t_stat), df=min(len(nn_samples), len(xgb_samples)) - 1))

                stars = pvalue_to_stars(p_val)

                # Add annotation with arrow
                ax.annotate(f"{delta_pct:.1f}%{stars}", xy=(3, nn_mean),
                            xytext=(3, nn_mean * 0.85),
                            fontsize=9, ha="center", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

        fig.suptitle(f"{platform.capitalize()}: Multi-task NN Achieves Lower RMSE than Baselines\n"
                     f"Error bars: 95% CI from 3 seeds. *p<0.05, **p<0.01, ***p<0.001 vs XGBoost",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        save_figure(fig, platform, "phase1", "Baseline_Comparison_With_NN",
                    "Figure 1: Baseline comparison with NN, showing improvement and significance vs XGBoost.")

    # ---------- Final Summary Table with 3 Platforms (Figure 8) ----------

    def plot_final_summary_table(self, results: Dict, config: Config):
        """Figure 8: Final table with 3 platforms (Walmart, Kroger, Target)."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")

        data = [["Platform", "Best α", "RMSE_H", "RMSE_HHI", "MAE_H", "MAE_HHI", "KL_Div", "Perplexity"]]

        for plat in config.platforms:
            if plat in results:
                agg = results[plat]["in_platform_agg"]
                mean = agg["mean"]
                std = agg["std"]
                row = [
                    plat.capitalize(),
                    f"{results[plat]['best_alpha']:.1f}",
                    f"{mean.get('rmse_h', 0):.4f} ± {std.get('rmse_h', 0):.4f}",
                    f"{mean.get('rmse_hhi', 0):.4f} ± {std.get('rmse_hhi', 0):.4f}",
                    f"{mean.get('mae_h', 0):.4f} ± {std.get('mae_h', 0):.4f}",
                    f"{mean.get('mae_hhi', 0):.4f} ± {std.get('mae_hhi', 0):.4f}",
                    f"{mean.get('kl_divergence', 0):.4f} ± {std.get('kl_divergence', 0):.4f}",
                    f"{mean.get('perplexity', 0):.2f} ± {std.get('perplexity', 0):.2f}",
                ]
                data.append(row)

        tab = ax.table(cellText=data, loc="center", cellLoc="center", colWidths=[0.12] * 8)
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        tab.scale(1.2, 1.8)

        for i in range(8):
            tab[(0, i)].set_facecolor("#3F51B5")
            tab[(0, i)].set_text_props(color="white", fontweight="bold")

        ax.set_title("Table I / Figure 8: Main Results (TEST)\nValues are mean ± std across 3 random seeds",
                     fontsize=12, fontweight="bold", pad=20)

        save_figure(fig, "ALL", "final", "Main_Results_Table",
                    "Figure 8: Final results table with all 3 platforms.")

    # Other visualization methods

    def plot_test_radar(self, metrics: Dict, platform: str):
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
        labels = ["RMSE_H", "RMSE_HHI", "MAE_H", "MAE_HHI", "KL_Div", "Perplexity (log10)"]
        vals = [metrics.get("rmse_h", 0), metrics.get("rmse_hhi", 0),
                metrics.get("mae_h", 0), metrics.get("mae_hhi", 0),
                metrics.get("kl_divergence", 0), np.log10(metrics.get("perplexity", 1) + 1)]
        maxv = max(vals) if max(vals) > 0 else 1
        vn = [v / maxv for v in vals]
        ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        vn += vn[:1]
        ang += ang[:1]
        ax.plot(ang, vn, "o-", linewidth=2, color="#2196F3")
        ax.fill(ang, vn, alpha=0.25, color="#2196F3")
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{platform.capitalize()} TEST Metrics Radar", fontsize=11, fontweight="bold", pad=12)
        save_figure(fig, platform, "phase1", "Test_Metrics_Radar")

    def plot_domain_difference_radar(self, results: Dict, config: Config):
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
        labels = ["RMSE_H", "RMSE_HHI", "MAE_H", "MAE_HHI"]
        colors = {"walmart": "#0071CE", "kroger": "#E31837", "target": "#CC0000"}

        for plat in config.platforms:
            if plat in results:
                mean = results[plat]["in_platform_agg"]["mean"]
                values = [mean.get("rmse_h", 0), mean.get("rmse_hhi", 0),
                          mean.get("mae_h", 0), mean.get("mae_hhi", 0)]
                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                v = values + values[:1]
                ang = angles + angles[:1]
                color = colors.get(plat, "#808080")
                ax.plot(ang, v, "o-", linewidth=2, label=plat.capitalize(), color=color)
                ax.fill(ang, v, alpha=0.15, color=color)

        ax.set_xticks(np.linspace(0, 2 * np.pi, len(labels), endpoint=False))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("Domain Difference Radar (In-Platform, TEST)", fontsize=11, fontweight="bold", pad=12)
        ax.legend(loc="upper right", fontsize=8)
        save_figure(fig, "ALL", "phase2", "Domain_Difference_Radar")

    def plot_training_time(self, time_records: Dict[str, Dict[float, float]]):
        fig, ax = plt.subplots(figsize=self.figsize)
        platforms = list(time_records.keys())
        if not platforms:
            return
        alphas = sorted(list(time_records[platforms[0]].keys()))
        x = np.arange(len(alphas))
        width = 0.25
        for i, plat in enumerate(platforms):
            times = [time_records[plat].get(a, 0) / 60 for a in alphas]
            ax.bar(x + i * width, times, width, label=plat.capitalize(), alpha=0.85)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Training Time (minutes)")
        ax.set_title("Training Time by Platform and Alpha", fontsize=11, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"α={a}" for a in alphas])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        save_figure(fig, "ALL", "phase1", "Training_Time_Bar")

    def plot_memory_usage(self, memory_records: Dict[str, float]):
        fig, ax = plt.subplots(figsize=self.figsize)
        plats = list(memory_records.keys())
        mems = [memory_records[p] for p in plats]
        bars = ax.bar(plats, mems, edgecolor="black", alpha=0.85)
        for b, m in zip(bars, mems):
            ax.annotate(f"{m:.1f} MB", (b.get_x() + b.get_width() / 2, m), ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Platform")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Peak Memory Usage by Platform", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        save_figure(fig, "ALL", "phase1", "Peak_Memory_Bar")


# ============================================================================
# Baselines (with NN integration for Figure 1)
# ============================================================================

class BaselineConstantMean:
    """Predict global mean (H, HHI) from TRAIN."""

    def fit(self, X_cat_tr, X_num_tr, y_tr):
        self.mu = y_tr.mean(axis=0)

    def predict(self, X_cat_te, X_num_te):
        return np.tile(self.mu, (len(X_cat_te), 1))


class BaselineLinear:
    """LinearRegression with same encoded features."""

    def fit(self, X_cat_tr, X_num_tr, y_tr):
        X = np.hstack([X_cat_tr.astype(np.float32), X_num_tr.astype(np.float32)])
        self.mdl = LinearRegression()
        self.mdl.fit(X, y_tr)

    def predict(self, X_cat_te, X_num_te):
        X = np.hstack([X_cat_te.astype(np.float32), X_num_te.astype(np.float32)])
        return self.mdl.predict(X)


class BaselineXGB:
    """XGBoost baseline."""

    def __init__(self):
        if not HAS_XGB:
            raise RuntimeError("XGBoost not available.")
        self.mdl_h = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                      random_state=42, n_jobs=4, objective="reg:squarederror")
        self.mdl_hhi = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                        random_state=42, n_jobs=4, objective="reg:squarederror")

    def fit(self, X_cat_tr, X_num_tr, y_tr):
        X = np.hstack([X_cat_tr.astype(np.float32), X_num_tr.astype(np.float32)])
        self.mdl_h.fit(X, y_tr[:, 0])
        self.mdl_hhi.fit(X, y_tr[:, 1])

    def predict(self, X_cat_te, X_num_te):
        X = np.hstack([X_cat_te.astype(np.float32), X_num_te.astype(np.float32)])
        return np.column_stack([self.mdl_h.predict(X), self.mdl_hhi.predict(X)])


def run_baselines_for_platform(platform: str, data_dir: Path, config: Config, logger: logging.Logger):
    """Train baselines on TRAIN, evaluate on TEST with 3-seed bootstrap CI."""
    train_ds = DiversityDataset(data_dir, platform, "train")
    test_ds = DiversityDataset(data_dir, platform, "test")

    Xc_tr = train_ds.X_cat.numpy()
    Xn_tr = train_ds.X_num.numpy()
    y_tr = train_ds.y_div.numpy()
    Xc_te = test_ds.X_cat.numpy()
    Xn_te = test_ds.X_num.numpy()
    y_te = test_ds.y_div.numpy()

    baselines = [("ConstantMean", BaselineConstantMean()),
                 ("Linear", BaselineLinear())]
    if HAS_XGB:
        baselines.append(("XGBoost", BaselineXGB()))
    else:
        logger.warning("[Baselines] XGBoost not available.")

    results = {}

    # Use seeds for CI (to match paper's 3 random seeds approach)
    for name, mdl in baselines:
        rmse_h_list = []
        rmse_hhi_list = []

        for seed in config.seeds:
            np.random.seed(seed)
            mdl_copy = deepcopy(mdl)
            mdl_copy.fit(Xc_tr, Xn_tr, y_tr)
            pred = mdl_copy.predict(Xc_te, Xn_te)

            rmse_h_list.append(np.sqrt(mean_squared_error(y_te[:, 0], pred[:, 0])))
            rmse_hhi_list.append(np.sqrt(mean_squared_error(y_te[:, 1], pred[:, 1])))

        results[name] = {
            "H": {
                "mean": float(np.mean(rmse_h_list)),
                "std": float(np.std(rmse_h_list)),
                "ci": (float(np.percentile(rmse_h_list, 2.5)), float(np.percentile(rmse_h_list, 97.5))),
                "samples": np.array(rmse_h_list)
            },
            "HHI": {
                "mean": float(np.mean(rmse_hhi_list)),
                "std": float(np.std(rmse_hhi_list)),
                "ci": (float(np.percentile(rmse_hhi_list, 2.5)), float(np.percentile(rmse_hhi_list, 97.5))),
                "samples": np.array(rmse_hhi_list)
            }
        }

    # Get constant means for Figure 2
    constant_means = {"h": y_tr[:, 0].mean(), "hhi": y_tr[:, 1].mean()}

    return results, constant_means


# ============================================================================
# Few-shot fine-tuning
# ============================================================================

def few_shot_finetune(init_model: MultiTaskDiversityNet, train_subset_loader: DataLoader,
                      val_loader: DataLoader, config: Config, alpha: float, logger: logging.Logger):
    """
    Few-shot fine-tuning on a small target subset.

    The encoder is frozen for the first few epochs, then unfrozen.
    The optimizer is rebuilt only once at the freeze→unfreeze transition
    so that Adam's momentum and adaptive statistics are preserved.
    """

    device = config.device
    model = deepcopy(init_model).to(device)
    criterion = MultiTaskLoss(alpha=alpha)

    # Start with encoder frozen
    model.freeze_encoder()

    # Build optimizer with only trainable parameters (heads only initially)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.few_shot_lr, weight_decay=config.weight_decay)

    best_state = None
    best_val = float("inf")

    for epoch in range(config.few_shot_epochs):
        # Rebuild the optimizer once when unfreezing the encoder
        if epoch == config.few_shot_freeze_encoder_epochs:
            # Unfreeze encoder and include its parameters in the optimizer
            model.unfreeze_encoder()
            opt = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.few_shot_lr,
                weight_decay=config.weight_decay,
            )

        # Determine if BN should be in eval mode
        freeze_bn = epoch < config.few_shot_freeze_encoder_epochs

        train_epoch(model, train_subset_loader, opt, criterion, device, freeze_encoder_bn=freeze_bn)
        val_res = evaluate(model, val_loader, criterion, device)
        val_loss_b = val_res["loss"]["head_b"]

        if val_loss_b < best_val:
            best_val = val_loss_b
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_few_shot_experiments(config: Config, visualizer: Visualizer, logger: logging.Logger,
                             best_models: Dict[str, MultiTaskDiversityNet], results: Dict[str, Any],
                             data_dir: Path):
    """Run few-shot experiments and compute recovery curves."""

    for src in config.platforms:
        for trg in config.platforms:
            if src == trg:
                continue

            logger.info(f"Few-shot: {src.upper()} -> {trg.upper()}")

            train_ds = DiversityDataset(data_dir, trg, "train")
            val_ds = DiversityDataset(data_dir, trg, "val")
            test_ds = DiversityDataset(data_dir, trg, "test")

            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

            src_best_alpha = results[src]["best_alpha"]
            src_model = best_models[src]

            # Zero-shot metrics
            zs_metrics = evaluate(src_model, test_loader, MultiTaskLoss(alpha=0.0), config.device,
                                  return_predictions=True, head_b_only=True)["metrics"]

            # In-platform reference
            inplat_mean = results[trg]["in_platform_agg"]["mean"]

            # Few-shot results storage
            few_shot_results = {}

            for ratio in config.few_shot_ratios:
                rmse_h_list, rmse_hhi_list = [], []

                for seed in config.seeds:
                    set_seed(seed)
                    n_train = len(train_ds)
                    n_sub = max(1, int(np.ceil(ratio * n_train)))
                    idx = np.random.RandomState(seed).choice(n_train, size=n_sub, replace=False)
                    subset = Subset(train_ds, idx.tolist())
                    sub_loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)

                    ft_model = few_shot_finetune(src_model, sub_loader, val_loader, config,
                                                 alpha=src_best_alpha, logger=logger)
                    test_res = evaluate(ft_model, test_loader, MultiTaskLoss(alpha=src_best_alpha),
                                        config.device, return_predictions=True)

                    rmse_h_list.append(test_res["metrics"]["rmse_h"])
                    rmse_hhi_list.append(test_res["metrics"]["rmse_hhi"])

                few_shot_results[ratio] = {
                    "rmse_h": rmse_h_list,
                    "rmse_hhi": rmse_hhi_list
                }

            # Plot with recovery rate (Figure 5)
            visualizer.plot_few_shot_learning_curve(few_shot_results, src, trg, zs_metrics, inplat_mean)


# ============================================================================
# Main
# ============================================================================

def main():
    config = Config()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "checkpoints").mkdir(exist_ok=True)
    logger = setup_logging(config.run_dir)

    logger.info("=" * 70)
    logger.info("Neural Network Diversity Project - Final Corrected Pipeline")
    logger.info(f"Platforms: {config.platforms}")
    logger.info(f"Alpha options: {config.alpha_options}")
    logger.info(f"Few-shot ratios: {config.few_shot_ratios}")
    logger.info(f"All figures will be saved to: {FIGURE_OUTPUT_DIR}")
    logger.info("=" * 70)

    set_seed(config.seed)
    viz = Visualizer(config, logger)

    # Phase 0: load vocabs/scalers
    logger.info("Phase 0: Loading data & vocab health checks...")
    data_dir = Path(config.data_root)
    vocabs, scalers, platform_dims = {}, {}, {}

    for p in config.platforms:
        vocabs[p], scalers[p] = load_vocab_and_scaler(data_dir, p)
        tr = DiversityDataset(data_dir, p, "train")
        platform_dims[p] = {"num_numerical": tr.num_numerical, "dist_output_dim": tr.dist_output_dim}
        logger.info(f"  {p}: {len(tr)} samples, K={platform_dims[p]['dist_output_dim']}")

    enforce_vocab_health(vocabs, logger=logger)
    viz.plot_vocab_sizes(vocabs, platform_dims)

    vocab_sizes = get_vocab_sizes(vocabs)
    num_numerical = platform_dims[config.platforms[0]]["num_numerical"]

    # Storage
    results: Dict[str, Any] = {}
    best_models: Dict[str, MultiTaskDiversityNet] = {}
    best_seeds: Dict[str, int] = {}
    time_records = {p: {} for p in config.platforms}
    memory_records = {p: 0.0 for p in config.platforms}
    all_alpha_results = {}
    seed_level_metrics_all = {}

    # Phase 1: In-platform NN training
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1: In-Platform Training (NN) - With α=0 Single-Task Baseline")
    logger.info("=" * 70)

    for platform in config.platforms:
        logger.info(f"\n--- Training on {platform.upper()} ---")
        train_loader, val_loader, test_loader, dims = create_dataloaders(data_dir, platform, config.batch_size)
        dist_output_dim = dims["dist_output_dim"]

        best_alpha = None
        best_seed_for_plat = None
        best_kpi = float("inf")
        alpha_results = {}
        seed_level_metrics_all[platform] = {}

        for alpha in config.alpha_options:
            logger.info(f"  Alpha={alpha}" + (" (Single-task baseline)" if alpha == 0.0 else ""))
            seed_level_metrics_all[platform][alpha] = {"rmse_h": [], "rmse_hhi": [], "business_kpi": []}
            total_time_alpha = 0.0

            for seed in config.seeds:
                set_seed(seed)
                model = MultiTaskDiversityNet(vocab_sizes=vocab_sizes, num_numerical=num_numerical,
                                              dist_output_dim=dist_output_dim, embedding_dim=config.embedding_dim,
                                              shared_hidden_dims=config.shared_hidden_dims,
                                              head_hidden_dim=config.head_hidden_dim, dropout=config.dropout)
                model, rec = train_model(model, train_loader, val_loader, config, alpha, logger,
                                         config.run_dir / "checkpoints", f"{platform}_alpha{alpha}_seed{seed}")
                total_time_alpha += rec.total_time
                memory_records[platform] = max(memory_records[platform], rec.peak_memory_mb)

                val_result = evaluate(model, val_loader, MultiTaskLoss(alpha=alpha), config.device,
                                      return_predictions=True)
                metrics = deepcopy(val_result["metrics"])
                metrics["val_loss_b"] = val_result["loss"]["head_b"]
                metrics["business_kpi"] = compute_business_kpi(metrics)

                seed_level_metrics_all[platform][alpha]["rmse_h"].append(metrics["rmse_h"])
                seed_level_metrics_all[platform][alpha]["rmse_hhi"].append(metrics["rmse_hhi"])
                seed_level_metrics_all[platform][alpha]["business_kpi"].append(metrics["business_kpi"])

                # Plot combined training curve for best alpha candidate
                if alpha == 0.3 and seed == config.seeds[0]:
                    viz.plot_training_curves_combined(rec, platform, alpha, rec.peak_memory_mb)

                if metrics["business_kpi"] < best_kpi:
                    best_kpi = metrics["business_kpi"]
                    best_alpha = alpha
                    best_seed_for_plat = seed

            rh = seed_level_metrics_all[platform][alpha]["rmse_h"]
            rhhi = seed_level_metrics_all[platform][alpha]["rmse_hhi"]
            kpi = seed_level_metrics_all[platform][alpha]["business_kpi"]
            alpha_results[alpha] = {
                "business_kpi_mean": float(np.mean(kpi)),
                "business_kpi_std": float(np.std(kpi)),
                "rmse_h_mean": float(np.mean(rh)), "rmse_h_std": float(np.std(rh)),
                "rmse_hhi_mean": float(np.mean(rhhi)), "rmse_hhi_std": float(np.std(rhhi)),
            }
            time_records[platform][alpha] = total_time_alpha / max(len(config.seeds), 1)

        all_alpha_results[platform] = alpha_results
        logger.info(f"Best alpha for {platform} (by KPI): {best_alpha} (seed={best_seed_for_plat}, KPI={best_kpi:.6f})")
        best_seeds[platform] = best_seed_for_plat

        # Plot alpha scan (Figure 3)
        viz.plot_alpha_scan_validation(alpha_results, platform)

        # Evaluate on TEST across all seeds for best alpha
        test_metrics_seeds = []
        nn_test_rmse_h = []
        nn_test_rmse_hhi = []

        for seed in config.seeds:
            ckpt_path = config.run_dir / "checkpoints" / f"{platform}_alpha{best_alpha}_seed{seed}_best.pt"
            ckpt = torch.load(ckpt_path, map_location=config.device)
            arch_cfg = ckpt["config"]
            model = MultiTaskDiversityNet(**arch_cfg)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(config.device)

            if seed == best_seed_for_plat:
                best_models[platform] = deepcopy(model)

            test_res = evaluate(model, test_loader, MultiTaskLoss(alpha=best_alpha), config.device,
                                return_predictions=True)
            test_metrics_seeds.append(test_res["metrics"])
            nn_test_rmse_h.append(test_res["metrics"]["rmse_h"])
            nn_test_rmse_hhi.append(test_res["metrics"]["rmse_hhi"])

            if seed == best_seed_for_plat:
                preds = test_res["predictions"]
                viz.plot_test_radar(test_res["metrics"], platform)
                viz.plot_feature_importance_permutation(model, preds, platform)

        keys = ["rmse_h", "rmse_hhi", "mae_h", "mae_hhi", "kl_divergence", "perplexity"]
        agg_mean = {k: float(np.mean([m.get(k, np.nan) for m in test_metrics_seeds if k in m])) for k in keys}
        agg_std = {k: float(np.std([m.get(k, np.nan) for m in test_metrics_seeds if k in m])) for k in keys}

        # NN results for Figure 1
        nn_results = {
            "H": {
                "mean": float(np.mean(nn_test_rmse_h)),
                "std": float(np.std(nn_test_rmse_h)),
                "ci": (float(np.percentile(nn_test_rmse_h, 2.5)), float(np.percentile(nn_test_rmse_h, 97.5))),
                "samples": np.array(nn_test_rmse_h)
            },
            "HHI": {
                "mean": float(np.mean(nn_test_rmse_hhi)),
                "std": float(np.std(nn_test_rmse_hhi)),
                "ci": (float(np.percentile(nn_test_rmse_hhi, 2.5)), float(np.percentile(nn_test_rmse_hhi, 97.5))),
                "samples": np.array(nn_test_rmse_hhi)
            }
        }

        results[platform] = {
            "best_alpha": best_alpha,
            "best_seed": best_seed_for_plat,
            "best_kpi": best_kpi,
            "alpha_results": alpha_results,
            "seed_level_metrics": seed_level_metrics_all[platform],
            "in_platform_agg": {"mean": agg_mean, "std": agg_std},
            "cross_platform": {},
            "nn_test_results": nn_results,
        }

        # Run baselines and plot Figure 1 & Figure 2
        baseline_results, constant_means = run_baselines_for_platform(platform, data_dir, config, logger)
        results[platform]["baseline_results"] = baseline_results
        results[platform]["constant_means"] = constant_means

        # Figure 1: Baseline comparison with NN
        viz.plot_baseline_comparison_with_nn(baseline_results, nn_results, platform)

        # Figure 2: Scatter with constant baseline
        final_preds = test_res["predictions"]  # From last seed
        viz.plot_true_vs_pred_with_baseline(final_preds, platform, constant_means)

    viz.plot_training_time(time_records)
    viz.plot_memory_usage(memory_records)

    # Phase 2: Cross-platform
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2: Cross-Platform Generalization")
    logger.info("=" * 70)

    for src in config.platforms:
        src_model = best_models[src]
        for trg in config.platforms:
            if src == trg:
                continue
            _, _, trg_test_loader, _ = create_dataloaders(data_dir, trg, config.batch_size)
            zs = evaluate(src_model, trg_test_loader, MultiTaskLoss(alpha=0.0), config.device,
                          return_predictions=True, head_b_only=True)
            results[src]["cross_platform"][trg] = {"zero_shot": zs["metrics"]}
            logger.info(f"Zero-shot {src.upper()}→{trg.upper()}: RMSE_H={zs['metrics']['rmse_h']:.4f}, "
                        f"RMSE_HHI={zs['metrics']['rmse_hhi']:.4f}")

    # Figure 4: Zero-shot transfer waterfall
    viz.plot_transfer_waterfall(results, config)
    viz.plot_domain_difference_radar(results, config)

    # Figure 5: Few-shot learning curves
    run_few_shot_experiments(config, viz, logger, best_models, results, data_dir)

    # Figure 8: Final summary table
    viz.plot_final_summary_table(results, config)

    # Save results
    with open(config.run_dir / "metrics_all.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for plat, plat_res in results.items():
            results_serializable[plat] = {}
            for k, v in plat_res.items():
                if k in ["nn_test_results", "baseline_results"]:
                    continue  # Skip numpy arrays
                results_serializable[plat][k] = v
        json.dump(results_serializable, f, indent=2)

    # Export manifest
    logger.info("\n" + "=" * 70)
    logger.info("Export Manifest")
    logger.info("=" * 70)
    logger.info(f"Total figures exported: {len(EXPORTED_FIGURES)}")
    logger.info(f"Output directory: {FIGURE_OUTPUT_DIR}")
    logger.info("-" * 70)
    for i, (fp, desc) in enumerate(EXPORTED_FIGURES, 1):
        logger.info(f"{i:3d}. {fp}\n     - {desc}")
    logger.info("-" * 70)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()