"""
ResNet for CIFAR-10 with a no-skip baseline

Overview
- Use CuPy on GPU when available, otherwise fall back to NumPy
- Move data to the xp backend after loading; convert back to NumPy for plotting/logging
- Manual backprop for conv, BN, pooling, loss, and optimizer
- Train the main model (with residual connections) and plot train/val curves
- Additionally train a no-residual baseline (small subset + few epochs)
- Produce gradient-flow comparison to illustrate the effect of skip connections
"""

# -------------------- Imports --------------------
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pickle
import os
import urllib.request
import tarfile
import time
from collections import defaultdict, OrderedDict

# -------------------- Backend selector (xp) --------------------
try:
    import cupy as cp
    XP_BACKEND = "cupy"
except Exception:
    cp = None
    XP_BACKEND = "numpy"

xp = cp if XP_BACKEND == "cupy" else np

# Seeds
np.random.seed(42)
if cp is not None:
    cp.random.seed(42)

# -------------------- Host<->Device helpers --------------------
def to_xp(a):
    """Convert a NumPy array to xp (CuPy on GPU if available)."""
    if XP_BACKEND == "cupy":
        if isinstance(a, np.ndarray):
            return cp.asarray(a)
    return a

def to_np(a):
    """Convert an xp array back to NumPy (for printing and plotting)."""
    if XP_BACKEND == "cupy" and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


# =====================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# =====================================================================

def download_cifar10(data_dir='./data'):
    """Download and extract CIFAR-10."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')

    if not os.path.exists(file_path):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(cifar10_url, file_path)

        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("Download complete!")

    return os.path.join(data_dir, 'cifar-10-batches-py')


def load_cifar10_batch(file_path):
    """Load one CIFAR-10 batch and return NumPy arrays."""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    data = batch[b'data']
    labels = batch[b'labels']

    # Shape: (N, 3, 32, 32); will convert to xp later
    data = data.reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)

    return data, labels


def load_cifar10(data_dir=None):
    """Load full CIFAR-10 and normalize to [-1, 1]."""
    if data_dir is None:
        data_dir = download_cifar10()

    # Train
    X_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        X_train.append(data); y_train.append(labels)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # Test
    test_file = os.path.join(data_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_file)

    # Normalize
    X_train = (X_train / 127.5) - 1.0
    X_test  = (X_test  / 127.5) - 1.0

    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


def create_validation_split(X_train, y_train, val_size=5000):
    """Split into training and validation sets."""
    num_train = len(X_train) - val_size
    X_val = X_train[num_train:]
    y_val = y_train[num_train:]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]
    return X_train, y_train, X_val, y_val


# =====================================================================
# PART 2: ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# =====================================================================

class ReLU:
    @staticmethod
    def forward(x):
        return xp.maximum(0, x)

    @staticmethod
    def backward(x):
        return (x > 0).astype(xp.float32)


class Softmax:
    @staticmethod
    def forward(x):
        x_max = xp.max(x, axis=1, keepdims=True)
        exp_x = xp.exp(x - x_max)
        return exp_x / xp.sum(exp_x, axis=1, keepdims=True)


# =====================================================================
# PART 3: BATCH NORMALIZATION IMPLEMENTATION
# =====================================================================

class BatchNorm2D:
    """
    BatchNorm for 2D conv features.
    Normalize per-channel; use running mean/var at inference time.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (xp)
        self.gamma = xp.ones((1, num_features, 1, 1), dtype=xp.float32)
        self.beta  = xp.zeros((1, num_features, 1, 1), dtype=xp.float32)

        # Running statistics for inference (xp)
        self.running_mean = xp.zeros((1, num_features, 1, 1), dtype=xp.float32)
        self.running_var  = xp.ones((1, num_features, 1, 1), dtype=xp.float32)

        # Gradient buffers
        self.dgamma = None
        self.dbeta  = None

        # Cache
        self.cache = None
        self.training = True

    def forward(self, x):
        if self.training:
            batch_mean = xp.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var  = xp.var(x, axis=(0, 2, 3), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * batch_var

            x_norm = (x - batch_mean) / xp.sqrt(batch_var + self.eps)
            out = self.gamma * x_norm + self.beta
            self.cache = (x, x_norm, batch_mean, batch_var)
        else:
            x_norm = (x - self.running_mean) / xp.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout):
        x, x_norm, mean, var = self.cache
        N, C, H, W = x.shape

        self.dgamma = xp.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
        self.dbeta  = xp.sum(dout,           axis=(0, 2, 3), keepdims=True)

        dx_norm = dout * self.gamma
        std = xp.sqrt(var + self.eps)
        dx = (1.0 / (N * H * W)) * (1.0 / std) * (
            N * H * W * dx_norm
            - xp.sum(dx_norm, axis=(0, 2, 3), keepdims=True)
            - x_norm * xp.sum(dx_norm * x_norm, axis=(0, 2, 3), keepdims=True)
        )
        return dx


# =====================================================================
# PART 4: CONVOLUTION AND POOLING OPERATIONS
# =====================================================================

def im2col(x, kernel_h, kernel_w, stride=1, pad=0):
    """Image-to-column transform for GEMM-based convolution."""
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    if pad > 0:
        x = xp.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

    col = xp.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=xp.float32)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x_idx in range(kernel_w):
            x_max = x_idx + stride * out_w
            col[:, :, y, x_idx, :, :] = x[:, :, y:y_max:stride, x_idx:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, kernel_h, kernel_w, stride=1, pad=0):
    """Column-to-image inverse transform."""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    img = xp.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=xp.float32)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Conv2D:
    """2D convolution layer with manual backprop."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride  = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        self.W = (xp.random.randn(out_channels, in_channels, kernel_size, kernel_size)
                  * xp.sqrt(2.0 / fan_in)).astype(xp.float32)
        self.b = xp.zeros((out_channels, 1), dtype=xp.float32)

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)
        W_col = self.W.reshape(self.out_channels, -1)

        out = xp.dot(col, W_col.T) + self.b.T
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.cache = (x, col)
        return out

    def backward(self, dout):
        x, col = self.cache
        N, C, H, W = x.shape

        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.dW = xp.dot(dout_reshaped.T, col).reshape(self.W.shape)
        self.db = xp.sum(dout_reshaped, axis=0).reshape(self.b.shape)

        W_col = self.W.reshape(self.out_channels, -1)
        dcol = xp.dot(dout_reshaped, W_col)
        dx = col2im(dcol, x.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        return dx


class MaxPool2D:
    """2D max pooling."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        col = im2col(x, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size * self.pool_size)

        out = xp.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.cache = (x, col, out)
        return out

    def backward(self, dout):
        x, col, out = self.cache
        N, C, H, W = x.shape

        col_r = col.reshape(-1, self.pool_size * self.pool_size)
        col_max = xp.max(col_r, axis=1, keepdims=True)
        mask = (col_r == col_max)

        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1)
        dcol = mask * dout_flat[:, None]

        dx = col2im(dcol.reshape(col.shape), x.shape, self.pool_size, self.pool_size, self.stride, 0)
        return dx


class GlobalAvgPool2D:
    """Global average pooling."""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        out = xp.mean(x, axis=(2, 3))
        self.cache = (x.shape, out.copy())
        return out

    def backward(self, dout):
        (N, C, H, W), _ = self.cache
        dx = xp.ones((N, C, H, W), dtype=xp.float32) * (dout[:, :, None, None] / (H * W))
        return dx


# =====================================================================
# PART 5: RESIDUAL BLOCK IMPLEMENTATION
# =====================================================================

class ResidualBlock:
    """
    Standard residual block.
    Flow: Conv -> BN -> ReLU -> Conv -> BN -> add identity -> ReLU.
    Use a 1x1 projection when channels/stride do not match.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = BatchNorm2D(out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.bn_shortcut = BatchNorm2D(out_channels)

        self.cache = {}

    def forward(self, x):
        identity = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        self.cache['relu1_input'] = out
        out = ReLU.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.shortcut is not None:
            identity = self.shortcut.forward(identity)
            identity = self.bn_shortcut.forward(identity)

        out = out + identity
        self.cache['relu2_input'] = out
        out = ReLU.forward(out)

        self.cache['identity'] = identity
        return out

    def backward(self, dout):
        dout = dout * ReLU.backward(self.cache['relu2_input'])

        didentity = dout

        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = dout * ReLU.backward(self.cache['relu1_input'])
        dout = self.bn1.backward(dout)
        dx_main = self.conv1.backward(dout)

        if self.shortcut is not None:
            didentity = self.bn_shortcut.backward(didentity)
            dx_shortcut = self.shortcut.backward(didentity)
        else:
            dx_shortcut = didentity

        dx = dx_main + dx_shortcut
        return dx

    def get_gradient_norm(self):
        norms = {}
        norms['conv1_W'] = float(to_np(xp.linalg.norm(self.conv1.dW))) if self.conv1.dW is not None else 0.0
        norms['conv2_W'] = float(to_np(xp.linalg.norm(self.conv2.dW))) if self.conv2.dW is not None else 0.0
        if self.shortcut is not None:
            norms['shortcut_W'] = float(to_np(xp.linalg.norm(self.shortcut.dW))) if self.shortcut.dW is not None else 0.0
        return norms


# =====================================================================
# PART 6: RESNET MODEL ARCHITECTURE
# =====================================================================

class ResNet:
    """
    ResNet for CIFAR-10.
    Layout:
      Initial conv: 3x3, 16
      Three stages: [16, 32, 64] with 2 blocks each; first block of stages 2 and 3 uses stride 2
      Global average pooling followed by a fully connected layer
    """

    def __init__(self, num_classes=10):
        self.num_classes = num_classes

        self.conv1 = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = BatchNorm2D(16)

        self.stage1 = self._make_stage(16, 16, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(16, 32, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(32, 64, num_blocks=2, stride=2)

        self.global_pool = GlobalAvgPool2D()

        self.fc = self._init_fc_layer(64, num_classes)

        self.layer_outputs = {}
        self.gradient_norms = defaultdict(list)
        self.fc_input = None

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        blocks = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return blocks

    def _init_fc_layer(self, in_features, out_features):
        W = (xp.random.randn(out_features, in_features) * xp.sqrt(2.0 / in_features)).astype(xp.float32)
        b = xp.zeros(out_features, dtype=xp.float32)
        return {'W': W, 'b': b, 'dW': None, 'db': None}

    def forward(self, x, training=True):
        self._set_training_mode(training)

        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = ReLU.forward(x)
        self.layer_outputs['initial'] = x.copy()

        for i, block in enumerate(self.stage1):
            x = block.forward(x)
            self.layer_outputs[f'stage1_block{i}'] = x.copy()

        for i, block in enumerate(self.stage2):
            x = block.forward(x)
            self.layer_outputs[f'stage2_block{i}'] = x.copy()

        for i, block in enumerate(self.stage3):
            x = block.forward(x)
            self.layer_outputs[f'stage3_block{i}'] = x.copy()

        x = self.global_pool.forward(x)
        self.fc_input = x.copy()

        logits = xp.dot(x, self.fc['W'].T) + self.fc['b']
        return logits

    def backward(self, dout):
        self.fc['db'] = xp.sum(dout, axis=0)

        if self.fc_input is not None:
            self.fc['dW'] = xp.dot(dout.T, self.fc_input)
        else:
            _, pooled = self.global_pool.cache
            self.fc['dW'] = xp.dot(dout.T, pooled)

        dx = xp.dot(dout, self.fc['W'])
        dx = self.global_pool.backward(dx)

        for block in reversed(self.stage3):
            dx = block.backward(dx)
            self._record_gradient_norms('stage3', block)

        for block in reversed(self.stage2):
            dx = block.backward(dx)
            self._record_gradient_norms('stage2', block)

        for block in reversed(self.stage1):
            dx = block.backward(dx)
            self._record_gradient_norms('stage1', block)

        dx = dx * ReLU.backward(self.layer_outputs['initial'])
        dx = self.bn1.backward(dx)
        dx = self.conv1.backward(dx)
        return dx

    def _set_training_mode(self, training):
        self.bn1.training = training
        for stage in [self.stage1, self.stage2, self.stage3]:
            for block in stage:
                block.bn1.training = training
                block.bn2.training = training
                if hasattr(block, 'bn_shortcut'):
                    block.bn_shortcut.training = training

    def _record_gradient_norms(self, stage_name, block):
        norms = block.get_gradient_norm()
        for key, value in norms.items():
            self.gradient_norms[f'{stage_name}_{key}'].append(float(value))

    def get_all_parameters(self):
        params = OrderedDict()
        params['conv1_W'] = self.conv1.W
        params['conv1_b'] = self.conv1.b
        params['bn1_gamma'] = self.bn1.gamma
        params['bn1_beta']  = self.bn1.beta

        for stage_idx, stage in enumerate([self.stage1, self.stage2, self.stage3], 1):
            for block_idx, block in enumerate(stage):
                prefix = f'stage{stage_idx}_block{block_idx}'
                params[f'{prefix}_conv1_W'] = block.conv1.W
                params[f'{prefix}_conv1_b'] = block.conv1.b
                params[f'{prefix}_conv2_W'] = block.conv2.W
                params[f'{prefix}_conv2_b'] = block.conv2.b
                params[f'{prefix}_bn1_gamma'] = block.bn1.gamma
                params[f'{prefix}_bn1_beta']  = block.bn1.beta
                params[f'{prefix}_bn2_gamma'] = block.bn2.gamma
                params[f'{prefix}_bn2_beta']  = block.bn2.beta
                if hasattr(block, 'shortcut') and block.shortcut is not None:
                    params[f'{prefix}_shortcut_W'] = block.shortcut.W
                    params[f'{prefix}_shortcut_b'] = block.shortcut.b
                if hasattr(block, 'bn_shortcut') and block.bn_shortcut is not None:
                    params[f'{prefix}_bn_shortcut_gamma'] = block.bn_shortcut.gamma
                    params[f'{prefix}_bn_shortcut_beta']  = block.bn_shortcut.beta

        params['fc_W'] = self.fc['W']
        params['fc_b'] = self.fc['b']
        return params

    def get_all_gradients(self):
        grads = OrderedDict()
        grads['conv1_W'] = self.conv1.dW
        grads['conv1_b'] = self.conv1.db
        grads['bn1_gamma'] = self.bn1.dgamma
        grads['bn1_beta']  = self.bn1.dbeta

        for stage_idx, stage in enumerate([self.stage1, self.stage2, self.stage3], 1):
            for block_idx, block in enumerate(stage):
                prefix = f'stage{stage_idx}_block{block_idx}'
                grads[f'{prefix}_conv1_W'] = block.conv1.dW
                grads[f'{prefix}_conv1_b'] = block.conv1.db
                grads[f'{prefix}_conv2_W'] = block.conv2.dW
                grads[f'{prefix}_conv2_b'] = block.conv2.db
                grads[f'{prefix}_bn1_gamma'] = block.bn1.dgamma
                grads[f'{prefix}_bn1_beta']  = block.bn1.dbeta
                grads[f'{prefix}_bn2_gamma'] = block.bn2.dgamma
                grads[f'{prefix}_bn2_beta']  = block.bn2.dbeta
                if hasattr(block, 'shortcut') and block.shortcut is not None:
                    grads[f'{prefix}_shortcut_W'] = block.shortcut.dW
                    grads[f'{prefix}_shortcut_b'] = block.shortcut.db
                if hasattr(block, 'bn_shortcut') and block.bn_shortcut is not None:
                    grads[f'{prefix}_bn_shortcut_gamma'] = block.bn_shortcut.dgamma
                    grads[f'{prefix}_bn_shortcut_beta']  = block.bn_shortcut.dbeta

        grads['fc_W'] = self.fc['dW']
        grads['fc_b'] = self.fc['db']
        return grads


# =====================================================================
# PART 7: MANUAL OPTIMIZER IMPLEMENTATION
# =====================================================================

class SGDMomentum:
    """Momentum SGD (manual) with L2 weight decay."""

    def __init__(self, learning_rate=0.1, momentum=0.9, weight_decay=1e-4):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, model):
        params = model.get_all_parameters()
        grads  = model.get_all_gradients()

        for name in params.keys():
            p = params[name]
            g = grads.get(name, None)
            if g is None:
                continue

            if name not in self.velocities:
                self.velocities[name] = xp.zeros_like(p)

            if 'W' in name and g.shape == p.shape:
                g = g + self.weight_decay * p

            self.velocities[name] = self.momentum * self.velocities[name] - self.learning_rate * g
            p += self.velocities[name]


class LearningRateScheduler:
    """Step-decay learning-rate scheduler."""
    def __init__(self, initial_lr, decay_epochs, decay_factor):
        self.initial_lr = initial_lr
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor

    def get_lr(self, epoch):
        num_decays = epoch // self.decay_epochs
        return self.initial_lr * (self.decay_factor ** num_decays)


# =====================================================================
# PART 8: LOSS FUNCTION AND METRICS
# =====================================================================

def cross_entropy_loss(logits, labels):
    """Cross-entropy loss and gradient (returns Python float loss)."""
    batch_size = logits.shape[0]
    probs = Softmax.forward(logits)
    probs = xp.clip(probs, 1e-10, 1.0)

    labels_onehot = xp.zeros_like(probs)
    labels_onehot[xp.arange(batch_size), labels] = 1

    loss = -xp.sum(labels_onehot * xp.log(probs)) / batch_size
    loss = float(to_np(loss))

    grad = (probs - labels_onehot) / batch_size
    return loss, grad


def compute_accuracy(logits, labels):
    """Compute classification accuracy (percentage)."""
    preds = xp.argmax(logits, axis=1)
    acc = xp.mean((preds == labels).astype(xp.float32))
    return float(to_np(acc)) * 100.0


# =====================================================================
# PART 9: DATA AUGMENTATION
# =====================================================================

def data_augmentation(images, training=True):
    """Basic augmentation: random horizontal flip and random crop with reflect padding."""
    if not training:
        return images

    augmented = []
    for img in images:
        if xp.random.rand() > 0.5:
            img = img[:, :, ::-1]
        if xp.random.rand() > 0.5:
            pad = 4
            try:
                img_padded = xp.pad(img, ((0,0),(pad,pad),(pad,pad)), mode='reflect')
            except Exception:
                img_padded = xp.pad(img, ((0,0),(pad,pad),(pad,pad)), mode='edge')
            cx = int(xp.random.randint(0, 2*pad))
            cy = int(xp.random.randint(0, 2*pad))
            img = img_padded[:, cy:cy+32, cx:cx+32]
        augmented.append(img)
    return xp.array(augmented)


# =====================================================================
# PART 10: TRAINING LOOP
# =====================================================================

class Trainer:
    """Training and evaluation loop with history tracking for loss, accuracy, and gradient flow."""
    def __init__(self, model, optimizer, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loss_history = []
        self.train_acc_history  = []
        self.val_loss_history   = []
        self.val_acc_history    = []
        self.gradient_flow_history = defaultdict(list)

    def train_epoch(self, X_train, y_train, batch_size=128):
        num_samples = int(X_train.shape[0])
        num_batches = num_samples // batch_size

        epoch_loss = 0.0
        epoch_acc  = 0.0

        indices = xp.random.permutation(num_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for b in range(num_batches):
            s = b * batch_size
            e = s + batch_size
            X_batch = X_train[s:e]
            y_batch = y_train[s:e]

            X_batch = data_augmentation(X_batch, training=True)

            logits = self.model.forward(X_batch, training=True)
            loss, grad = cross_entropy_loss(logits, y_batch)

            self.model.backward(grad)
            self.optimizer.update(self.model)

            epoch_loss += loss
            epoch_acc  += compute_accuracy(logits, y_batch)

            if b % 10 == 0:
                self._track_gradient_flow()

        return epoch_loss / num_batches, epoch_acc / num_batches

    def evaluate(self, X, y, batch_size=128):
        """Evaluate on validation or test in mini-batches."""
        num_samples = int(X.shape[0])
        num_batches = (num_samples + batch_size - 1) // batch_size

        total_loss = 0.0
        total_acc  = 0.0

        for b in range(num_batches):
            s = b * batch_size
            e = min(s + batch_size, num_samples)
            X_batch = X[s:e]
            y_batch = y[s:e]

            logits = self.model.forward(X_batch, training=False)
            loss, _ = cross_entropy_loss(logits, y_batch)

            total_loss += loss * (e - s)
            total_acc  += compute_accuracy(logits, y_batch) * (e - s)

        return total_loss / num_samples, total_acc / num_samples

    def _track_gradient_flow(self):
        """Record norms of key weight gradients for gradient-flow comparison."""
        grads = self.model.get_all_gradients()
        for name, g in grads.items():
            if g is not None and 'W' in name:
                norm = float(to_np(xp.linalg.norm(g)))
                self.gradient_flow_history[name].append(norm)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128):
        """Full training with per-epoch validation."""
        print("\n" + "=" * 80)
        print("RESNET TRAINING ON CIFAR-10 (XP backend: {})".format(XP_BACKEND))
        print("=" * 80)
        print(f"Architecture: ResNet with 6 Residual Blocks")
        print(f"Training samples: {int(X_train.shape[0])}")
        print(f"Validation samples: {int(X_val.shape[0])}")
        print(f"Batch size: {batch_size}")
        print(f"Initial learning rate: {self.optimizer.learning_rate}")
        print("=" * 80 + "\n")

        for epoch in range(epochs):
            start = time.time()
            if self.lr_scheduler:
                self.optimizer.learning_rate = self.lr_scheduler.get_lr(epoch)

            tr_loss, tr_acc = self.train_epoch(X_train, y_train, batch_size)
            self.train_loss_history.append(tr_loss)
            self.train_acc_history.append(tr_acc)

            va_loss, va_acc = self.evaluate(X_val, y_val, batch_size)
            self.val_loss_history.append(va_loss)
            self.val_acc_history.append(va_acc)

            dt = time.time() - start
            print(f"Epoch {epoch+1}/{epochs} ({dt:.2f}s) | "
                  f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
                  f"Val Loss: {va_loss:.4f}  | Val Acc: {va_acc:.2f}%  | "
                  f"LR: {self.optimizer.learning_rate:.5f}")

        return self.train_loss_history, self.train_acc_history, self.val_loss_history, self.val_acc_history


# =====================================================================
# PART 11: VISUALIZATION
# =====================================================================

def plot_training_curves(train_loss, train_acc, val_loss, val_acc, test_acc=None):
    """Plot loss and accuracy curves for training and validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = np.arange(1, len(train_loss) + 1)

    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss,   'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc,   'r-', label='Validation Accuracy', linewidth=2)
    if test_acc is not None:
        ax2.axhline(y=test_acc, color='g', linestyle='--', label=f'Final Test Acc: {test_acc:.2f}%', linewidth=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Training and Validation Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nTraining curves saved as 'training_curves.png'")


# =====================================================================
# PART 12: NO-SKIP BASELINE (Requirement #6)
# =====================================================================

class ResidualBlock_NoSkip:
    """Two-conv block without skip connection; used as a baseline."""
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels,  out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = BatchNorm2D(out_channels)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = BatchNorm2D(out_channels)
        # Keep placeholder attributes for parity with the residual version to avoid attribute errors
        self.shortcut = None
        self.bn_shortcut = None
        self.cache = {}

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        self.cache['relu1_input'] = out
        out = ReLU.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        self.cache['relu2_input'] = out
        out = ReLU.forward(out)
        return out

    def backward(self, dout):
        dout = dout * ReLU.backward(self.cache['relu2_input'])
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = dout * ReLU.backward(self.cache['relu1_input'])
        dout = self.bn1.backward(dout)
        dx   = self.conv1.backward(dout)
        return dx

    def get_gradient_norm(self):
        return {
            'conv1_W': float(to_np(xp.linalg.norm(self.conv1.dW))) if self.conv1.dW is not None else 0.0,
            'conv2_W': float(to_np(xp.linalg.norm(self.conv2.dW))) if self.conv2.dW is not None else 0.0,
        }


class ResNetNoSkip(ResNet):
    """Same scaffold as ResNet but blocks do not use skip connections."""
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        # Still downsample in the first block of each stage via stride to keep feature sizes aligned
        blocks = [ResidualBlock_NoSkip(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock_NoSkip(out_channels, out_channels, stride=1))
        return blocks


def analyze_gradient_flow_comparison(grad_with, grad_without=None):
    """Plot gradient-flow comparison for models with and without skip connections."""
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)
    key_layer = 'stage3_block1_conv2_W'
    if key_layer in grad_with:
        ax1.plot(grad_with[key_layer][:100], 'b-', label='With Skip', linewidth=2)
    if grad_without and key_layer in grad_without:
        ax1.plot(grad_without[key_layer][:100], 'r--', label='Without Skip', linewidth=2)
    ax1.set_xlabel('Training Steps'); ax1.set_ylabel('Gradient Norm (log)'); ax1.set_yscale('log')
    ax1.set_title('Deep Layer Gradient Comparison'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    stages = ['Stage 1', 'Stage 2', 'Stage 3']
    means_with, means_without = [], []
    for s in range(1, 4):
        vals = []
        for name, v in grad_with.items():
            if f'stage{s}' in name and 'W' in name and len(v) > 0:
                vals.extend(v[-20:])
        means_with.append(np.mean(vals) if vals else 0.0)

        if grad_without:
            vals = []
            for name, v in grad_without.items():
                if f'stage{s}' in name and 'W' in name and len(v) > 0:
                    vals.extend(v[-20:])
            means_without.append(np.mean(vals) if vals else 0.0)

    x_pos = np.arange(len(stages))
    width = 0.35
    ax2.bar(x_pos - width/2, means_with,    width, label='With Skip',    alpha=0.7)
    if grad_without:
        ax2.bar(x_pos + width/2, means_without, width, label='Without Skip', alpha=0.7)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(stages)
    ax2.set_ylabel('Avg Grad Norm'); ax2.set_title('Gradient Distribution Across Depth')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3); ax3.axis('off')
    explanation = """
How Residual Connections Improve Gradient Flow:

No-skip:   y = F(x),  dL/dx = dL/dy * dF/dx  (may vanish)
With-skip: y = F(x) + x,  dL/dx = dL/dy * (dF/dx + I)  (identity preserves gradient)

Observed: deeper layers keep healthier gradient magnitudes; faster and more stable convergence.
"""
    ax3.text(0.03, 0.95, explanation, transform=ax3.transAxes,
             fontsize=10, va='top', family='monospace')

    ax4 = plt.subplot(2, 2, 4)
    layer_names, g_with, g_without = [], [], []
    for s in range(1, 4):
        for b in range(2):
            for c in (1, 2):
                name = f'stage{s}_block{b}_conv{c}_W'
                if name in grad_with and len(grad_with[name]) > 0:
                    layer_names.append(f'S{s}B{b}C{c}')
                    g_with.append(np.mean(grad_with[name][-20:]))
                    if grad_without and name in grad_without and len(grad_without[name]) > 0:
                        g_without.append(np.mean(grad_without[name][-20:]))
                    else:
                        g_without.append(0.0)
    if layer_names:
        x = np.arange(len(layer_names))
        ax4.plot(x, g_with, 'b-o', label='With Skip', markersize=5)
        if any(g_without):
            ax4.plot(x, g_without, 'r--s', label='Without Skip', markersize=5)
        ax4.set_xticks(x[::2]); ax4.set_xticklabels(layer_names[::2], rotation=45)
        ax4.set_yscale('log'); ax4.set_xlabel('Layer (Shallow → Deep)')
        ax4.set_ylabel('Avg Grad Norm'); ax4.set_title('Layer-wise Gradient Flow')
        ax4.legend(); ax4.grid(True, alpha=0.3)

    plt.suptitle('Gradient Flow Analysis: Impact of Residual Connections', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradient_flow_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gradient flow analysis saved as 'gradient_flow_analysis.png'")


# =====================================================================
# PART 13: MAIN EXECUTION
# =====================================================================

def main():
    """End-to-end training pipeline with baseline and visualizations."""
    print("Loading CIFAR-10 dataset...")
    X_train_np, y_train_np, X_test_np, y_test_np = load_cifar10()
    X_train_np, y_train_np, X_val_np, y_val_np = create_validation_split(X_train_np, y_train_np, val_size=5000)

    print(f"Training samples: {X_train_np.shape}")
    print(f"Validation samples: {X_val_np.shape}")
    print(f"Test samples: {X_test_np.shape}")

    # Move to xp backend
    X_train = to_xp(X_train_np.astype(np.float32))
    y_train = to_xp(y_train_np.astype(np.int64))
    X_val   = to_xp(X_val_np.astype(np.float32))
    y_val   = to_xp(y_val_np.astype(np.int64))
    X_test  = to_xp(X_test_np.astype(np.float32))
    y_test  = to_xp(y_test_np.astype(np.int64))

    # Build model and optimizer
    print("\nInitializing ResNet model (with skip connections)...")
    model = ResNet(num_classes=10)

    print("\n" + "=" * 80)
    print("RESNET ARCHITECTURE DETAILS")
    print("=" * 80)
    print("Layer Configuration:")
    print("  - Initial: Conv(3→16, 3×3, stride=1, padding=1) → BatchNorm → ReLU")
    print("  - Stage 1: 2 × ResBlock(16→16)")
    print("  - Stage 2: 2 × ResBlock(16→32, first stride=2)")
    print("  - Stage 3: 2 × ResBlock(32→64, first stride=2)")
    print("  - Output: GlobalAvgPool → FC(64→10)")
    print("\nRegularization: BatchNorm, L2 Weight Decay 1e-4, Data Aug (flip+crop)")
    print("=" * 80 + "\n")

    lr_scheduler = LearningRateScheduler(initial_lr=0.1, decay_epochs=30, decay_factor=0.1)
    optimizer    = SGDMomentum(learning_rate=0.1, momentum=0.9, weight_decay=1e-4)
    trainer      = Trainer(model, optimizer, lr_scheduler)

    print("Starting training...")
    train_loss, train_acc, val_loss, val_acc = trainer.train(
        X_train, y_train, X_val, y_val, epochs=50, batch_size=128
    )

    print("\n" + "=" * 80)
    print("FINAL EVALUATION (With Skip)")
    print("=" * 80)
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("=" * 80 + "\n")

    print("Generating visualizations...")
    plot_training_curves(train_loss, train_acc, val_loss, val_acc, test_acc)

    # No-skip baseline (small subset + few epochs)
    print("\nTraining comparison model WITHOUT skip connections on a small subset...")
    subset = min(8000, int(X_train.shape[0]))
    X_train_small = X_train[:subset]
    y_train_small = y_train[:subset]
    X_val_small   = X_val[:2000]
    y_val_small   = y_val[:2000]

    model_ns   = ResNetNoSkip(num_classes=10)
    opt_ns     = SGDMomentum(learning_rate=0.1, momentum=0.9, weight_decay=1e-4)
    sch_ns     = LearningRateScheduler(initial_lr=0.1, decay_epochs=20, decay_factor=0.1)
    trainer_ns = Trainer(model_ns, opt_ns, sch_ns)

    _ = trainer_ns.train(X_train_small, y_train_small, X_val_small, y_val_small, epochs=10, batch_size=128)
    _, test_acc_ns = trainer_ns.evaluate(X_test, y_test)

    print("\nComparison (No Skip) - Test Accuracy: {:.2f}%".format(test_acc_ns))
    analyze_gradient_flow_comparison(trainer.gradient_flow_history, trainer_ns.gradient_flow_history)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"✓ ResNet (with skip) Final Test Accuracy: {test_acc:.2f}%")
    print(f"✓ No-Skip Baseline (subset) Test Accuracy: {test_acc_ns:.2f}%")
    print(f"✓ Gradient-flow comparison figures saved.")
    print("=" * 80)

    return model, test_acc


if __name__ == "__main__":
    model, test_accuracy = main()
