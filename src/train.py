import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Monomial order (degree 3):
# 0: X^3, 1: X^2Y, 2: X^2Z, 3: XY^2, 4: XYZ, 5: XZ^2, 6: Y^3, 7: Y^2Z, 8: YZ^2, 9: Z^3


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    # Only numpy/random used


@dataclass
class DatasetConfig:
    n_samples: int = 10000
    frac_singular: float = 0.30
    frac_nodal_within_singular: float = 0.40
    coef_scale: float = 1.0  # std dev of normal for real/imag parts


def rand_complex(scale: float = 1.0) -> complex:
    return np.complex128(np.random.normal(0, scale) + 1j * np.random.normal(0, scale))


def sample_generic(scale: float = 1.0) -> np.ndarray:
    # 10 complex coefficients a0..a9
    re = np.random.normal(0, scale, size=10)
    im = np.random.normal(0, scale, size=10)
    coeffs = re + 1j * im
    return coeffs.astype(np.complex128)


def sample_singular_nodal(scale: float = 1.0) -> np.ndarray:
    # Enforce a0=a1=a2=0; choose a3,a4,a5 s.t. discriminant != 0; remaining free
    a = np.zeros(10, dtype=np.complex128)
    # Quadratic part
    while True:
        a3, a4, a5 = rand_complex(scale), rand_complex(scale), rand_complex(scale)
        disc = a4 * a4 - 4 * a3 * a5
        if disc != 0:  # over C, symbolic inequality; numerical exactness okay for random draws
            break
    a[3], a[4], a[5] = a3, a4, a5
    # Cubic terms free
    a[6:] = sample_generic(scale)[6:]
    return a


def sample_singular_non_nodal(scale: float = 1.0) -> np.ndarray:
    # Enforce a0=a1=a2=0; set quadratic part with discriminant == 0 (double line or higher)
    a = np.zeros(10, dtype=np.complex128)
    # With prob 0.5 make quadratic part identically 0; else make it a square of a linear form
    if np.random.rand() < 0.5:
        a3 = a4 = a5 = 0.0 + 0.0j
    else:
        # Q = λ (u y + v z)^2 => a3=λ u^2, a4=2λuv, a5=λ v^2 ; ensures Δ=0
        lam = rand_complex(scale)
        u = rand_complex(scale)
        v = rand_complex(scale)
        a3 = lam * (u ** 2)
        a4 = 2 * lam * u * v
        a5 = lam * (v ** 2)
    a[3], a[4], a[5] = a3, a4, a5
    # Cubic terms free
    a[6:] = sample_generic(scale)[6:]
    return a


def label_nodal_at_100(a: np.ndarray) -> int:
    # Algebraic label per description
    # Singular at (1:0:0) iff a0=a1=a2=0
    if not (np.isclose(a[0], 0).all() and np.isclose(a[1], 0).all() and np.isclose(a[2], 0).all()):
        return 0
    # Quadratic discriminant
    a3, a4, a5 = a[3], a[4], a[5]
    disc = a4 * a4 - 4 * a3 * a5
    return int(not np.isclose(disc, 0))


def build_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = cfg.n_samples
    n_sing = int(round(cfg.frac_singular * n))
    n_nodal = int(round(cfg.frac_nodal_within_singular * n_sing))
    n_non_nodal = n_sing - n_nodal
    n_generic = n - n_sing

    coeffs_list = []
    labels_list = []

    # Singular nodal
    for _ in range(n_nodal):
        a = sample_singular_nodal(cfg.coef_scale)
        coeffs_list.append(a)
        labels_list.append(1)

    # Singular non-nodal
    for _ in range(n_non_nodal):
        a = sample_singular_non_nodal(cfg.coef_scale)
        coeffs_list.append(a)
        labels_list.append(0)

    # Generic (no constraints); label algebraically
    for _ in range(n_generic):
        a = sample_generic(cfg.coef_scale)
        coeffs_list.append(a)
        labels_list.append(label_nodal_at_100(a))

    coeffs = np.stack(coeffs_list, axis=0)  # (N,10) complex
    labels = np.array(labels_list, dtype=np.int64)

    # Shuffle dataset
    idx = np.random.permutation(len(labels))
    coeffs = coeffs[idx]
    labels = labels[idx]

    # Convert to 20 real features: interleave Re/Im per coefficient
    feats = np.empty((coeffs.shape[0], 20), dtype=np.float32)
    feats[:, 0::2] = coeffs.real.astype(np.float32)
    feats[:, 1::2] = coeffs.imag.astype(np.float32)

    return feats, labels, coeffs


# ---- Pure NumPy MLP with Adam and BCE-with-logits ----

class NumpyMLP:
    def __init__(self, in_dim: int = 20, hidden: int = 512, weight_decay: float = 1e-4, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.in_dim = in_dim
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # He init for ReLU
        self.W1 = np.random.randn(in_dim, hidden).astype(np.float32) * math.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = np.random.randn(hidden, 1).astype(np.float32) * math.sqrt(2.0 / hidden)
        self.b2 = np.zeros((1,), dtype=np.float32)

        # Adam states
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)
        self.t = 0

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # stable sigmoid
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        z = np.empty_like(x, dtype=np.float32)
        z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        expx = np.exp(x[neg_mask])
        z[neg_mask] = expx / (1.0 + expx)
        return z

    @staticmethod
    def bce_with_logits(logits: np.ndarray, y: np.ndarray) -> float:
        # logits: (N,)
        # y: (N,) in {0,1}
        # stable formulation: max(l,0) - l*y + log(1+exp(-|l|))
        l = logits
        t = y
        m = np.maximum(l, 0.0)
        loss = m - l * t + np.log1p(np.exp(-np.abs(l)))
        return float(np.mean(loss))

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1  # (N,H)
        a1 = np.maximum(z1, 0.0)    # ReLU
        logits = (a1 @ self.W2 + self.b2).squeeze(-1)  # (N,)
        cache = (X, z1, a1)
        return logits, cache

    def step(self, grads):
        (gW1, gb1, gW2, gb2) = grads
        self.t += 1
        # Adam for W1
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * gW1
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (gW1 * gW1)
        mW1_hat = self.mW1 / (1 - self.beta1 ** self.t)
        vW1_hat = self.vW1 / (1 - self.beta2 ** self.t)
        self.W1 -= self.lr * mW1_hat / (np.sqrt(vW1_hat) + self.eps)
        # Adam for b1
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * gb1
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (gb1 * gb1)
        mb1_hat = self.mb1 / (1 - self.beta1 ** self.t)
        vb1_hat = self.vb1 / (1 - self.beta2 ** self.t)
        self.b1 -= self.lr * mb1_hat / (np.sqrt(vb1_hat) + self.eps)
        # Adam for W2
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * gW2
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (gW2 * gW2)
        mW2_hat = self.mW2 / (1 - self.beta1 ** self.t)
        vW2_hat = self.vW2 / (1 - self.beta2 ** self.t)
        self.W2 -= self.lr * mW2_hat / (np.sqrt(vW2_hat) + self.eps)
        # Adam for b2
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * gb2
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (gb2 * gb2)
        mb2_hat = self.mb2 / (1 - self.beta1 ** self.t)
        vb2_hat = self.vb2 / (1 - self.beta2 ** self.t)
        self.b2 -= self.lr * mb2_hat / (np.sqrt(vb2_hat) + self.eps)

    def backward(self, cache, logits: np.ndarray, y: np.ndarray, l2: float):
        X, z1, a1 = cache
        N = X.shape[0]
        # d logits for BCE-with-logits
        p = self.sigmoid(logits)
        dlogits = (p - y).reshape(-1, 1)  # (N,1)
        # Gradients
        gW2 = (a1.T @ dlogits) / N + l2 * self.W2
        gb2 = np.mean(dlogits, axis=0)
        da1 = dlogits @ self.W2.T  # (N,H)
        dz1 = da1 * (z1 > 0.0)
        gW1 = (X.T @ dz1) / N + l2 * self.W1
        gb1 = np.mean(dz1, axis=0)
        return gW1.astype(np.float32), gb1.astype(np.float32), gW2.astype(np.float32), gb2.astype(np.float32)


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, seed: int = 42):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    # Stratified split
    pos_idx = idx[y == 1]
    neg_idx = idx[y == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_test = int(round(len(pos_idx) * test_size))
    n_neg_test = int(round(len(neg_idx) * test_size))
    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true.astype(np.int64) == y_pred.astype(np.int64)).astype(np.float32)))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Handle edge cases
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    s_pos = y_score[pos]
    s_neg = y_score[neg]
    # Pairwise comparison AUC (with ties as 0.5)
    # This uses broadcasting; memory ~ n_pos*n_neg floats
    cmp = (s_pos[:, None] > s_neg[None, :]).astype(np.float32)
    ties = (s_pos[:, None] == s_neg[None, :]).astype(np.float32)
    auc = (np.sum(cmp) + 0.5 * np.sum(ties)) / (n_pos * n_neg)
    return float(auc)


def iterate_minibatches(X, y, batch_size: int, shuffle: bool = True, seed: int = 42):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]


def main():
    parser = argparse.ArgumentParser(description="Train MLP to detect nodal singularity at (1:0:0) for cubic surfaces.")
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--test-split', type=float, default=0.25)
    parser.add_argument('--coef-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-disc-feature', action='store_true', help='Append discriminant Δ as explicit input features (Re, Im)')
    args = parser.parse_args()

    seed_all(args.seed)

    # Dataset
    ds_cfg = DatasetConfig(
        n_samples=args.n_samples,
        frac_singular=0.30,
        frac_nodal_within_singular=0.40,
        coef_scale=args.coef_scale,
    )
    X, y, coeffs_complex = build_dataset(ds_cfg)

    # Optionally append discriminant features (Re Δ, Im Δ)
    if args.include_disc_feature:
        disc = coeffs_complex[:, 4] * coeffs_complex[:, 4] - 4 * coeffs_complex[:, 3] * coeffs_complex[:, 5]
        disc_feats = np.stack([disc.real.astype(np.float32), disc.imag.astype(np.float32)], axis=1)
        X = np.concatenate([X, disc_feats], axis=1)

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=args.test_split, seed=args.seed)

    # Model (NumPy)
    model = NumpyMLP(in_dim=X.shape[1], hidden=args.hidden, weight_decay=args.weight_decay, lr=args.lr)

    # Train
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    test_aucs = []
    # Pre-generate synthetic test sets for monomial-specific accuracy curves (fixed size for efficiency)
    feat_dim = X.shape[1]
    def build_synthetic_monomial_dataset(mono_idx: int, samples: int, seed: int):
        rng = np.random.RandomState(seed)
        feats = np.empty((samples, feat_dim), dtype=np.float32)
        labels_syn = np.empty((samples,), dtype=np.int64)
        for i in range(samples):
            a = np.zeros((10,), dtype=np.complex128)
            a[mono_idx] = np.complex128(rng.normal(0, 1.0) + 1j * rng.normal(0, 1.0))
            # features
            f = np.empty((20,), dtype=np.float32)
            f[0::2] = a.real.astype(np.float32)
            f[1::2] = a.imag.astype(np.float32)
            if args.include_disc_feature:
                d = a[4] * a[4] - 4 * a[3] * a[5]
                f = np.concatenate([f, np.array([d.real, d.imag], dtype=np.float32)], axis=0)
            feats[i] = f.astype(np.float32)
            # labels
            labels_syn[i] = label_nodal_at_100(a)
        return feats, labels_syn

    sy_y3_feats, sy_y3_labels = build_synthetic_monomial_dataset(6, samples=300, seed=args.seed + 123)
    sy_y2z_feats, sy_y2z_labels = build_synthetic_monomial_dataset(7, samples=300, seed=args.seed + 456)
    y3_accs = []
    y2z_accs = []

    # Additional synthetic sets for XY^2 - XZ^2 and XY^2 - XZ^2 + Y^3
    def build_xy2_minus_xz2(samples: int, seed: int):
        rng = np.random.RandomState(seed)
        feats = np.empty((samples, feat_dim), dtype=np.float32)
        labels_syn = np.empty((samples,), dtype=np.int64)
        for i in range(samples):
            a = np.zeros((10,), dtype=np.complex128)
            c = np.complex128(rng.normal(0, 1.0) + 1j * rng.normal(0, 1.0))
            a[3] = c     # XY^2
            a[5] = -c    # XZ^2
            # features/labels
            f = np.empty((20,), dtype=np.float32)
            f[0::2] = a.real.astype(np.float32)
            f[1::2] = a.imag.astype(np.float32)
            if args.include_disc_feature:
                d = a[4] * a[4] - 4 * a[3] * a[5]
                f = np.concatenate([f, np.array([d.real, d.imag], dtype=np.float32)], axis=0)
            feats[i] = f.astype(np.float32)
            labels_syn[i] = label_nodal_at_100(a)
        return feats, labels_syn

    def build_xy2_minus_xz2_plus_y3(samples: int, seed: int):
        rng = np.random.RandomState(seed)
        feats = np.empty((samples, feat_dim), dtype=np.float32)
        labels_syn = np.empty((samples,), dtype=np.int64)
        for i in range(samples):
            a = np.zeros((10,), dtype=np.complex128)
            c = np.complex128(rng.normal(0, 1.0) + 1j * rng.normal(0, 1.0))
            a[3] = c     # XY^2
            a[5] = -c    # XZ^2
            a[6] = np.complex128(rng.normal(0, 1.0) + 1j * rng.normal(0, 1.0))  # Y^3
            # features/labels
            f = np.empty((20,), dtype=np.float32)
            f[0::2] = a.real.astype(np.float32)
            f[1::2] = a.imag.astype(np.float32)
            if args.include_disc_feature:
                d = a[4] * a[4] - 4 * a[3] * a[5]
                f = np.concatenate([f, np.array([d.real, d.imag], dtype=np.float32)], axis=0)
            feats[i] = f.astype(np.float32)
            labels_syn[i] = label_nodal_at_100(a)
        return feats, labels_syn

    xy2mxz2_feats, xy2mxz2_labels = build_xy2_minus_xz2(300, seed=args.seed + 789)
    xy2mxz2py3_feats, xy2mxz2py3_labels = build_xy2_minus_xz2_plus_y3(300, seed=args.seed + 101)
    xy2mxz2_accs = []
    xy2mxz2py3_accs = []

    # Singular-only sensitivity curves (track per epoch for a3,a4,a5)
    sens_a3_epochs = []
    sens_a4_epochs = []
    sens_a5_epochs = []

    # Helper: feature sensitivity by monomial (defined here for use inside epoch loop)
    def feature_sensitivity_by_monomial(model: 'NumpyMLP', Xsample: np.ndarray) -> np.ndarray:
        # Returns size-10 array: sensitivity per monomial, combining Re/Im by L1 sum
        logits, cache = model.forward(Xsample)
        Xc, z1, a1 = cache
        mask = (z1 > 0.0).astype(np.float32)  # (N,H)
        w2 = model.W2.squeeze(1)              # (H,)
        dlogit_dz1 = mask * w2[None, :]       # (N,H)
        dlogit_dX = dlogit_dz1 @ model.W1.T   # (N,D)
        grad_abs = np.mean(np.abs(dlogit_dX), axis=0)  # (D,)
        # Combine real/imag pairs into monomial score
        scores = []
        for j in range(10):
            r = grad_abs[2*j]
            im = grad_abs[2*j + 1]
            scores.append(r + im)
        return np.array(scores, dtype=np.float32)

    for epoch in range(1, args.epochs + 1):
        # Training epoch
        tr_losses = []
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size=args.batch_size, shuffle=True, seed=(args.seed + epoch)):
            logits, cache = model.forward(xb)
            loss = model.bce_with_logits(logits, yb)
            # Add L2 regularization loss (0.5 * lambda * ||W||^2) not included in BCE part
            l2_loss = 0.5 * args.weight_decay * (np.sum(model.W1 * model.W1) + np.sum(model.W2 * model.W2))
            total_loss = loss + l2_loss
            grads = model.backward(cache, logits, yb, l2=args.weight_decay)
            model.step(grads)
            tr_losses.append(total_loss)

        # Evaluation
        logits_train, _ = model.forward(X_train)
        train_loss = model.bce_with_logits(logits_train, y_train) + 0.5 * args.weight_decay * (
            np.sum(model.W1 * model.W1) + np.sum(model.W2 * model.W2)
        )
        p_test, _ = model.forward(X_test)
        test_loss = model.bce_with_logits(p_test, y_test) + 0.5 * args.weight_decay * (
            np.sum(model.W1 * model.W1) + np.sum(model.W2 * model.W2)
        )
        train_prob = NumpyMLP.sigmoid(logits_train)
        test_prob = NumpyMLP.sigmoid(p_test)
        train_acc = accuracy(y_train, (train_prob >= 0.5).astype(np.int64))
        test_acc = accuracy(y_test, (test_prob >= 0.5).astype(np.int64))
        test_auc = roc_auc(y_test, test_prob)

        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
        test_aucs.append(float(test_auc))

        # Monomial-specific testing curves (Y^3 and Y^2·Z) per epoch
        y3_logits, _ = model.forward(sy_y3_feats)
        y3_probs = NumpyMLP.sigmoid(y3_logits)
        y3_preds = (y3_probs >= 0.5).astype(np.int64)
        y3_accs.append(accuracy(sy_y3_labels, y3_preds))

        y2z_logits, _ = model.forward(sy_y2z_feats)
        y2z_probs = NumpyMLP.sigmoid(y2z_logits)
        y2z_preds = (y2z_probs >= 0.5).astype(np.int64)
        y2z_accs.append(accuracy(sy_y2z_labels, y2z_preds))

        # Expression-specific testing curves (XY^2 - XZ^2) and (XY^2 - XZ^2 + Y^3)
        e1_logits, _ = model.forward(xy2mxz2_feats)
        e1_probs = NumpyMLP.sigmoid(e1_logits)
        e1_preds = (e1_probs >= 0.5).astype(np.int64)
        xy2mxz2_accs.append(accuracy(xy2mxz2_labels, e1_preds))

        e2_logits, _ = model.forward(xy2mxz2py3_feats)
        e2_probs = NumpyMLP.sigmoid(e2_logits)
        e2_preds = (e2_probs >= 0.5).astype(np.int64)
        xy2mxz2py3_accs.append(accuracy(xy2mxz2py3_labels, e2_preds))

        # Singular-only sensitivity for a3,a4,a5 this epoch
        re_te = X_test[:, 0::2]; im_te = X_test[:, 1::2]
        a_te = re_te + 1j * im_te
        sing_te = (np.isclose(a_te[:, 0], 0) & np.isclose(a_te[:, 1], 0) & np.isclose(a_te[:, 2], 0))
        if np.any(sing_te):
            sens_scores_epoch = feature_sensitivity_by_monomial(model, X_test[sing_te])
            sens_a3_epochs.append(float(sens_scores_epoch[3]))
            sens_a4_epochs.append(float(sens_scores_epoch[4]))
            sens_a5_epochs.append(float(sens_scores_epoch[5]))
        else:
            sens_a3_epochs.append(float('nan'))
            sens_a4_epochs.append(float('nan'))
            sens_a5_epochs.append(float('nan'))

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | test loss {test_loss:.4f} | acc {test_acc:.4f} | auc {test_auc:.4f}")

    # -------------------- Plotting --------------------
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    epochs = np.arange(1, args.epochs + 1)

    # 1) Loss curve
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss (with L2)')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    fig1.suptitle('Figure 1: Training and Test Loss Curves', fontsize=12)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    f1_path = os.path.join(out_dir, f'loss_curve_{stamp}.png')
    fig1.savefig(f1_path, dpi=150)
    plt.close(fig1)

    # 2) Training and testing accuracy
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(epochs, train_accs, label='Train Accuracy')
    ax2.plot(epochs, test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    fig2.suptitle('Figure 2: Training and Test Accuracy over Epochs', fontsize=12)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    f2_path = os.path.join(out_dir, f'accuracy_curve_{stamp}.png')
    fig2.savefig(f2_path, dpi=150)
    plt.close(fig2)

    # Helper to build feature vectors from complex coefficients a (10,)
    def coeffs_to_feats(a: np.ndarray) -> np.ndarray:
        f = np.empty((20,), dtype=np.float32)
        f[0::2] = a.real.astype(np.float32)
        f[1::2] = a.imag.astype(np.float32)
        return f

    # 3) Monomial-specific accuracy curves across epochs (Y^3 and Y^2·Z)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(epochs, y3_accs, label='Y^3', color='#4C78A8')
    ax3.plot(epochs, y2z_accs, label='Y^2·Z', color='#F58518')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend()
    ax3.set_title('Monomial Test Accuracy vs Epoch (Synthetic Sets)')
    fig3.suptitle('Figure 3: Accuracy Curves for Y^3 and Y^2·Z over Training', fontsize=12)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    f3_path = os.path.join(out_dir, f'monomial_accuracy_curves_{stamp}.png')
    fig3.savefig(f3_path, dpi=150)
    plt.close(fig3)

    # 3b) Expression-specific accuracy curves across epochs (XY^2 - XZ^2) and (XY^2 - XZ^2 + Y^3)
    fig3b, ax3b = plt.subplots(figsize=(8, 5))
    ax3b.plot(epochs, xy2mxz2_accs, label='XY^2 − XZ^2', color='#54A24B')
    ax3b.plot(epochs, xy2mxz2py3_accs, label='XY^2 − XZ^2 + Y^3', color='#E45756')
    ax3b.set_xlabel('Epoch')
    ax3b.set_ylabel('Accuracy')
    ax3b.set_ylim(0.0, 1.0)
    ax3b.grid(True, linestyle='--', alpha=0.4)
    ax3b.legend()
    ax3b.set_title('Expression Test Accuracy vs Epoch (Synthetic Sets)')
    fig3b.suptitle('Figure 3b: Accuracy Curves for XY^2 − XZ^2 and XY^2 − XZ^2 + Y^3', fontsize=12)
    fig3b.tight_layout(rect=[0, 0.03, 1, 0.95])
    f3b_path = os.path.join(out_dir, f'expression_accuracy_curves_{stamp}.png')
    fig3b.savefig(f3b_path, dpi=150)
    plt.close(fig3b)

    # 5) Interpretability: Input-gradient sensitivity per monomial (uses helper defined above)

    sens_scores = feature_sensitivity_by_monomial(model, X_test)
    monom_labels = ['X^3','X^2Y','X^2Z','XY^2','XYZ','XZ^2','Y^3','Y^2Z','YZ^2','Z^3']
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    bars5 = ax5.bar(monom_labels, sens_scores, color=['#bbb']*10)
    # Highlight the quadratic-part monomials y^2, yz, z^2 (indices 3,4,5)
    for idx, color in zip([3,4,5], ['#54A24B','#54A24B','#54A24B']):
        bars5[idx].set_color(color)
    ax5.set_ylabel('Avg |∂logit/∂input| (L1 over Re/Im)')
    ax5.set_title('Sensitivity by Monomial (higher = greater influence)')
    ax5.tick_params(axis='x', rotation=30)
    fig5.suptitle('Figure 5: Feature Sensitivity — Quadratic part (y^2, yz, z^2) highlighted', fontsize=12)
    fig5.tight_layout(rect=[0, 0.03, 1, 0.92])
    f5_path = os.path.join(out_dir, f'feature_sensitivity_{stamp}.png')
    fig5.savefig(f5_path, dpi=150)
    plt.close(fig5)

    # 5b) Singular-only sensitivity curves across epochs for y^2, yz, z^2
    fig5b, ax5b = plt.subplots(figsize=(8, 5))
    ax5b.plot(epochs, sens_a3_epochs, label='y^2 (a3)', color='#54A24B')
    ax5b.plot(epochs, sens_a4_epochs, label='yz (a4)', color='#2E7D32')
    ax5b.plot(epochs, sens_a5_epochs, label='z^2 (a5)', color='#66BB6A')
    ax5b.set_xlabel('Epoch')
    ax5b.set_ylabel('Avg |∂logit/∂input| (singular only)')
    ax5b.set_title('Singular-only Feature Sensitivity vs Epoch (Quadratic part)')
    ax5b.grid(True, linestyle='--', alpha=0.4)
    ax5b.legend()
    fig5b.suptitle('Figure 5b: Per-epoch Sensitivity on Singular Subset — y^2, yz, z^2', fontsize=12)
    fig5b.tight_layout(rect=[0, 0.03, 1, 0.95])
    f5b_path = os.path.join(out_dir, f'feature_sensitivity_singular_epochs_{stamp}.png')
    fig5b.savefig(f5b_path, dpi=150)
    plt.close(fig5b)

    # 6) Evidence of discriminant usage: predicted prob vs |Δ| for singular samples in test set
    # Select singular samples in test set
    re_te = X_test[:, 0::2]
    im_te = X_test[:, 1::2]
    a_te = re_te + 1j * im_te
    sing_te = (np.isclose(a_te[:, 0], 0) & np.isclose(a_te[:, 1], 0) & np.isclose(a_te[:, 2], 0))
    if np.any(sing_te):
        disc_te = a_te[:, 4] * a_te[:, 4] - 4 * a_te[:, 3] * a_te[:, 5]
        delta_mag = np.abs(disc_te[sing_te]) + 1e-12
        logits_sing, _ = model.forward(X_test[sing_te])
        probs_sing = NumpyMLP.sigmoid(logits_sing)
        # Bin by log10(|Δ|)
        logd = np.log10(delta_mag)
        bins = np.linspace(np.percentile(logd, 5), np.percentile(logd, 95), 10)
        bin_idx = np.digitize(logd, bins)
        xs = []
        ys = []
        es = []
        for b in range(bin_idx.min(), bin_idx.max()+1):
            sel = (bin_idx == b)
            if np.sum(sel) < 3:
                continue
            xs.append(np.mean(logd[sel]))
            ys.append(float(np.mean(probs_sing[sel])))
            es.append(float(np.std(probs_sing[sel], ddof=1) / np.sqrt(np.sum(sel))))
        if len(xs) >= 2:
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            ax6.errorbar(xs, ys, yerr=es, fmt='-o', color='#4C78A8', ecolor='#9ecae1', capsize=4)
            ax6.set_xlabel('log10 |Δ| (singular samples)')
            ax6.set_ylabel('Predicted P(nodal)')
            ax6.set_title('Predicted nodal probability increases with |Δ| when singular')
            ax6.grid(True, linestyle='--', alpha=0.4)
            fig6.suptitle('Figure 6: Evidence of Discriminant Usage by the Network', fontsize=12)
            fig6.tight_layout(rect=[0, 0.03, 1, 0.92])
            f6_path = os.path.join(out_dir, f'prob_vs_discriminant_{stamp}.png')
            fig6.savefig(f6_path, dpi=150)
            plt.close(fig6)
        else:
            f6_path = '(insufficient singular samples to plot)'
    else:
        f6_path = '(no singular samples found in test set)'

    # If discriminant feature is included, run ablation study and Δ sensitivity plot
    if args.include_disc_feature:
        # Ablation: zero out last two features (Δ Re, Δ Im) at test time
        X_test_disc0 = X_test.copy()
        X_test_disc0[:, -2:] = 0.0
        logits_base, _ = model.forward(X_test)
        logits_abla, _ = model.forward(X_test_disc0)
        prob_base = NumpyMLP.sigmoid(logits_base)
        prob_abla = NumpyMLP.sigmoid(logits_abla)
        acc_base = accuracy(y_test, (prob_base >= 0.5).astype(np.int64))
        acc_abla = accuracy(y_test, (prob_abla >= 0.5).astype(np.int64))
        auc_base = roc_auc(y_test, prob_base)
        auc_abla = roc_auc(y_test, prob_abla)

        # Δ-feature sensitivity (final model)
        def delta_sensitivity(model: 'NumpyMLP', Xs: np.ndarray) -> float:
            logits, cache = model.forward(Xs)
            Xc, z1, a1 = cache
            mask = (z1 > 0.0).astype(np.float32)
            w2 = model.W2.squeeze(1)
            dlogit_dz1 = mask * w2[None, :]
            dlogit_dX = dlogit_dz1 @ model.W1.T
            grad_abs = np.mean(np.abs(dlogit_dX), axis=0)
            return float(grad_abs[-2] + grad_abs[-1])

        delta_sens_val = delta_sensitivity(model, X_test)

        # Singular-only ablation metrics (restrict to a0=a1=a2=0 in test set)
        re_te_full = X_test[:, 0::2]
        im_te_full = X_test[:, 1::2]
        a_te_full = re_te_full + 1j * im_te_full
        sing_mask_te = (np.isclose(a_te_full[:, 0], 0) & np.isclose(a_te_full[:, 1], 0) & np.isclose(a_te_full[:, 2], 0))

        if np.any(sing_mask_te):
            y_test_sing = y_test[sing_mask_te]
            prob_base_sing = prob_base[sing_mask_te]
            prob_abla_sing = prob_abla[sing_mask_te]
            acc_base_sing = accuracy(y_test_sing, (prob_base_sing >= 0.5).astype(np.int64))
            acc_abla_sing = accuracy(y_test_sing, (prob_abla_sing >= 0.5).astype(np.int64))
            auc_base_sing = roc_auc(y_test_sing, prob_base_sing)
            auc_abla_sing = roc_auc(y_test_sing, prob_abla_sing)
        else:
            acc_base_sing = acc_abla_sing = float('nan')
            auc_base_sing = auc_abla_sing = float('nan')

        # Bar plot: test metrics with vs without Δ feature
        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(10, 4))
        ax7a.bar(['with Δ', 'Δ masked'], [acc_base, acc_abla], color=['#4C78A8', '#9ecae1'])
        ax7a.set_ylim(0.0, 1.0)
        ax7a.set_ylabel('Accuracy')
        ax7a.set_title('Test Accuracy: Δ present vs masked')
        for i, v in enumerate([acc_base, acc_abla]):
            ax7a.text(i, v + 0.01, f"{v:.3f}", ha='center')

        ax7b.bar(['with Δ', 'Δ masked'], [auc_base, auc_abla], color=['#54A24B', '#a1d99b'])
        ax7b.set_ylim(0.0, 1.0)
        ax7b.set_ylabel('ROC-AUC')
        ax7b.set_title('Test AUC: Δ present vs masked')
        for i, v in enumerate([auc_base, auc_abla]):
            ax7b.text(i, v + 0.01, f"{v:.3f}", ha='center')

        fig7.suptitle(f'Figure 7: Impact of Explicit Δ Feature (Δ sensitivity ~ {delta_sens_val:.3e})', fontsize=12)
        fig7.tight_layout(rect=[0, 0.03, 1, 0.90])
        f7_path = os.path.join(out_dir, f'delta_feature_ablation_{stamp}.png')
        fig7.savefig(f7_path, dpi=150)
        plt.close(fig7)

        # Singular-only ablation figure (if singular samples exist)
        if np.any(sing_mask_te):
            fig7s, (ax7sa, ax7sb) = plt.subplots(1, 2, figsize=(10, 4))
            ax7sa.bar(['with Δ', 'Δ masked'], [acc_base_sing, acc_abla_sing], color=['#4C78A8', '#9ecae1'])
            ax7sa.set_ylim(0.0, 1.0)
            ax7sa.set_ylabel('Accuracy')
            ax7sa.set_title('Singular-only Accuracy: Δ present vs masked')
            for i, v in enumerate([acc_base_sing, acc_abla_sing]):
                ax7sa.text(i, v + 0.01, f"{v:.3f}", ha='center')

            ax7sb.bar(['with Δ', 'Δ masked'], [auc_base_sing, auc_abla_sing], color=['#54A24B', '#a1d99b'])
            ax7sb.set_ylim(0.0, 1.0)
            ax7sb.set_ylabel('ROC-AUC')
            ax7sb.set_title('Singular-only AUC: Δ present vs masked')
            for i, v in enumerate([auc_base_sing, auc_abla_sing]):
                ax7sb.text(i, v + 0.01, f"{v:.3f}", ha='center')

            fig7s.suptitle('Figure 7s: Δ Feature Ablation on Singular-only Test Subset', fontsize=12)
            fig7s.tight_layout(rect=[0, 0.03, 1, 0.90])
            f7s_path = os.path.join(out_dir, f'delta_feature_ablation_singular_only_{stamp}.png')
            fig7s.savefig(f7s_path, dpi=150)
            plt.close(fig7s)
        else:
            f7s_path = '(no singular samples found in test set)'

    # 4) Dataset composition bar chart
    # Reconstruct coefficients from features for the WHOLE dataset X
    re = X[:, 0::2]
    im = X[:, 1::2]
    a_all = re + 1j * im
    sing = (np.isclose(a_all[:, 0], 0) & np.isclose(a_all[:, 1], 0) & np.isclose(a_all[:, 2], 0))
    disc = a_all[:, 4] * a_all[:, 4] - 4 * a_all[:, 3] * a_all[:, 5]
    nodal = sing & (~np.isclose(disc, 0))
    sing_non_nodal = sing & (np.isclose(disc, 0))
    non_sing = ~sing
    counts = [int(np.sum(nodal)), int(np.sum(sing_non_nodal)), int(np.sum(non_sing))]
    labels_comp = ['Singular Nodal', 'Singular Non-Nodal', 'Non-Singular']
    colors = ['#54A24B', '#E45756', '#72B7B2']

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    bars4 = ax4.bar(labels_comp, counts, color=colors)
    ax4.set_ylabel('Count')
    ax4.set_title('Dataset Composition by Algebraic Category at (1:0:0)')
    for b, v in zip(bars4, counts):
        ax4.text(b.get_x() + b.get_width()/2, v + max(counts)*0.01, str(v), ha='center', va='bottom')
    fig4.suptitle('Figure 4: Dataset Composition — Nodal, Singular Non-Nodal, and Non-Singular', fontsize=12)
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    f4_path = os.path.join(out_dir, f'dataset_composition_{stamp}.png')
    fig4.savefig(f4_path, dpi=150)
    plt.close(fig4)

    print('Saved figures:')
    print(f'  1) Loss curve             : {f1_path}')
    print(f'  2) Accuracy curves        : {f2_path}')
    print(f'  3) Monomial accuracy curves (Y^3 & Y^2·Z): {f3_path}')
    print(f'  3b) Expression accuracy curves (XY^2−XZ^2 & +Y^3): {f3b_path}')
    print(f'  4) Dataset composition    : {f4_path}')
    print(f'  5) Feature sensitivity    : {f5_path}')
    print(f'  5b) Singular-only sensitivity (epochs): {f5b_path}')
    print(f'  6) Prob vs |Δ| (singular) : {f6_path}')
    if args.include_disc_feature:
        print(f'  7) Δ feature ablation     : {f7_path}')
        try:
            print(f'  7s) Δ ablation (singular-only) : {f7s_path}')
        except NameError:
            pass


if __name__ == '__main__':
    main()
