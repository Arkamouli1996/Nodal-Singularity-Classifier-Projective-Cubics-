# Nodal Singularity Classifier — Projective Cubics (Neural Network)

This README documents the neural network training script `neural network code/train.py` (permalink above).

Contents
- Overview
- Quick start (how to run)
- 1) What is a nodal singularity?
- 2) What the neural network is trying to do
- 3) How the training data is generated (verbatim code snippet)
- 4) Hyperparameters, epoch etc. (verbatim code snippet)
- 5) Final training numbers (from your provided run)
- 6) What is masking and how it is used here
- 7) Future idea: detecting cuspidal singularities
- Outputs saved by the script
- Notes

Overview
========
This script trains a small MLP (NumPy implementation) to detect whether a homogeneous cubic in variables (X:Y:Z) has a nodal singularity located at the projective point (1:0:0). Features are the real and imaginary parts of the 10 complex coefficients of the cubic (20 real features), optionally augmented with the complex discriminant Δ (2 additional real features). The code also runs interpretability experiments (feature sensitivity and synthetic monomial/expression test sets) and saves diagnostic figures.

Quick start
===========
Run the training script :
python "train.py"

1) What is a nodal singularity?
================================
A nodal singularity (node / ordinary double point) is an isolated singular point on a plane curve where two distinct smooth branches cross transversely (locally like xy = 0). For projective cubics, a node at (1:0:0) appears algebraically when the coefficients of X^3, X^2Y, X^2Z vanish and the quadratic part (in Y,Z) has nonzero discriminant Δ = a4^2 − 4 a3 a5.

2) What the neural network is trying to do
=========================================
- Primary task: binary classification — predict if the cubic has a nodal singularity at (1:0:0) (label = 1) or not (label = 0).
- Input: coefficients of the homogeneous cubic (complex coefficients split into Re/Im per monomial).
- Model: small two-layer MLP (ReLU hidden layer, single-logit output) implemented in pure NumPy with Adam and BCE-with-logits loss.
- Additional analyses: synthetic monomial/expression test sets, feature sensitivity via input-gradient proxies, discriminant-ablation experiments.

3) How is the training data generated? (verbatim)
=================================================
The following code is verbatim from `train.py` and shows the sampling functions and dataset builder used to generate labels and features.

```python
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
```

4) What are the hyperparameters, epoch etc.? (verbatim)
=======================================================
The CLI hyperparameters and their default values are defined verbatim in the script as follows:

```python
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
```

Other model defaults (from the MLP constructor) are:
- in_dim: inferred from input (20 or 22 if Δ appended)
- hidden: default 512
- weight_decay: default 1e-4
- lr (learning rate): default 1e-3
- Adam betas: beta1=0.9, beta2=0.999, eps=1e-8

Epoch: the default number of epochs is 350 (see --epochs default above). The training loop prints logs on epoch 1, every 10 epochs, and on the final epoch.

5) What is the final training accuracy etc.?
============================================
From the run output provided (full log printed by the script), the final reported epoch (350) metrics are:

Epoch 350 | train loss 0.0314 | test loss 0.1199 | acc 0.9520 | auc 0.9872

Summary of notable metrics from the same run:
- Initial: Epoch 001 | train loss 0.3674 | test loss 0.3713 | acc 0.8804 | auc 0.7711
- Early high AUC: Epoch 100 | acc 0.9560 | auc 0.9882
- Best observed accuracy in the printed log: 0.9584 at epoch 290
- Final model (epoch 350): test accuracy 95.20%, ROC-AUC ≈ 0.9872

Interpretation: the trained model achieves strong discriminative performance on the held-out test split used in this run.

6) What is masking and how it is used here?
===========================================
- Masking here refers to zeroing-out / removing selected input features at test time (or during ablation experiments) and measuring the resulting change in model performance.
- The script performs several masking/ablation experiments:
  - If `--include-disc-feature` is used, the discriminant Δ is appended as explicit features (2 real numbers). The script performs a Δ ablation: it zeroes the Δ features in the test set and compares accuracy/AUC with and without Δ present (plots saved).
  - The script also computes feature sensitivity using an input-gradient proxy (averaged absolute gradient of the logit w.r.t. each input feature) and aggregates Re/Im pairs into per-monomial scores. This is used to highlight which monomials (e.g., a3, a4, a5 corresponding to Y^2, YZ, Z^2) the model relies on most.
  - Additionally, a "singular-only" sensitivity curve tracks sensitivity for quadratic monomials across epochs restricted to samples that are singular at (1:0:0).
- Purpose: quantify importance of monomials (and discriminant) for nodal detection; validate model uses algebraically relevant signals.

7) Future idea — Detecting cuspidal singularities (concise plan)
================================================================
- Cusps (analytic type y^2 = x^3) differ from nodes because there is a single branch and higher contact order.
- Dataset: generate labeled cuspidal examples (enforce a0=a1=a2=0 and impose local higher-order vanishing conditions that yield cusp vs node). Because cusps are more constrained, prefer symbolic algebra checks or higher-precision arithmetic when labelling.
- Labels: either use multi-class labels (non-singular / nodal / cuspidal) or two-stage pipeline: (1) singular vs not; (2) classify singular type.
- Features: keep existing monomial coefficients, add Hessian-derived features, higher-jet coefficients at the singular point, or symbolic invariants that distinguish A1 vs A2.
- Model & training: multi-class classifier, class-weighting / oversampling for cusps, and the same interpretability and masking tools to analyze which features identify cusps.
- Caveats: cusps are rarer and may require more careful, stable algebraic testing to generate correct labels.


Notes
=====
- Exact code lives in `neural network code/train.py` (permalink in the header). The snippets above were copied verbatim from that file as requested.
- To reproduce the run whose logs you supplied, run with the defaults (n_samples=10000, epochs=350, batch_size=128, lr=1e-3, weight_decay=1e-4, hidden=512, seed=42).
- If you want the README to also include more verbatim code (e.g., the MLP class, training loop, plotting code) I can append those exact blocks on request.

If you want, I can:
- add a short "How to reproduce" command-line example with exact commands, or
- include additional verbatim snippets (e.g., the entire NumpyMLP class or the training loop) into this README.
