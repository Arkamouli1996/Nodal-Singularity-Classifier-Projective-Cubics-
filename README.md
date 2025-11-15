# Nodal Singularity Classifier (Projective Cubics)

This project trains a simple feedforward neural network to predict whether the projective point (1:0:0) is a nodal singularity for a homogeneous cubic polynomial F(X,Y,Z) with complex coefficients.

- Inputs: coefficients of the 10 cubic monomials, each complex c = a+ib represented as (Re, Im) -> 20 real features.
- Label: 1 if (1:0:0) is a nodal singularity; 0 otherwise. Determined algebraically.
- Model: 20 -> 512 (ReLU) -> 1, trained with BCE-with-logits, Adam, L2 via weight_decay.
- Data: 10,000 samples by default with controlled composition:
  - 30% singular at (1:0:0): among these 40% nodal (Δ ≠ 0) and 60% non-nodal (Δ = 0)
  - 70% generic (no enforced constraints)
- Split: 75%/25% train/test.

## Algebraic criteria

Let the monomial order be:
X^3, X^2Y, X^2Z, XY^2, XYZ, XZ^2, Y^3, Y^2Z, YZ^2, Z^3,
with complex coefficients a0..a9. In the affine chart X=1, set f(y,z) = F(1,y,z).

- Singularity at (1:0:0) ⇔ f(0,0)=0 and ∂f/∂y(0,0)=∂f/∂z(0,0)=0 ⇔ a0=a1=a2=0.
- Quadratic part at the point: Q(y,z) = a3 y^2 + a4 yz + a5 z^2.
- Nodal ⇔ discriminant Δ = a4^2 - 4 a3 a5 ≠ 0 (over C, factors as two distinct lines).

## Quick start

1. Install dependencies (pure NumPy implementation – no external ML libs needed):

```powershell
pip install -r requirements.txt
```

2. Run training (350 epochs default):

```powershell
python src/train.py
```

Optional flags:

```powershell
python src/train.py --epochs 350 --n-samples 10000 --batch-size 128 --lr 1e-3 --weight-decay 1e-4
```

The script prints train/test loss, accuracy, and a ROC-AUC estimate.
