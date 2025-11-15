import os
import re
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def find_latest_timestamped_outputs(outputs_dir: str):
    pattern = re.compile(r"_(\d{8}_\d{6})\.png$")
    stamps = {}
    for path in glob.glob(os.path.join(outputs_dir, '*.png')):
        m = pattern.search(os.path.basename(path))
        if not m:
            continue
        stamp = m.group(1)
        stamps.setdefault(stamp, []).append(path)
    if not stamps:
        return None, []
    # Pick the stamp with the most files; tie-breaker: latest lexicographically
    best_stamp = sorted(stamps.items(), key=lambda kv: (len(kv[1]), kv[0]))[-1][0]
    return best_stamp, stamps[best_stamp]


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=16, y=0.98)
    y = 0.92
    for para in paragraphs:
        # naive wrap at ~95 chars
        words = para.split()
        line = ''
        lines = []
        for w in words:
            if len(line) + 1 + len(w) > 95:
                lines.append(line)
                line = w
            else:
                line = (line + ' ' + w).strip()
        if line:
            lines.append(line)
        for ln in lines:
            fig.text(0.06, y, ln, fontsize=11, va='top')
            y -= 0.03
        y -= 0.02
        if y < 0.08:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: str, title: str | None = None):
    fig = plt.figure(figsize=(8.5, 11))
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
        top = 0.94
    else:
        top = 0.99
    ax = fig.add_axes([0.05, 0.04, 0.90, top - 0.06])
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(root, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    stamp, files = find_latest_timestamped_outputs(outputs_dir)
    now_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = os.path.join(outputs_dir, f'summary_report_{now_stamp}.pdf')

    # Canonical figure basenames we try to embed if present
    want = [
        'loss_curve',
        'accuracy_curve',
        'prob_vs_discriminant',
        'delta_feature_ablation',
        'delta_feature_ablation_singular_only',
        'feature_sensitivity',
        'feature_sensitivity_singular_epochs',
        'monomial_accuracy_curves',
        'expression_accuracy_curves',
        'dataset_composition',
    ]

    chosen = {}
    if stamp:
        for base in want:
            candidate = os.path.join(outputs_dir, f'{base}_{stamp}.png')
            if os.path.exists(candidate):
                chosen[base] = candidate

    with PdfPages(out_pdf) as pdf:
        # Page 1: Overview
        add_text_page(pdf, 'Nodal Singularity Classifier — Summary', [
            'Goal: Learn to predict whether the projective point (1:0:0) is a nodal singularity for a random ' \
            'homogeneous cubic in three variables with complex coefficients.',
            'Labeling (exact algebra): Singular at (1:0:0) iff a0=a1=a2=0. Nodal iff Δ=a4^2−4 a3 a5 ≠ 0 in chart X=1.',
            'Model: Pure NumPy MLP (20→512→1), BCE-with-logits, Adam, L2. Optional explicit features Re(Δ), Im(Δ).',
            'Data: 10k samples with fixed composition; inputs are 20 real features (Re/Im of 10 complex coeffs); '
            'becomes 22 when Δ is included.',
            'Outputs: Training curves, composition, sensitivity analyses, P(nodal) vs |Δ|, and Δ ablation figures.',
        ])

        # Page 2: Key takeaways
        add_text_page(pdf, 'Key Points', [
            '1) The network learns a two-stage rule: (a) gate on singularity via a0,a1,a2; (b) if singular, decide nodal via Δ.',
            '2) Providing Δ explicitly improves optimization and raises training/test accuracy; the model relies on Δ when relevant.',
            '3) Global ablation (mask Δ on whole test set) shows a modest gap because many samples are non‑singular and Δ can be '
            'reconstructed from (a3,a4,a5).',
            '4) Singular-only ablation magnifies the gap: when conditioned on a0=a1=a2=0, Δ becomes the pivotal signal.',
        ])

        # Page 3+: Figures if available
        titles = {
            'loss_curve': 'Loss vs Epoch (train/test)',
            'accuracy_curve': 'Accuracy vs Epoch (train/test)',
            'prob_vs_discriminant': 'Predicted P(nodal) vs log10 |Δ| (singular)',
            'delta_feature_ablation': 'Δ Feature Ablation (global test set)',
            'delta_feature_ablation_singular_only': 'Δ Feature Ablation (singular-only)',
            'feature_sensitivity': 'Sensitivity by Monomial',
            'feature_sensitivity_singular_epochs': 'Singular-only Sensitivity vs Epoch (y^2, yz, z^2)',
            'monomial_accuracy_curves': 'Monomial Accuracy Curves (Y^3, Y^2·Z)',
            'expression_accuracy_curves': 'Expression Accuracy (XY^2−XZ^2, +Y^3)',
            'dataset_composition': 'Dataset Composition by Algebraic Category',
        }

        for base in want:
            if base in chosen:
                add_image_page(pdf, chosen[base], titles.get(base))

        # Final page: Run info
        info_lines = [
            f'Outputs source timestamp: {stamp if stamp else "(not found)"}',
            f'Report generated: {now_stamp}',
            f'Embedded figures: {", ".join(chosen.keys()) if chosen else "(none found)"}',
        ]
        add_text_page(pdf, 'Run Information', info_lines)

    print(out_pdf)


if __name__ == '__main__':
    main()
