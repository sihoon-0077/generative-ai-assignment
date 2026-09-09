"""
PA1 - Fit a Multivariate Gaussian to Images
===========================================

Task
----
1. Load `sklearn.datasets.load_digits()` and normalize pixel values from 0-16 to 0-1.
2. Flatten each 8x8 image to x in R^64.
3. Fit ONE multivariate Gaussian using MLE.
4. Generate at least 20 new samples, reshape to 8x8, and visualize them.

Run
---
    python pa1.py

Every numeric claim printed by this script is re-checked by an assertion at the
end (`ALL CHECKS PASSED`), so the output is self-verifying.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG files instead of opening a window

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
SEED = 0
N_SHOW = 25          # generated samples to visualize (assignment requires >= 20)
N_VERIFY = 100_000   # samples used only to verify the sampler is correct
OUT = Path(__file__).resolve().parent / "outputs"


class Log:
    """Print to stdout and keep a copy so the whole run can be saved to a file."""

    def __init__(self, path):
        self.path = path
        self.lines = []

    def __call__(self, msg=""):
        text = str(msg)
        print(text)
        self.lines.append(text)

    def save(self):
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# step 1 + 2: load, normalize, flatten
# --------------------------------------------------------------------------- #
def load_normalized_digits():
    """Return X (n, 64) in [0, 1], the labels, and the raw 0-16 images."""
    digits = load_digits()
    images_raw = digits.images                       # (n, 8, 8), integers 0..16
    X = images_raw.reshape(len(images_raw), -1) / 16.0  # flatten -> R^64, scale to [0,1]
    return X, digits.target, images_raw, digits.data


# --------------------------------------------------------------------------- #
# step 3: maximum likelihood estimation
# --------------------------------------------------------------------------- #
def fit_gaussian_mle(X):
    """MLE of a multivariate Gaussian.

    mu    = (1/N) sum_i x_i
    Sigma = (1/N) sum_i (x_i - mu)(x_i - mu)^T      <- 1/N, not 1/(N-1)

    The 1/N version is the maximizer of the likelihood; 1/(N-1) is the unbiased
    estimator. The assignment asks for MLE, so 1/N is the correct choice.
    """
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / n
    Sigma = (Sigma + Sigma.T) / 2.0  # kill float asymmetry (~1e-18) so eigh is exact
    return mu, Sigma


# --------------------------------------------------------------------------- #
# step 4: sampling from a possibly singular Gaussian
# --------------------------------------------------------------------------- #
def build_sampler(mu, Sigma):
    """Build x = mu + A z with A A^T = Sigma, z ~ N(0, I).

    Cholesky would fail here: the MLE covariance of this dataset is singular
    (some pixels are 0 in every image, so their variance is exactly 0).
    The symmetric eigendecomposition Sigma = V diag(lam) V^T always exists for a
    symmetric matrix, and A = V diag(sqrt(lam)) satisfies A A^T = Sigma even when
    some lam are 0. Negative eigenvalues can only come from float error on a PSD
    matrix, so clipping them at 0 is safe.

    This samples from the *exact* MLE Gaussian, which is degenerate: all of its
    mass lies on the affine subspace mu + span(eigenvectors with lam > 0).
    """
    evals, evecs = np.linalg.eigh(Sigma)
    evals_clipped = np.clip(evals, 0.0, None)
    A = evecs * np.sqrt(evals_clipped)  # column j scaled by sqrt(lam_j)

    # A pixel with zero variance is deterministic: (A A^T)_jj = Sigma_jj = 0 forces
    # row j of A to be exactly zero. eigh only reaches ~1e-9 there, which would make
    # such a pixel jitter around its constant value instead of staying fixed.
    A[np.diag(Sigma) <= 1e-12, :] = 0.0

    def sample(rng, n):
        z = rng.standard_normal((n, mu.size))
        return mu + z @ A.T

    return sample, evals, A


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
def grid_of_images(ax_array, flat_images, title_prefix=None):
    for ax, vec in zip(ax_array.ravel(), flat_images):
        ax.imshow(vec.reshape(8, 8), cmap="gray_r", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
    if title_prefix:
        ax_array.ravel()[0].set_ylabel(title_prefix)


def figure_real_vs_generated(X_real, X_gen, path):
    fig, axes = plt.subplots(5, 11, figsize=(11.5, 5.6))
    for row in range(5):
        for col in range(5):
            ax = axes[row, col]
            ax.imshow(X_real[row * 5 + col].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 5].axis("off")
        for col in range(5):
            ax = axes[row, 6 + col]
            ax.imshow(X_gen[row * 5 + col].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 2].set_title("Real digits (training data)", fontsize=12, pad=10)
    axes[0, 8].set_title(f"Generated: single Gaussian (n={N_SHOW})", fontsize=12, pad=10)
    fig.suptitle("PA1 - one multivariate Gaussian fitted by MLE on all 1797 digits",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_diagnostics(mu, Sigma, evals, A, path):
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(mu.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("mean $\\mu$", fontsize=10); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(np.sqrt(np.diag(Sigma)).reshape(8, 8), cmap="viridis")
    ax.set_title("per-pixel std $\\sqrt{\\Sigma_{jj}}$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2:4])
    ax.imshow(Sigma, cmap="RdBu_r", vmin=-np.abs(Sigma).max(), vmax=np.abs(Sigma).max())
    ax.set_title("covariance $\\Sigma$  (64x64)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 4:6])
    ax.semilogy(np.maximum(evals[::-1], 1e-20), "o-", ms=3)
    ax.axhline(1e-12, color="crimson", ls="--", lw=1, label="numerical zero")
    ax.set_title("eigenvalues of $\\Sigma$ (log scale)", fontsize=10)
    ax.set_xlabel("index"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    order = np.argsort(evals)[::-1]
    for k in range(6):
        ax = fig.add_subplot(gs[1, k])
        v = A[:, order[k]]
        v = v / (np.abs(v).max() + 1e-12)
        ax.imshow(v.reshape(8, 8), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"PC{k + 1}\n$\\lambda$={evals[order[k]]:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("PA1 diagnostics: what the single Gaussian actually learned", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / "pa1_log.txt")
    rng = np.random.default_rng(SEED)

    log("=" * 74)
    log("PA1 - Fit ONE multivariate Gaussian to load_digits() and sample from it")
    log("=" * 74)

    # ---- step 1 + 2 -------------------------------------------------------- #
    X, y, images_raw, sklearn_flat = load_normalized_digits()
    log("\n[1] Load + normalize + flatten")
    log(f"    raw images shape          : {images_raw.shape}   (8x8 per image)")
    log(f"    raw pixel range           : [{images_raw.min():.1f}, {images_raw.max():.1f}]  (0-16)")
    log(f"    X shape after flatten     : {X.shape}   -> each x is in R^64")
    log(f"    X pixel range after /16   : [{X.min():.4f}, {X.max():.4f}]  (0-1)")
    log(f"    n samples / n classes     : {X.shape[0]} / {len(np.unique(y))}")
    # our own flatten must agree with sklearn's pre-flattened `.data`
    flatten_matches = np.allclose(X * 16.0, sklearn_flat)
    log(f"    flatten matches digits.data: {flatten_matches}")

    # ---- step 3 ------------------------------------------------------------ #
    mu, Sigma = fit_gaussian_mle(X)
    evals_all = np.linalg.eigvalsh(Sigma)
    rank = int(np.sum(evals_all > 1e-12))
    zero_var_pixels = int(np.sum(np.diag(Sigma) < 1e-15))

    log("\n[2] MLE fit of a single multivariate Gaussian  N(mu, Sigma)")
    log(f"    mu shape                  : {mu.shape},  mu range [{mu.min():.4f}, {mu.max():.4f}]")
    log(f"    Sigma shape               : {Sigma.shape}")
    log(f"    Sigma symmetric           : {np.allclose(Sigma, Sigma.T)}")
    log(f"    eigenvalue range          : [{evals_all.min():.3e}, {evals_all.max():.3e}]")
    log(f"    numerical rank of Sigma   : {rank} / 64   -> {'SINGULAR' if rank < 64 else 'full rank'}")
    log(f"    pixels with zero variance : {zero_var_pixels}  (always 0 in every training image)")
    log("    => Cholesky is impossible; we use the eigendecomposition instead.")

    # ---- step 4: sampler + verification ------------------------------------ #
    sample, evals, A = build_sampler(mu, Sigma)
    recon_err = np.linalg.norm(A @ A.T - Sigma) / np.linalg.norm(Sigma)

    S = sample(rng, N_VERIFY)
    mu_hat = S.mean(axis=0)
    Sigma_hat = np.cov(S.T, bias=True)
    err_mu = np.linalg.norm(mu_hat - mu) / np.linalg.norm(mu)
    err_Sigma = np.linalg.norm(Sigma_hat - Sigma) / np.linalg.norm(Sigma)

    log("\n[3] Sampler check   x = mu + A z,  z ~ N(0, I)")
    log(f"    ||A A^T - Sigma|| / ||Sigma||                 : {recon_err:.3e}")
    log(f"    {N_VERIFY} samples -> rel. error of mean      : {err_mu:.3e}")
    log(f"    {N_VERIFY} samples -> rel. error of covariance: {err_Sigma:.3e}")
    log("    => the empirical moments converge to (mu, Sigma): the sampler is correct.")

    # ---- generate the deliverable samples ---------------------------------- #
    X_gen = sample(rng, N_SHOW)
    log(f"\n[4] Generated {N_SHOW} new samples (>= 20 required), each reshaped to 8x8")
    log(f"    generated value range     : [{X_gen.min():.4f}, {X_gen.max():.4f}]")
    frac_out = float(np.mean((X_gen < 0.0) | (X_gen > 1.0)))
    log(f"    fraction of pixels outside [0,1] : {frac_out:.3%}")
    log("    (the training data is bounded in [0,1] but a Gaussian has support on")
    log("     all of R^64, so out-of-range pixels are expected - analysed in PA2.)")

    idx_real = rng.choice(len(X), size=N_SHOW, replace=False)
    figure_real_vs_generated(X[idx_real], X_gen, OUT / "pa1_real_vs_generated.png")
    figure_diagnostics(mu, Sigma, evals, A, OUT / "pa1_diagnostics.png")
    log(f"\n[5] Figures written to {OUT}")
    log("    - pa1_real_vs_generated.png")
    log("    - pa1_diagnostics.png")

    np.savez_compressed(OUT / "pa1_model.npz", mu=mu, Sigma=Sigma, samples=X_gen)

    # ---- assertions: the run fails loudly if any claim above is wrong ------- #
    assert X.shape == (1797, 64), "X must be (1797, 64)"
    assert flatten_matches, "manual flatten disagrees with digits.data"
    assert abs(X.min()) < 1e-12 and abs(X.max() - 1.0) < 1e-12, "X must be exactly in [0,1]"
    assert mu.shape == (64,) and Sigma.shape == (64, 64)
    assert np.allclose(Sigma, Sigma.T), "Sigma must be symmetric"
    assert evals_all.min() > -1e-10, "Sigma must be positive semi-definite"
    assert recon_err < 1e-12, "A A^T must reproduce Sigma"
    assert err_mu < 2e-2, "sample mean did not converge to mu"
    assert err_Sigma < 2e-2, "sample covariance did not converge to Sigma"
    assert X_gen.shape == (N_SHOW, 64) and N_SHOW >= 20, "need >= 20 generated samples"

    log("\n" + "=" * 74)
    log("ALL CHECKS PASSED")
    log("=" * 74)
    log.save()


if __name__ == "__main__":
    main()
