"""
PA2 - Class-conditional Gaussians + pixel-range analysis
========================================================

Task
----
(A) Extend PA1 by fitting 10 Gaussians p(x | y = c), one for each digit class.
    Generate samples conditioned on c. Compare the quality with the
    single-Gaussian model and explain why conditioning can help.

(B) Compare the pixel-value ranges of the training images and the generated
    samples. Report the minimum and maximum values for each, describe the
    differences, and explain the findings based on the fitted Gaussian model.

Run
---
    python pa2.py

The comparison in (A) is not done by eye: three independent quantitative
metrics are computed (held-out log-likelihood, an oracle-classifier score, and
nearest-neighbour distance to real data). The explanation in (B) is verified by
predicting the out-of-range rate analytically and checking it against the
empirical rate from the generated samples.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorized
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------- #
SEED = 0
N_PER_CLASS_SHOW = 8      # conditional samples visualised per digit
N_EVAL = 2_000            # samples used for the quality metrics
N_RANGE = 20_000          # samples used for the pixel-range statistics
OUT = Path(__file__).resolve().parent / "outputs"


class Log:
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
# model: same MLE + degenerate sampler as PA1 (kept here so PA2 runs standalone)
# --------------------------------------------------------------------------- #
def load_normalized_digits():
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1) / 16.0
    return X, digits.target


def fit_gaussian_mle(X):
    """MLE: mu = mean, Sigma = (1/N) sum (x-mu)(x-mu)^T  (1/N, not 1/(N-1))."""
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / n
    return mu, (Sigma + Sigma.T) / 2.0


def build_sampler(mu, Sigma):
    """x = mu + A z with A A^T = Sigma. Works for singular Sigma (no Cholesky)."""
    evals, evecs = np.linalg.eigh(Sigma)
    A = evecs * np.sqrt(np.clip(evals, 0.0, None))
    # A pixel with zero variance is deterministic: (A A^T)_jj = Sigma_jj = 0 forces
    # row j of A to be exactly zero. eigh only reaches ~1e-9 there, which would make
    # such a pixel jitter around its constant value - and since that value is 0, it
    # would land just below 0 half the time and pollute the Part B statistics.
    A[np.diag(Sigma) <= 1e-12, :] = 0.0

    def sample(rng, n):
        return mu + rng.standard_normal((n, mu.size)) @ A.T

    return sample


def gaussian_logpdf(X, mu, Sigma_reg):
    """log N(x; mu, Sigma_reg). Sigma_reg must be positive definite."""
    d = mu.size
    L = np.linalg.cholesky(Sigma_reg)
    sol = np.linalg.solve(L, (X - mu).T)          # (d, n)
    maha = np.einsum("ij,ij->j", sol, sol)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)


def logsumexp(M, axis):
    m = M.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_conditional_grid(cond_samples, path):
    fig, axes = plt.subplots(10, N_PER_CLASS_SHOW, figsize=(N_PER_CLASS_SHOW * 0.95, 10.2))
    for c in range(10):
        for k in range(N_PER_CLASS_SHOW):
            ax = axes[c, k]
            ax.imshow(cond_samples[c][k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(f"c = {c}", fontsize=11, rotation=0,
                              labelpad=22, va="center")
    fig.suptitle("PA2 (A) — samples from $p(x \\mid y=c)$, one row per class",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_model_comparison(X_real, gen_single, gen_cond, cond_labels, path):
    # three 4x4 blocks side by side: real | single Gaussian | class-conditional
    fig, axes = plt.subplots(4, 17, figsize=(15.5, 4.4))

    def draw(offset, data, labels, title):
        for row in range(4):
            for col in range(4):
                ax = axes[row, offset + col]
                ax.imshow(data[row * 4 + col].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if labels is not None:
                    ax.set_title(str(labels[row * 4 + col]), fontsize=8, pad=2)
        axes[0, offset + 1].annotate(title, xy=(0.5, 1.35), xycoords="axes fraction",
                                     ha="center", fontsize=12)

    draw(0, X_real, None, "Real digits")
    draw(6, gen_single, None, "Single Gaussian (PA1)")
    draw(13, gen_cond, cond_labels, "Class-conditional (PA2)")
    for row in range(4):
        for col in (4, 5, 10, 11, 12):
            axes[row, col].axis("off")
    fig.suptitle("PA2 (A) — conditioning on the class removes the between-class blur",
                 fontsize=13, y=1.12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_means(mu_global, class_mu, path):
    fig, axes = plt.subplots(1, 11, figsize=(13, 1.9))
    axes[0].imshow(mu_global.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("global $\\mu$\n(PA1)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for c in range(10):
        ax = axes[c + 1]
        ax.imshow(class_mu[c].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"$\\mu_{{{c}}}$", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The single Gaussian can only place its mean at the average of all digits",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_pixel_ranges(X_train, gen_single, gen_cond, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))

    ax = axes[0]
    bins = np.linspace(-1.0, 2.0, 181)
    ax.hist(X_train.ravel(), bins=bins, density=True, alpha=0.55,
            label=f"training  [{X_train.min():.2f}, {X_train.max():.2f}]", color="#444444")
    ax.hist(gen_single.ravel(), bins=bins, density=True, histtype="step", lw=1.8,
            label=f"single Gaussian  [{gen_single.min():.2f}, {gen_single.max():.2f}]",
            color="#d1495b")
    ax.hist(gen_cond.ravel(), bins=bins, density=True, histtype="step", lw=1.8,
            label=f"class-conditional  [{gen_cond.min():.2f}, {gen_cond.max():.2f}]",
            color="#0f7173")
    ax.axvspan(-1.0, 0.0, color="orange", alpha=0.10)
    ax.axvspan(1.0, 2.0, color="orange", alpha=0.10)
    ax.axvline(0.0, color="k", lw=1, ls="--")
    ax.axvline(1.0, color="k", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("pixel value")
    ax.set_ylabel("density (log)")
    ax.set_title("Pixel-value distribution\n(shaded = impossible for real images)", fontsize=11)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ex = gen_single[0].reshape(8, 8)
    im = ax.imshow(ex, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(8):
        for j in range(8):
            v = ex[i, j]
            if v < 0 or v > 1:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="lime", lw=2))
    ax.set_title("One generated sample\n(green = pixel outside [0,1])", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("PA2 (B) — a Gaussian has support on all of $\\mathbb{R}^{64}$, "
                 "the data does not", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / "pa2_log.txt")
    rng = np.random.default_rng(SEED)

    X, y = load_normalized_digits()
    log("=" * 78)
    log("PA2 - class-conditional Gaussians  +  pixel-range analysis")
    log("=" * 78)
    log(f"\ndata: X {X.shape}, values in [{X.min():.1f}, {X.max():.1f}], "
        f"{len(np.unique(y))} classes")

    # ===================================================================== #
    # PART A.1 - fit the 10 class-conditional Gaussians on the full dataset
    # ===================================================================== #
    log("\n" + "-" * 78)
    log("PART A.1  Fit p(x | y = c) for c = 0..9   (MLE per class)")
    log("-" * 78)

    mu_global, Sigma_global = fit_gaussian_mle(X)
    classes = np.unique(y)
    priors, class_mu, class_Sigma, class_sampler = [], [], [], []

    log(f"{'c':>2} {'N_c':>5} {'prior':>7} {'rank(Sig_c)':>12} {'tr(Sig_c)':>11}")
    for c in classes:
        Xc = X[y == c]
        m, S = fit_gaussian_mle(Xc)
        priors.append(len(Xc) / len(X))
        class_mu.append(m)
        class_Sigma.append(S)
        class_sampler.append(build_sampler(m, S))
        rank_c = int(np.sum(np.linalg.eigvalsh(S) > 1e-12))
        log(f"{c:>2} {len(Xc):>5} {priors[-1]:>7.4f} {rank_c:>9}/64 {np.trace(S):>11.4f}")

    priors = np.array(priors)
    class_mu = np.array(class_mu)
    class_Sigma = np.array(class_Sigma)

    # --- law of total covariance: the reason conditioning helps, as an identity
    within = np.einsum("c,cij->ij", priors, class_Sigma)
    diffs = class_mu - mu_global
    between = np.einsum("c,ci,cj->ij", priors, diffs, diffs)
    ltc_err = np.linalg.norm(within + between - Sigma_global) / np.linalg.norm(Sigma_global)
    tr_w, tr_b, tr_t = np.trace(within), np.trace(between), np.trace(Sigma_global)

    log("\nLaw of total covariance:  Sigma_total = E_c[Sigma_c] + Cov_c[mu_c]")
    log(f"    identity check ||lhs-rhs||/||rhs||     : {ltc_err:.3e}   (exact)")
    log(f"    trace(Sigma_total)                     : {tr_t:.4f}")
    log(f"    trace(within-class  E_c[Sigma_c])      : {tr_w:.4f}  ({tr_w / tr_t:.1%})")
    log(f"    trace(between-class Cov_c[mu_c])       : {tr_b:.4f}  ({tr_b / tr_t:.1%})")
    log(f"    => {tr_b / tr_t:.1%} of the single Gaussian's variance is variation")
    log("       BETWEEN digit classes. Conditioning removes exactly that part.")

    # --- conditional sampling: p(x | y = c)
    cond_samples = [class_sampler[c](rng, N_PER_CLASS_SHOW) for c in classes]
    figure_conditional_grid(cond_samples, OUT / "pa2_conditional_samples.png")
    figure_means(mu_global, class_mu, OUT / "pa2_class_means.png")

    # ===================================================================== #
    # PART A.2 - quantitative comparison: single vs class-conditional
    # ===================================================================== #
    log("\n" + "-" * 78)
    log("PART A.2  Quality comparison  (three independent metrics)")
    log("-" * 78)

    sampler_single = build_sampler(mu_global, Sigma_global)

    def sample_conditional(rng, n):
        """Draw c ~ p(y), then x ~ p(x | y=c). Returns (X, labels)."""
        labels = rng.choice(classes, size=n, p=priors)
        out = np.empty((n, X.shape[1]))
        for c in classes:
            idx = np.where(labels == c)[0]
            if idx.size:
                out[idx] = class_sampler[c](rng, idx.size)
        return out, labels

    gen_single = sampler_single(rng, N_EVAL)
    gen_cond, gen_cond_y = sample_conditional(rng, N_EVAL)

    # ---- metric 1: held-out log-likelihood ---------------------------------
    # The MLE covariances are singular, so the density on R^64 does not exist.
    # For likelihood *evaluation only* we use Sigma + lam*I and pick lam on a
    # validation split (never on the test split).
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED,
                                            stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED,
                                          stratify=ytmp)
    log(f"\n[metric 1] held-out log-likelihood   (train {len(Xtr)} / val {len(Xva)} "
        f"/ test {len(Xte)})")

    mu_s, Sig_s = fit_gaussian_mle(Xtr)
    tr_priors, tr_mu, tr_Sig = [], [], []
    for c in classes:
        Xc = Xtr[ytr == c]
        m, S = fit_gaussian_mle(Xc)
        tr_priors.append(len(Xc) / len(Xtr)); tr_mu.append(m); tr_Sig.append(S)
    tr_priors = np.array(tr_priors)

    I = np.eye(X.shape[1])
    lam_grid = np.logspace(-6, 0, 25)

    def ll_single(Xq, lam):
        return gaussian_logpdf(Xq, mu_s, Sig_s + lam * I).mean()

    def ll_cond(Xq, lam):
        comp = np.stack([np.log(tr_priors[c]) + gaussian_logpdf(Xq, tr_mu[c],
                                                                tr_Sig[c] + lam * I)
                         for c in classes], axis=1)
        return logsumexp(comp, axis=1).mean()

    lam_s = lam_grid[int(np.argmax([ll_single(Xva, l) for l in lam_grid]))]
    lam_c = lam_grid[int(np.argmax([ll_cond(Xva, l) for l in lam_grid]))]
    te_s, te_c = ll_single(Xte, lam_s), ll_cond(Xte, lam_c)

    log(f"    single Gaussian     : lambda* = {lam_s:.2e}   test LL/sample = {te_s:9.3f}")
    log(f"    class-conditional   : lambda* = {lam_c:.2e}   test LL/sample = {te_c:9.3f}")
    log(f"    improvement                                   = {te_c - te_s:9.3f} nats/sample")
    log("    (higher is better; both models regularised the same way and tuned on val)")

    # ---- metric 2: oracle classifier ---------------------------------------
    clf = LogisticRegression(max_iter=5000)
    clf.fit(Xtr, ytr)
    oracle_acc = clf.score(Xte, yte)

    p_single = clf.predict_proba(gen_single)
    p_cond = clf.predict_proba(gen_cond)
    ent = lambda P: float(np.mean(-(P * np.log(P + 1e-12)).sum(axis=1)))
    cond_acc = float(np.mean(clf.predict(gen_cond) == gen_cond_y))

    log(f"\n[metric 2] oracle classifier (logistic regression, real test acc "
        f"= {oracle_acc:.1%})")
    log(f"    conditional samples classified as the intended c : {cond_acc:.1%}")
    log(f"    mean max class probability  - single Gaussian    : {p_single.max(1).mean():.3f}")
    log(f"    mean max class probability  - class-conditional  : {p_cond.max(1).mean():.3f}")
    log(f"    mean predictive entropy     - single Gaussian    : {ent(p_single):.3f} nats")
    log(f"    mean predictive entropy     - class-conditional  : {ent(p_cond):.3f} nats")
    log("    (low entropy / high confidence = the sample looks like ONE definite digit)")

    # ---- metric 3: nearest-neighbour distance to real data -----------------
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
    d_single = nn.kneighbors(gen_single)[0].ravel().mean()
    d_cond = nn.kneighbors(gen_cond)[0].ravel().mean()
    d_real = nn.kneighbors(Xte)[0].ravel().mean()

    log("\n[metric 3] mean L2 distance to the nearest REAL training image")
    log(f"    real held-out images (reference floor) : {d_real:.3f}")
    log(f"    single Gaussian samples               : {d_single:.3f}")
    log(f"    class-conditional samples             : {d_cond:.3f}")
    log("    (lower = lands closer to the real data manifold)")

    idx_real = rng.choice(len(X), size=16, replace=False)
    figure_model_comparison(X[idx_real], gen_single, gen_cond, gen_cond_y,
                            OUT / "pa2_single_vs_conditional.png")

    # ===================================================================== #
    # PART B - pixel-value ranges
    # ===================================================================== #
    log("\n" + "-" * 78)
    log("PART B  Pixel-value ranges: training images vs generated samples")
    log("-" * 78)

    big_single = sampler_single(rng, N_RANGE)
    big_cond, _ = sample_conditional(rng, N_RANGE)

    def out_frac(A):
        return float(np.mean((A < 0.0) | (A > 1.0)))

    rows = [
        ("training images", X, None),
        (f"generated - single Gaussian", big_single, None),
        (f"generated - class-conditional", big_cond, None),
    ]
    log(f"\n{'source':<32} {'min':>10} {'max':>10} {'% outside [0,1]':>17}")
    for name, A, _ in rows:
        log(f"{name:<32} {A.min():>10.4f} {A.max():>10.4f} {out_frac(A):>16.2%}")

    log(f"\n    below 0  - single      : {np.mean(big_single < 0):.2%}"
        f"   |  max negative magnitude {abs(big_single.min()):.3f}")
    log(f"    above 1  - single      : {np.mean(big_single > 1):.2%}")
    log(f"    below 0  - conditional : {np.mean(big_cond < 0):.2%}"
        f"   |  max negative magnitude {abs(big_cond.min()):.3f}")
    log(f"    above 1  - conditional : {np.mean(big_cond > 1):.2%}")

    # --- analytic prediction: is the Gaussian model really the explanation? --
    # Marginal of pixel j is N(mu_j, Sigma_jj), so
    #   P(x_j < 0) + P(x_j > 1) = Phi(-mu_j/s_j) + Phi((mu_j-1)/s_j)
    def predicted_out_frac(mu, Sigma):
        s = np.sqrt(np.diag(Sigma))
        ok = s > 1e-12                       # zero-variance pixels are deterministic
        p = np.zeros_like(s)
        p[ok] = ndtr(-mu[ok] / s[ok]) + ndtr((mu[ok] - 1.0) / s[ok])
        return float(p.mean())

    pred_single = predicted_out_frac(mu_global, Sigma_global)
    pred_cond = float(np.sum(priors * np.array(
        [predicted_out_frac(class_mu[c], class_Sigma[c]) for c in classes])))
    emp_single, emp_cond = out_frac(big_single), out_frac(big_cond)

    n_zero_var = int(np.sum(np.diag(Sigma_global) < 1e-15))
    log("\n    Verification of the explanation (theory vs simulation):")
    log(f"      single Gaussian    predicted {pred_single:.4%}  vs  observed {emp_single:.4%}")
    log(f"      class-conditional  predicted {pred_cond:.4%}  vs  observed {emp_cond:.4%}")
    log(f"      pixels with zero variance (always generated exactly at their")
    log(f"      constant training value, so never out of range) : {n_zero_var}")

    figure_pixel_ranges(X, big_single, big_cond, OUT / "pa2_pixel_ranges.png")

    log(f"\nFigures written to {OUT}")
    for f in ("pa2_conditional_samples.png", "pa2_single_vs_conditional.png",
              "pa2_class_means.png", "pa2_pixel_ranges.png"):
        log(f"    - {f}")

    # ===================================================================== #
    # assertions
    # ===================================================================== #
    assert ltc_err < 1e-12, "law of total covariance identity must hold exactly"
    assert abs(priors.sum() - 1.0) < 1e-12, "priors must sum to 1"
    assert te_c > te_s, "conditional model must beat the single Gaussian on held-out LL"
    assert oracle_acc > 0.90, "oracle classifier is not trustworthy"
    assert cond_acc > 0.30, "conditional samples are not recognisable as their class"
    assert ent(p_cond) < ent(p_single), "conditional samples must be less ambiguous"
    assert d_cond < d_single, "conditional samples must be closer to real data"
    assert abs(X.min()) < 1e-12 and abs(X.max() - 1.0) < 1e-12, "training data must be [0,1]"
    assert big_single.min() < 0.0 < 1.0 < big_single.max(), "Gaussian must leave [0,1]"
    # tolerance 0.005 is ~3x the Monte-Carlo noise at N_RANGE=20000; loose enough to
    # never flake, tight enough that the earlier 1.6%/8.2% bug would still be caught.
    assert abs(pred_single - emp_single) < 0.005, "analytic out-of-range rate mismatch"
    assert abs(pred_cond - emp_cond) < 0.005, "analytic out-of-range rate mismatch"

    log("\n" + "=" * 78)
    log("ALL CHECKS PASSED")
    log("=" * 78)
    log.save()


if __name__ == "__main__":
    main()
