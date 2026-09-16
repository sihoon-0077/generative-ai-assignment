"""
PA2 - Class-conditional Gaussians + pixel-range analysis
========================================================

[과제 원문]
(A) Extend PA1 by fitting 10 Gaussians p(x | y = c), one for each digit class.
    Generate samples conditioned on c. Compare the quality with the
    single-Gaussian model and explain why conditioning can help.

(B) Compare the pixel-value ranges of the training images and the generated
    samples. Report the minimum and maximum values for each, describe the
    differences, and explain the findings based on the fitted Gaussian model.

[이 스크립트가 하는 일]
PA1의 단일 Gaussian은 숫자처럼 보이는 흐릿한 얼룩만 만들었다. 이유는
"봉우리가 하나뿐인 분포로 봉우리 10개짜리 데이터를 모델링했기 때문"이다.
여기서는 클래스마다 Gaussian을 따로 학습해서 그 한계를 정면으로 해결하고,
얼마나 좋아졌는지를 눈이 아니라 독립적인 지표 3개로 측정한다.

[핵심 개념 4가지]
  - class-conditional = GMM : p(x) = SUM_c pi_c N(x; mu_c, Sigma_c). 봉우리 10개
  - law of total covariance : "왜 conditioning이 돕는가"의 정량적 답.
                              전체 분산의 42.1%가 between-class였다
  - 수치 안정성             : Cholesky로 logdet, log-sum-exp trick
  - marginal도 Gaussian     : Part B에서 범위 이탈률을 이론적으로 예측하는 근거

[검증 설계]
(A)의 비교는 눈으로 하지 않는다. 원리가 서로 다른 지표 3개를 쓴다:
    1. held-out log-likelihood  (확률모델로서 얼마나 좋은가)
    2. oracle classifier        (사람이 보는 "숫자다움"의 대리 지표)
    3. 최근접 실제이미지 거리    (실제 데이터 영역에 얼마나 가까운가)
(B)의 설명은 "그럴듯한 말"로 끝내지 않는다. 범위 이탈률을 이론으로 예측하고
실측과 대조한다. 이 대조 때문에 실제로 버그를 하나 잡았다 (build_sampler 참고).

[실행]
    python pa2.py
"""

from pathlib import Path

import matplotlib

# pyplot import 전에 호출해야 backend가 확정된다 (PA1 주석 참고)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr  # 표준정규 CDF Phi. scipy.stats.norm.cdf보다 가볍고 빠름
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #
SEED = 0
N_PER_CLASS_SHOW = 8      # 클래스당 시각화할 conditional 샘플 수
N_EVAL = 2_000            # 품질 지표 계산용 샘플 수
N_RANGE = 20_000          # 픽셀 범위 통계용 샘플 수 (= 1,280,000 픽셀)
OUT = Path(__file__).resolve().parent / "outputs"


class Log:
    """화면 출력 + 파일 저장 (PA1과 동일)."""

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
# 모델 - PA1과 동일한 MLE + degenerate 샘플러
#
# import로 공유할 수도 있었지만, 두 과제를 개별 제출할 수 있도록 각 폴더를
# 독립 실행 가능하게 만들었다. (트레이드오프: 코드 중복 vs 독립성 -> 독립성 선택)
# 자세한 개념 설명은 ../pa1_single_gaussian/pa1.py 의 주석 참고.
# --------------------------------------------------------------------------- #
def load_normalized_digits():
    """(1797, 64) 크기의 [0,1] 배열과 레이블을 반환."""
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1) / 16.0
    return X, digits.target


def fit_gaussian_mle(X):
    """MLE: mu = 표본평균, Sigma = (1/N) SUM (x-mu)(x-mu)^T

    분모가 1/N 인 것이 핵심. 1/(N-1)은 unbiased estimator지 MLE가 아니다
    (np.cov의 기본값이 1/(N-1)이라 그냥 쓰면 틀린다).
    """
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / n
    return mu, (Sigma + Sigma.T) / 2.0   # 부동소수점 비대칭 제거


def build_sampler(mu, Sigma):
    """x = mu + A z  (A A^T = Sigma). singular Sigma에서도 동작 (Cholesky 불가).

    ── 왜 Cholesky가 아닌 eigendecomposition인가 ──────────────────────
    Cholesky는 positive DEFINITE를 요구하는데 이 데이터의 Sigma는 positive
    SEMI-definite다 (항상 0인 픽셀 때문에 분산이 정확히 0인 축이 있음).
    대칭 행렬은 항상 Sigma = V diag(lambda) V^T 로 분해되므로
    A = V diag(sqrt(lambda)) 로 두면 lambda = 0 이어도 A A^T = Sigma 가 성립한다.

    [주의] 클래스별 Sigma_c는 PA1의 전체 Sigma(rank 61/64)보다 "더" singular하다
    (rank 48~54/64). 클래스당 데이터가 ~180개뿐이라 그 클래스에서 항상 0인
    픽셀이 훨씬 많기 때문. 이 degenerate 샘플러가 없으면 10개 전부 터진다.
    """
    evals, evecs = np.linalg.eigh(Sigma)
    # clip: 계산 오차로 생긴 -1e-19를 0으로. sqrt(음수)=nan 방지
    A = evecs * np.sqrt(np.clip(evals, 0.0, None))

    # ── [실제로 버그를 잡은 한 줄] ────────────────────────────────────────
    # 분산이 0인 픽셀은 결정론적이다: (A A^T)_jj = Sigma_jj = 0 이므로
    # A의 j번째 "행"은 정확히 0벡터여야 한다.
    # 그런데 eigh의 반올림 오차로 행 norm이 3.3e-9로 남는다. 그러면 그 픽셀이
    # 상수값 주위에서 미세하게 흔들리는데, 그 상수값이 하필 경계인 0이라
    # 절반이 0 미만으로 떨어져 Part B의 통계를 오염시킨다.
    #   수정 전: 이론 25.59% vs 실측 27.21% (conditional은 18.97% vs 27.17%)
    #   수정 후: 이론 25.59% vs 실측 25.64% (일치)
    # 오차를 숨긴 게 아니라 수학적으로 참인 제약을 명시한 것이다.
    A[np.diag(Sigma) <= 1e-12, :] = 0.0

    def sample(rng, n):
        return mu + rng.standard_normal((n, mu.size)) @ A.T

    return sample


def gaussian_logpdf(X, mu, Sigma_reg):
    """log N(x; mu, Sigma_reg) 를 샘플별로 계산. Sigma_reg는 positive definite여야 함.

    수식:
        log N(x; mu, Sigma)
          = -1/2 [ d log(2pi) + log|Sigma| + (x-mu)^T Sigma^-1 (x-mu) ]
                                  ^ logdet     ^ Mahalanobis 거리

    ── 왜 np.linalg.inv(Sigma)를 쓰지 않는가 (중요) ───────────────────

    (1) Mahalanobis 거리 - Cholesky 트릭
        Sigma = L L^T  =>  Sigma^-1 = L^-T L^-1  이므로

            (x-mu)^T Sigma^-1 (x-mu) = || L^-1 (x-mu) ||^2

        y = L^-1 (x-mu) 는 L y = (x-mu) 를 푸는 것이고, L이 삼각행렬이라
        전진대입으로 O(d^2)에 풀린다. 역행렬을 만드는 것보다 빠르고,
        수치적으로 훨씬 안정적이다 (역행렬은 조건수가 나쁘면 오차가 증폭).

    (2) logdet - 언더플로 회피
        np.linalg.det(Sigma)를 직접 구하면? 64개 고유값이 대부분 0.01 이하라
        곱하면 1e-150 수준 -> 언더플로로 0 -> log(0) = -inf. 계산 불가능.
        Cholesky를 쓰면

            |Sigma| = |L| |L^T| = |L|^2 = (PROD L_ii)^2
            log|Sigma| = 2 SUM log L_ii

        곱 대신 "로그의 합"이라 언더플로가 원천적으로 없다.
        고차원 확률 계산의 기본 테크닉.
    """
    d = mu.size
    L = np.linalg.cholesky(Sigma_reg)
    sol = np.linalg.solve(L, (X - mu).T)          # L^-1 (x-mu),  shape (d, n)
    # einsum("ij,ij->j"): i(행)를 합산하고 j(샘플)는 남김 -> 열별 제곱합
    maha = np.einsum("ij,ij->j", sol, sol)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)


def logsumexp(M, axis):
    """log SUM exp(a_i) 를 언더플로 없이 계산하는 log-sum-exp trick.

    ── 왜 필요한가 ────────────────────────────────────────────────────
    mixture의 log p(x) = log SUM_c pi_c N(x; mu_c, Sigma_c) 를 계산하려면
    log에서 exp로 돌아가야 한다. 그런데 a_i가 -800 같은 값이면
    exp(-800) = 0 (double의 최소가 약 1e-308, exp(-745)부터 0)
    -> 전부 0 -> log(0) = -inf. 정보가 통째로 사라진다.

    ── 항등식 ─────────────────────────────────────────────────────────
        log SUM_i exp(a_i) = m + log SUM_i exp(a_i - m),   m = max a_i

        증명: SUM exp(a_i) = SUM exp(m) exp(a_i - m) = exp(m) SUM exp(a_i - m)
              양변에 log를 취하면 끝.

    효과: a_i - m <= 0 이고 최댓값이 exp(0) = 1 이므로 절대 언더플로하지 않는다.
    나머지 항이 0으로 언더플로해도 무해하다 (원래 무시해도 될 크기였으므로).

    [연결] 이 패턴은 softmax, cross-entropy, HMM forward algorithm,
           VAE의 ELBO에 전부 나온다. 한 번 이해하면 평생 쓴다.
    """
    m = M.max(axis=axis, keepdims=True)   # keepdims: broadcasting을 위해 축 유지
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)


# --------------------------------------------------------------------------- #
# 시각화 (모델 자체와는 무관한 부분)
# --------------------------------------------------------------------------- #
def figure_conditional_grid(cond_samples, path):
    """행 하나가 클래스 하나 - p(x|y=c)에서 뽑은 샘플 그리드."""
    fig, axes = plt.subplots(10, N_PER_CLASS_SHOW, figsize=(N_PER_CLASS_SHOW * 0.95, 10.2))
    for c in range(10):
        for k in range(N_PER_CLASS_SHOW):
            ax = axes[c, k]
            ax.imshow(cond_samples[c][k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                # rotation=0 : 행 레이블을 눕히지 않고 가로로
                ax.set_ylabel(f"c = {c}", fontsize=11, rotation=0,
                              labelpad=22, va="center")
    fig.suptitle("PA2 (A) — samples from $p(x \\mid y=c)$, one row per class",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_model_comparison(X_real, gen_single, gen_cond, cond_labels, path):
    """진짜 | 단일 Gaussian | class-conditional 을 4x4 블록 3개로 나란히 비교."""
    # 17열 = 4(진짜) + 2(간격) + 4(단일) + 3(간격) + 4(conditional)
    fig, axes = plt.subplots(4, 17, figsize=(15.5, 4.4))

    def draw(offset, data, labels, title):
        """offset 열부터 4x4 블록 하나를 그린다."""
        for row in range(4):
            for col in range(4):
                ax = axes[row, offset + col]
                ax.imshow(data[row * 4 + col].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if labels is not None:
                    # conditional 블록만 "어떤 c로 조건을 걸었는지" 표시
                    ax.set_title(str(labels[row * 4 + col]), fontsize=8, pad=2)
        axes[0, offset + 1].annotate(title, xy=(0.5, 1.35), xycoords="axes fraction",
                                     ha="center", fontsize=12)

    draw(0, X_real, None, "Real digits")
    draw(6, gen_single, None, "Single Gaussian (PA1)")
    draw(13, gen_cond, cond_labels, "Class-conditional (PA2)")
    for row in range(4):
        for col in (4, 5, 10, 11, 12):      # 간격 열은 축을 지운다
            axes[row, col].axis("off")
    fig.suptitle("PA2 (A) — conditioning on the class removes the between-class blur",
                 fontsize=13, y=1.12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_means(mu_global, class_mu, path):
    """전체 평균 vs 클래스별 평균.

    왼쪽 global mu는 0~9를 전부 평균낸 것이라 어떤 숫자도 아니다. 그런데
    단일 Gaussian은 바로 거기서 밀도가 가장 높다 -> PA1이 실패한 이유의 그림.
    """
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
    """왼쪽: 픽셀값 분포 히스토그램. 오른쪽: 범위를 벗어난 픽셀을 표시한 샘플 하나."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))

    ax = axes[0]
    bins = np.linspace(-1.0, 2.0, 181)
    # 학습 데이터는 1/16의 배수라 이산적인 막대로 나온다 ([0,1]에 완전히 갇힘)
    ax.hist(X_train.ravel(), bins=bins, density=True, alpha=0.55,
            label=f"training  [{X_train.min():.2f}, {X_train.max():.2f}]", color="#444444")
    ax.hist(gen_single.ravel(), bins=bins, density=True, histtype="step", lw=1.8,
            label=f"single Gaussian  [{gen_single.min():.2f}, {gen_single.max():.2f}]",
            color="#d1495b")
    ax.hist(gen_cond.ravel(), bins=bins, density=True, histtype="step", lw=1.8,
            label=f"class-conditional  [{gen_cond.min():.2f}, {gen_cond.max():.2f}]",
            color="#0f7173")
    # 주황 음영 = 실제 이미지로는 불가능한 값 영역
    ax.axvspan(-1.0, 0.0, color="orange", alpha=0.10)
    ax.axvspan(1.0, 2.0, color="orange", alpha=0.10)
    ax.axvline(0.0, color="k", lw=1, ls="--")
    ax.axvline(1.0, color="k", lw=1, ls="--")
    ax.set_yscale("log")    # 꼬리가 얇아서 log 스케일이라야 보인다
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
                # 범위를 벗어난 픽셀에 초록 테두리
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
# main
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
    # PART A.1 - 클래스별 Gaussian 10개 학습
    #
    # ── 개념: class-conditional model = Gaussian Mixture Model ──────────
    #   pi_c = N_c / N                      (class prior, 그 숫자가 나올 확률)
    #   p(x | y=c) = N(x; mu_c, Sigma_c)
    #   전체로 보면  p(x) = SUM_c pi_c N(x; mu_c, Sigma_c)  -> 봉우리 10개!
    #   PA1의 근본 한계(unimodal)를 정면으로 해결하는 구조다.
    #
    # [참고] 일반적인 GMM 학습(EM 알고리즘)과는 다르다. GMM은 클래스 레이블을
    #        모르는 상태에서 latent variable로 추정하지만, 우리는 레이블 y를
    #        알고 있어서 그냥 나눠서 각각 MLE하면 끝이다 (supervised라 훨씬 쉽다).
    # ===================================================================== #
    log("\n" + "-" * 78)
    log("PART A.1  Fit p(x | y = c) for c = 0..9   (MLE per class)")
    log("-" * 78)

    mu_global, Sigma_global = fit_gaussian_mle(X)   # 비교 대상이 될 PA1 모델
    classes = np.unique(y)
    priors, class_mu, class_Sigma, class_sampler = [], [], [], []

    log(f"{'c':>2} {'N_c':>5} {'prior':>7} {'rank(Sig_c)':>12} {'tr(Sig_c)':>11}")
    for c in classes:
        # y == c 는 (1797,) bool 배열. 이걸로 인덱싱하면 True인 행만 뽑힌다
        # (boolean mask indexing). Xc는 (178, 64) 정도가 된다.
        Xc = X[y == c]
        m, S = fit_gaussian_mle(Xc)     # PA1과 똑같은 함수를 그대로 재사용
        priors.append(len(Xc) / len(X))
        class_mu.append(m)
        class_Sigma.append(S)
        class_sampler.append(build_sampler(m, S))
        rank_c = int(np.sum(np.linalg.eigvalsh(S) > 1e-12))
        log(f"{c:>2} {len(Xc):>5} {priors[-1]:>7.4f} {rank_c:>9}/64 {np.trace(S):>11.4f}")

    priors = np.array(priors)
    class_mu = np.array(class_mu)           # (10, 64)
    class_Sigma = np.array(class_Sigma)     # (10, 64, 64)

    # ===================================================================== #
    # law of total covariance - "왜 conditioning이 돕는가"의 정량적 답
    #
    # ── 항등식 (근사가 아니라 정확히 성립) ──────────────────────────────
    #   Sigma_total = SUM_c pi_c Sigma_c  +  SUM_c pi_c (mu_c - mu)(mu_c - mu)^T
    #                 \___ within-class __/    \______ between-class _______/
    #                 "같은 숫자끼리의 차이"      "숫자 종류가 달라서 생기는 차이"
    #
    # ── 유도 ───────────────────────────────────────────────────────────
    #   클래스 c 안에서 (x - mu)를 (x - mu_c) + (mu_c - mu) 로 쪼개면
    #     E[(x-mu)(x-mu)^T | c]
    #       = E[(x-mu_c)(x-mu_c)^T | c] + (mu_c-mu)(mu_c-mu)^T + 교차항
    #       = Sigma_c + (mu_c-mu)(mu_c-mu)^T
    #   교차항이 0인 이유: E[(x - mu_c) | c] = mu_c - mu_c = 0
    #   (클래스 안에서 mu_c가 정확히 평균이므로)
    #   이제 c에 대해 pi_c로 가중평균하면 위 항등식이 나온다.
    # ===================================================================== #
    # einsum 읽는 법: 입력엔 있는데 출력엔 없는 첨자는 "합산"된다.
    #   "c,cij->ij"  ->  result[i,j] = SUM_c priors[c] * class_Sigma[c,i,j]
    #   = SUM_c pi_c Sigma_c
    within = np.einsum("c,cij->ij", priors, class_Sigma)

    diffs = class_mu - mu_global             # (10, 64), 각 행이 mu_c - mu
    #   "c,ci,cj->ij"  ->  result[i,j] = SUM_c priors[c] * diffs[c,i] * diffs[c,j]
    #   = SUM_c pi_c (mu_c - mu)(mu_c - mu)^T   (외적의 가중합)
    between = np.einsum("c,ci,cj->ij", priors, diffs, diffs)

    # 항등식이 실제로 성립하는지 확인. 4e-16 = 부동소수점 한계 수준 = 정확히 성립
    ltc_err = np.linalg.norm(within + between - Sigma_global) / np.linalg.norm(Sigma_global)
    # trace(Sigma) = SUM_j Sigma_jj = 총 분산. 행렬을 스칼라 하나로 요약하는 척도
    tr_w, tr_b, tr_t = np.trace(within), np.trace(between), np.trace(Sigma_global)

    log("\nLaw of total covariance:  Sigma_total = E_c[Sigma_c] + Cov_c[mu_c]")
    log(f"    identity check ||lhs-rhs||/||rhs||     : {ltc_err:.3e}   (exact)")
    log(f"    trace(Sigma_total)                     : {tr_t:.4f}")
    log(f"    trace(within-class  E_c[Sigma_c])      : {tr_w:.4f}  ({tr_w / tr_t:.1%})")
    log(f"    trace(between-class Cov_c[mu_c])       : {tr_b:.4f}  ({tr_b / tr_t:.1%})")
    log(f"    => {tr_b / tr_t:.1%} of the single Gaussian's variance is variation")
    log("       BETWEEN digit classes. Conditioning removes exactly that part.")
    # 해석: 단일 Gaussian이 학습한 분산의 42.1%는 "숫자 모양의 다양성"이 아니라
    #       "어떤 숫자냐"에서 온다. 단일 모델은 이 둘을 구분할 수단이 없어
    #       하나의 Sigma에 뭉쳐 넣고, 그 결과 샘플이 숫자와 숫자 "사이"의
    #       빈 공간으로 흘러간다. conditioning은 between 항을 통째로 제거한다.

    # p(x | y = c) 에서 클래스별로 샘플 생성 -> 과제 (A)의 결과물
    cond_samples = [class_sampler[c](rng, N_PER_CLASS_SHOW) for c in classes]
    figure_conditional_grid(cond_samples, OUT / "pa2_conditional_samples.png")
    figure_means(mu_global, class_mu, OUT / "pa2_class_means.png")

    # ===================================================================== #
    # PART A.2 - 품질 비교: 눈이 아니라 독립 지표 3개로
    # ===================================================================== #
    log("\n" + "-" * 78)
    log("PART A.2  Quality comparison  (three independent metrics)")
    log("-" * 78)

    sampler_single = build_sampler(mu_global, Sigma_global)

    def sample_conditional(rng, n):
        """ancestral sampling: 먼저 c ~ p(y), 그 다음 x ~ p(x | y=c).

        joint 분포 p(x,y) = p(y) p(x|y) 를 인과 순서대로 뽑는 것.
        부모(y)를 먼저, 자식(x)을 나중에 - Bayesian network 샘플링의 기본 패턴.

        [구현 요령] 샘플마다 루프를 돌면 n번 호출이지만, 클래스별로 묶으면
        10번만 호출하면 된다. n=20000에서 10배 이상 차이난다.
        """
        labels = rng.choice(classes, size=n, p=priors)   # categorical 분포에서 샘플링
        # np.empty: 초기화 없이 메모리만 확보 (어차피 전부 덮어쓰므로 zeros보다 빠름)
        out = np.empty((n, X.shape[1]))
        for c in classes:
            idx = np.where(labels == c)[0]               # 그 클래스인 위치들
            if idx.size:
                out[idx] = class_sampler[c](rng, idx.size)   # 한꺼번에 채움
        return out, labels

    gen_single = sampler_single(rng, N_EVAL)
    gen_cond, gen_cond_y = sample_conditional(rng, N_EVAL)

    # ───────────────────────────────────────────────────────────────────── #
    # [지표 1] held-out log-likelihood - 확률모델로서의 성능
    #
    # ── regularization이 필요한 이유 (정직하게 밝혀야 할 부분) ──────────
    # MLE Sigma가 singular라 R^64 위의 density가 "존재하지 않는다".
    # cholesky가 터지고, 된다 해도 log|Sigma| = -inf 다.
    # 그래서 likelihood "평가에 한해" Sigma + lambda*I 를 쓴다 (ridge/Tikhonov).
    # 모든 고유값에 lambda를 더해 positive definite를 보장하는 것으로,
    # 기하학적으로는 납작한 61차원 평면을 lambda 두께로 부풀리는 것이다.
    #
    # [중요] 이건 모델을 바꾸는 행위다. 그래서 평가에만 쓰고 샘플링은 원래
    #        MLE 모델 그대로 한다. 이 구분을 안 하면 "MLE로 fit했다"는 주장이
    #        거짓이 된다. 리포트에도 명시했다.
    #
    # ── 왜 train/val/test 3분할인가 ────────────────────────────────────
    #   train : mu, Sigma 추정
    #   val   : 하이퍼파라미터 lambda 선택
    #   test  : 최종 성능 보고 - 딱 한 번만 사용
    # lambda를 test에서 고르면 test에 과적합되어 숫자가 부풀려진다.
    # "test는 절대 건드리지 않는다"가 원칙이고, 이걸 지키느냐가 논문 리뷰에서
    # 걸리는 지점이다.
    # ───────────────────────────────────────────────────────────────────── #
    # stratify=y : 각 split에 10개 클래스가 원래 비율대로 들어가게 한다.
    # 없으면 우연히 test에 숫자 3이 2개만 들어갈 수 있고 평가가 왜곡된다.
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED,
                                            stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED,
                                          stratify=ytmp)
    log(f"\n[metric 1] held-out log-likelihood   (train {len(Xtr)} / val {len(Xva)} "
        f"/ test {len(Xte)})")

    # 공정한 비교를 위해 두 모델 모두 train split에서만 다시 학습한다
    mu_s, Sig_s = fit_gaussian_mle(Xtr)
    tr_priors, tr_mu, tr_Sig = [], [], []
    for c in classes:
        Xc = Xtr[ytr == c]
        m, S = fit_gaussian_mle(Xc)
        tr_priors.append(len(Xc) / len(Xtr)); tr_mu.append(m); tr_Sig.append(S)
    tr_priors = np.array(tr_priors)

    I = np.eye(X.shape[1])
    # log 스케일 그리드인 이유: lambda의 효과는 곱셈적이다 (1e-6과 1e-5의 차이가
    # 0.1과 0.2의 차이보다 의미 있음). 선형 그리드면 작은 값 영역을 탐색 못 한다.
    lam_grid = np.logspace(-6, 0, 25)

    def ll_single(Xq, lam):
        """단일 Gaussian의 샘플당 평균 log-likelihood."""
        return gaussian_logpdf(Xq, mu_s, Sig_s + lam * I).mean()

    def ll_cond(Xq, lam):
        """mixture의 log p(x) = log SUM_c pi_c N(x; mu_c, Sigma_c).

        각 성분의 log(pi_c) + log N(...) 을 (n, 10) 으로 쌓은 뒤
        logsumexp로 합친다 (exp로 돌아가면 언더플로하므로).
        """
        comp = np.stack([np.log(tr_priors[c]) + gaussian_logpdf(Xq, tr_mu[c],
                                                                tr_Sig[c] + lam * I)
                         for c in classes], axis=1)
        return logsumexp(comp, axis=1).mean()

    # lambda는 val에서 고르고
    lam_s = lam_grid[int(np.argmax([ll_single(Xva, l) for l in lam_grid]))]
    lam_c = lam_grid[int(np.argmax([ll_cond(Xva, l) for l in lam_grid]))]
    # 최종 보고는 test에서 (여기서 처음이자 마지막으로 test를 쓴다)
    te_s, te_c = ll_single(Xte, lam_s), ll_cond(Xte, lam_c)

    log(f"    single Gaussian     : lambda* = {lam_s:.2e}   test LL/sample = {te_s:9.3f}")
    log(f"    class-conditional   : lambda* = {lam_c:.2e}   test LL/sample = {te_c:9.3f}")
    log(f"    improvement                                   = {te_c - te_s:9.3f} nats/sample")
    log("    (higher is better; both models regularised the same way and tuned on val)")
    # 해석: 한 번도 본 적 없는 진짜 숫자 이미지에 conditional 모델이
    #       e^57 배 높은 확률밀도를 부여한다. 생성모델의 가장 표준적인 정량 평가.

    # ───────────────────────────────────────────────────────────────────── #
    # [지표 2] oracle classifier - "숫자다움"의 대리 지표
    #
    # 생성모델 평가의 고전적 난제는 "좋다"를 정의하는 것이다. 실무 표준은
    # 진짜 데이터로 학습한 분류기를 판정자로 쓰는 것 (Inception Score, FID 계열).
    #
    # [순서가 중요] 판정자의 "자격"을 먼저 검증한다. 분류기 자체가 60%밖에
    # 못 맞추면 그 판정은 아무 의미가 없다. 이 단계를 빠뜨리는 게 흔한 실수다.
    # ───────────────────────────────────────────────────────────────────── #
    clf = LogisticRegression(max_iter=5000)
    clf.fit(Xtr, ytr)
    oracle_acc = clf.score(Xte, yte)        # 96.7% -> 판정자로 쓸 자격 있음

    p_single = clf.predict_proba(gen_single)    # (2000, 10) 클래스별 확률
    p_cond = clf.predict_proba(gen_cond)

    # entropy H(p) = -SUM_c p_c log p_c
    #   최소 0          : "확실히 7이다"
    #   최대 log 10=2.303 : "전혀 모르겠다" (균등)
    # np.log(P + 1e-12) : p_c가 정확히 0이면 log(0) = -inf.
    #   수학적으로 0*log0 = 0 이지만 컴퓨터는 0 * (-inf) = nan 이므로 방지한다.
    ent = lambda P: float(np.mean(-(P * np.log(P + 1e-12)).sum(axis=1)))

    # 의도한 클래스로 인식되는가? -> conditioning이 작동한다는 직접 증거
    # (단일 Gaussian은 "의도한 클래스"가 없어서 이 지표를 만들 수 없다.
    #  그 자체가 모델의 한계를 보여준다.)
    cond_acc = float(np.mean(clf.predict(gen_cond) == gen_cond_y))

    log(f"\n[metric 2] oracle classifier (logistic regression, real test acc "
        f"= {oracle_acc:.1%})")
    log(f"    conditional samples classified as the intended c : {cond_acc:.1%}")
    log(f"    mean max class probability  - single Gaussian    : {p_single.max(1).mean():.3f}")
    log(f"    mean max class probability  - class-conditional  : {p_cond.max(1).mean():.3f}")
    log(f"    mean predictive entropy     - single Gaussian    : {ent(p_single):.3f} nats")
    log(f"    mean predictive entropy     - class-conditional  : {ent(p_cond):.3f} nats")
    log("    (low entropy / high confidence = the sample looks like ONE definite digit)")
    # 해석: 단일 Gaussian 샘플은 entropy가 3배 높다 = 분류기가 "이게 뭔지
    #       모르겠다"고 말한다 = 여러 숫자 특징이 섞인 애매한 이미지.
    #       주관적 인상을 객관적 수치로 옮긴 것이 이 지표의 가치다.

    # ───────────────────────────────────────────────────────────────────── #
    # [지표 3] 실제 데이터까지의 최근접 거리
    #
    # [기준선이 핵심] 1.344가 좋은 숫자인지 나쁜 숫자인지는 그 자체로 알 수 없다.
    # "진짜 데이터끼리도 1.089는 떨어져 있다"를 재야 비로소 판단이 가능하다.
    # 앞으로 어떤 지표를 만들든 "이 숫자를 뭐랑 비교할 것인가"를 먼저 정할 것.
    #
    # [한계] 이 지표는 다양성을 못 본다. 학습 이미지 하나를 그대로 복사해서
    # 2000개 내놓으면 거리 0으로 "완벽"하게 나온다. 그래서 지표 1, 2와
    # 함께 봐야 한다 - 지표를 3개 쓴 이유가 각각의 맹점이 다르기 때문이다.
    # ───────────────────────────────────────────────────────────────────── #
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
    # kneighbors는 (거리, 인덱스) 튜플 반환. [0]이 거리이고 shape (n,1)이라 ravel
    d_single = nn.kneighbors(gen_single)[0].ravel().mean()
    d_cond = nn.kneighbors(gen_cond)[0].ravel().mean()
    d_real = nn.kneighbors(Xte)[0].ravel().mean()       # <- 기준선(floor)

    log("\n[metric 3] mean L2 distance to the nearest REAL training image")
    log(f"    real held-out images (reference floor) : {d_real:.3f}")
    log(f"    single Gaussian samples               : {d_single:.3f}")
    log(f"    class-conditional samples             : {d_cond:.3f}")
    log("    (lower = lands closer to the real data manifold)")

    idx_real = rng.choice(len(X), size=16, replace=False)
    figure_model_comparison(X[idx_real], gen_single, gen_cond, gen_cond_y,
                            OUT / "pa2_single_vs_conditional.png")

    # ===================================================================== #
    # PART B - 픽셀 값 범위
    # ===================================================================== #
    log("\n" + "-" * 78)
    log("PART B  Pixel-value ranges: training images vs generated samples")
    log("-" * 78)

    big_single = sampler_single(rng, N_RANGE)
    big_cond, _ = sample_conditional(rng, N_RANGE)

    def out_frac(A):
        """[0,1]을 벗어난 픽셀의 비율.

        (A < 0) | (A > 1) 은 원소별 bool 배열이고, np.mean(bool)은 True를 1로
        세므로 결과가 비율이 된다. 20000 x 64 = 128만 픽셀 기준.
        """
        return float(np.mean((A < 0.0) | (A > 1.0)))

    rows = [
        ("training images", X, None),
        (f"generated - single Gaussian", big_single, None),
        (f"generated - class-conditional", big_cond, None),
    ]
    log(f"\n{'source':<32} {'min':>10} {'max':>10} {'% outside [0,1]':>17}")
    for name, A, _ in rows:
        log(f"{name:<32} {A.min():>10.4f} {A.max():>10.4f} {out_frac(A):>16.2%}")

    # 위/아래를 나눠서 보는 이유: 비대칭이 크고, 그 비대칭 자체가 설명의 근거다
    log(f"\n    below 0  - single      : {np.mean(big_single < 0):.2%}"
        f"   |  max negative magnitude {abs(big_single.min()):.3f}")
    log(f"    above 1  - single      : {np.mean(big_single > 1):.2%}")
    log(f"    below 0  - conditional : {np.mean(big_cond < 0):.2%}"
        f"   |  max negative magnitude {abs(big_cond.min()):.3f}")
    log(f"    above 1  - conditional : {np.mean(big_cond > 1):.2%}")

    # ───────────────────────────────────────────────────────────────────── #
    # 설명이 맞다는 것을 "예측"으로 증명하기 - 이 과제에서 가장 중요한 검증
    #
    # ── 개념: marginalization ──────────────────────────────────────────
    # multivariate Gaussian에서 좌표 하나만 떼어내면 그것도 Gaussian이다:
    #     x ~ N(mu, Sigma)  =>  x_j ~ N(mu_j, Sigma_jj)
    # Sigma의 나머지 4032개 원소는 전혀 필요 없고 대각 원소만 보면 된다.
    # (직관: 다른 축을 전부 적분해서 없애면 그 축의 주변분포만 남는다)
    #
    # 따라서 픽셀 j가 [0,1]을 벗어날 확률은
    #     P(x_j < 0) = Phi(-mu_j / s_j)
    #     P(x_j > 1) = 1 - Phi((1-mu_j)/s_j) = Phi((mu_j - 1)/s_j)   (Phi 대칭성)
    #
    # ── 왜 이게 결정적인가 ─────────────────────────────────────────────
    # "Gaussian의 support가 R^64라서 벗어난다"까지는 누구나 말할 수 있고,
    # 거기서 멈추면 검증 불가능한 이야기다. 하지만 "그 설명이 맞다면 정확히
    # 25.59%여야 한다"로 만들면 틀렸을 때 즉시 드러난다.
    # 실제로 이것 때문에 build_sampler의 3.3e-9 버그를 찾아냈다.
    # ───────────────────────────────────────────────────────────────────── #
    def predicted_out_frac(mu, Sigma):
        """이론적으로 예상되는 범위 이탈 픽셀 비율."""
        s = np.sqrt(np.diag(Sigma))
        # 분산 0인 픽셀은 결정론적이라 확률 0. 나누기 전에 걸러야 0/0 = nan을 피한다
        ok = s > 1e-12
        p = np.zeros_like(s)
        p[ok] = ndtr(-mu[ok] / s[ok]) + ndtr((mu[ok] - 1.0) / s[ok])
        return float(p.mean())

    pred_single = predicted_out_frac(mu_global, Sigma_global)
    # 전확률 법칙: P(out) = SUM_c pi_c P(out | c)
    pred_cond = float(np.sum(priors * np.array(
        [predicted_out_frac(class_mu[c], class_Sigma[c]) for c in classes])))
    emp_single, emp_cond = out_frac(big_single), out_frac(big_cond)

    n_zero_var = int(np.sum(np.diag(Sigma_global) < 1e-15))
    log("\n    Verification of the explanation (theory vs simulation):")
    log(f"      single Gaussian    predicted {pred_single:.4%}  vs  observed {emp_single:.4%}")
    log(f"      class-conditional  predicted {pred_cond:.4%}  vs  observed {emp_cond:.4%}")
    log(f"      pixels with zero variance (always generated exactly at their")
    log(f"      constant training value, so never out of range) : {n_zero_var}")
    #
    # ── 결과 해석 3가지 ────────────────────────────────────────────────
    # (1) 왜 벗어나는가: Gaussian의 support는 R^64 전체, 데이터는 [0,1]^64.
    #     fitting 오차가 아니라 "모델 가정 자체의 mismatch"다.
    #     어떤 mu, Sigma를 골라도 피할 수 없다.
    # (2) 왜 음수 쪽이 압도적인가 (20.21% vs 5.43%): 숫자 이미지는 대부분이
    #     배경이라 mu_j가 0 근처인 픽셀이 많다. mu_j = 0 이면 Phi(0) = 0.5,
    #     즉 정의상 절반이 음수다. 반대로 mu_j가 1 근처인 픽셀은 거의 없다
    #     (mu의 최댓값이 0.756).
    # (3) 왜 conditional이 덜 벗어나는가 (18.98% vs 25.64%): PART A.1의
    #     분산 분해가 그대로 예측한다. Sigma_c는 within-class만 담아 더 작고
    #     (trace 57.9%), s_j가 작아지면 Phi(-mu_j/s_j)도 작아진다.
    #     => A와 B가 같은 원인의 두 측면이다.

    figure_pixel_ranges(X, big_single, big_cond, OUT / "pa2_pixel_ranges.png")

    log(f"\nFigures written to {OUT}")
    for f in ("pa2_conditional_samples.png", "pa2_single_vs_conditional.png",
              "pa2_class_means.png", "pa2_pixel_ranges.png"):
        log(f"    - {f}")

    # ===================================================================== #
    # assertion 블록 - 리포트의 주장을 코드가 직접 검사한다
    #
    # 검증이 3종류로 나뉘고 허용오차도 다르다:
    #   (1) 수학적 항등식 -> 1e-12.  틀리면 코드 버그 확정
    #   (2) 통계적 수렴   -> 5e-3.   몬테카를로 오차 감안
    #   (3) 과학적 주장   -> 부등식. 리포트의 "결론" 자체를 검증
    #
    # (3)이 특히 중요하다. 리포트에 "conditioning이 낫다"고 썼는데 코드가
    # 그걸 assert한다. 나중에 코드를 고치다 그 주장이 깨지면 실행이 실패해서
    # 리포트와 코드가 어긋난 상태로 방치되지 않는다.
    # ===================================================================== #
    assert ltc_err < 1e-12, "law of total covariance identity must hold exactly"
    assert abs(priors.sum() - 1.0) < 1e-12, "priors must sum to 1"
    assert te_c > te_s, "conditional model must beat the single Gaussian on held-out LL"
    assert oracle_acc > 0.90, "oracle classifier is not trustworthy"
    assert cond_acc > 0.30, "conditional samples are not recognisable as their class"
    assert ent(p_cond) < ent(p_single), "conditional samples must be less ambiguous"
    assert d_cond < d_single, "conditional samples must be closer to real data"
    assert abs(X.min()) < 1e-12 and abs(X.max() - 1.0) < 1e-12, "training data must be [0,1]"
    # 연쇄 비교: a < b < c 는 (a<b) and (b<c) 로 풀린다.
    # "Gaussian은 반드시 [0,1]을 벗어난다"는 이론적 필연을 검증
    assert big_single.min() < 0.0 < 1.0 < big_single.max(), "Gaussian must leave [0,1]"
    # tolerance 0.005 is ~3x the Monte-Carlo noise at N_RANGE=20000; loose enough to
    # never flake, tight enough that the earlier 1.6%/8.2% bug would still be caught.
    # (허용오차를 "예전 버그를 여전히 잡을 수 있는가" 기준으로 정했다)
    assert abs(pred_single - emp_single) < 0.005, "analytic out-of-range rate mismatch"
    assert abs(pred_cond - emp_cond) < 0.005, "analytic out-of-range rate mismatch"

    log("\n" + "=" * 78)
    log("ALL CHECKS PASSED")
    log("=" * 78)
    log.save()


if __name__ == "__main__":
    main()
