"""
PA1 - Fit a Multivariate Gaussian to Images
===========================================

[과제 원문]
1. Load `sklearn.datasets.load_digits()` and normalize pixel values from 0-16 to 0-1.
2. Flatten each 8x8 image to x in R^64.
3. Fit ONE multivariate Gaussian using MLE.
4. Generate at least 20 new samples, reshape to 8x8, and visualize them.

[이 스크립트가 하는 일]
손글씨 숫자 이미지 1797장을 "R^64 공간의 점 1797개"로 보고, 그 점구름을
multivariate Gaussian(64차원 타원체) 하나로 감싼다. 그 다음 그 타원체 안에서
새로운 점을 뽑아 이미지로 되돌린다. generative model의 가장 기본 형태다.

[핵심 개념 3가지]
  - MLE      : likelihood를 최대화하는 파라미터. Gaussian은 미분해서 0으로 놓으면
               닫힌 해가 나온다 (경사하강법 불필요).
  - 샘플링    : x = mu + A z 형태의 affine 변환. A A^T = Sigma 인 A만 찾으면 된다.
  - singular : 이 데이터는 항상 0인 픽셀이 있어 Sigma가 singular고 Cholesky가
               실패한다. 고유값 분해로 우회한다.

[실행]
    python pa1.py

출력하는 모든 수치는 스크립트 맨 끝 assertion 블록이 다시 검사한다.
`ALL CHECKS PASSED`가 찍히면 모든 주장이 검증된 것이고, 하나라도 틀리면
exit code가 0이 아니게 된다. "그럴듯해 보인다"가 아니라 "검증됐다"를 만드는 구조.
"""

from pathlib import Path

import matplotlib

# [중요] matplotlib은 "어디에 그릴지"를 backend로 정하는데 기본값은 창을 띄우는
# GUI backend다. 디스플레이가 없는 환경(서버/CI)에서는 죽거나 멈춘다.
# "Agg"는 파일로만 출력하는 backend. pyplot을 import하는 순간 backend가 확정되므로
# 반드시 pyplot import보다 먼저 호출해야 한다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #
SEED = 0             # 난수 시드. 고정하면 몇 번을 돌려도 같은 결과 -> 재현성 확보
N_SHOW = 25          # 시각화할 생성 샘플 수 (과제 요구사항은 20개 이상)
N_VERIFY = 100_000   # 샘플러 검증에만 쓰는 샘플 수 (결과물에는 안 들어감)

# __file__ 기준 절대경로. 이게 없으면 상위 폴더에서 실행할 때
# outputs/ 가 엉뚱한 위치에 생긴다.
OUT = Path(__file__).resolve().parent / "outputs"


class Log:
    """화면 출력과 파일 저장을 동시에 하는 작은 유틸.

    실행 로그가 outputs/pa1_log.txt 에 그대로 남아서, 리포트에 적은 숫자가
    어디서 나왔는지의 증거가 된다.

    __call__ 을 정의하면 인스턴스를 함수처럼 쓸 수 있다 -> log("메시지")
    """

    def __init__(self, path):
        self.path = path
        self.lines = []

    def __call__(self, msg=""):
        text = str(msg)
        print(text)
        self.lines.append(text)

    def save(self):
        # encoding="utf-8" 명시: Windows 기본 인코딩(cp949)으로 쓰면 깨질 수 있다
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# [1단계 + 2단계] 데이터 로드 -> 정규화 -> flatten
# --------------------------------------------------------------------------- #
def load_normalized_digits():
    """digits 데이터를 (n, 64) 형태의 [0,1] 배열로 만들어 반환.

    ── 왜 flatten 하는가 ──────────────────────────────────────────────
    확률분포는 정의역이 있어야 한다. Gaussian N(mu, Sigma)는 R^d 위에 정의된
    분포이고 mu는 R^d 벡터, Sigma는 d x d 행렬이다. 8x8 행렬 위에 직접 정의된
    Gaussian 같은 건 없으므로, 이미지 한 장을 64차원 벡터 하나로 편다.

    잃는 것 : 픽셀 0과 1이 가로로 인접하다는 "공간 구조"가 완전히 사라진다.
              픽셀 순서를 무작위로 섞어도 모델 성능은 똑같다 (CNN과 정반대).
    얻는 것 : 대신 Sigma가 "픽셀 12와 13이 같이 진해진다"를 데이터에서 학습한다.
              구조를 가정하지 않고 상관관계로 복원하는 셈.

    ── reshape(len, -1) 의 -1 ─────────────────────────────────────────
    "나머지 차원은 알아서 계산해라"는 뜻. 1797 x ? = 1797 x 8 x 8 -> ? = 64
    numpy 기본은 C order(row-major)라 픽셀 인덱스가 j = 8*row + col 이 된다.
    나중에 .reshape(8, 8)로 되돌릴 때 같은 순서를 쓰므로 왕복이 정확히 맞는다.
    reshape는 데이터를 복사하지 않고 view만 바꾸므로 비용이 사실상 0이다.

    ── /16.0 정규화 ───────────────────────────────────────────────────
    원본이 0~16 정수이므로 16으로 나누면 [0,1]이 된다.
    수학적으로 필수는 아니다 (Gaussian은 스케일에 무관하고, 정규화를 안 해도
    mu와 Sigma가 같이 스케일되어 결과가 동일하다). 그래도 하는 이유:
      1) 과제 요구사항
      2) PA2에서 "생성값이 범위를 벗어났다"를 판단할 기준선이 생긴다  <- 핵심
      3) Sigma의 원소가 O(1) 크기라 고유값 분해가 수치적으로 안정적
    """
    digits = load_digits()
    images_raw = digits.images                          # (1797, 8, 8), 0~16 정수
    X = images_raw.reshape(len(images_raw), -1) / 16.0   # (1797, 64), [0,1] 실수

    # digits.data는 sklearn이 이미 flatten해둔 것. 내 flatten이 맞는지
    # 대조 검증하는 용도로 같이 반환한다 (main에서 np.allclose로 비교).
    return X, digits.target, images_raw, digits.data


# --------------------------------------------------------------------------- #
# [3단계] MLE - maximum likelihood estimation
# --------------------------------------------------------------------------- #
def fit_gaussian_mle(X):
    """Multivariate Gaussian의 MLE 추정.

    ── likelihood란 ───────────────────────────────────────────────────
    파라미터 theta = (mu, Sigma)를 고정하면 내 데이터가 관측될 확률밀도를
    계산할 수 있다:   L(theta) = PROD_i N(x_i; mu, Sigma)
    MLE는 이걸 최대화하는 theta를 고르는 것. "내가 실제로 본 데이터가 가장
    그럴듯해지는 파라미터"라는 뜻이다.

    곱은 다루기 나쁘다 (1797개를 곱하면 언더플로). log를 씌우면 합이 되고,
    log는 단조증가라 최대점 위치가 안 변한다:

        l(mu, Sigma) = -N/2 [ d log(2pi) + log|Sigma| ]
                       - 1/2 SUM_i (x_i - mu)^T Sigma^-1 (x_i - mu)

    ── 왜 닫힌 해가 나오는가 ──────────────────────────────────────────
    dl/dmu = 0 으로 놓으면 Sigma^-1이 양변에서 사라져 mu = 표본평균.
    dl/dSigma = 0 으로 놓으면 Sigma = 표본공분산(1/N 버전).

        mu    = (1/N) SUM_i x_i
        Sigma = (1/N) SUM_i (x_i - mu)(x_i - mu)^T        <- 분모가 1/N !

    즉 경사하강법도 반복 최적화도 필요 없다. 평균과 공분산만 계산하면 그게
    전역 최적해다. VAE나 diffusion은 전부 반복 최적화가 필요한데 Gaussian은
    한 방에 끝난다 -- 이게 Gaussian의 특별한 점.

    ── [함정] 1/N 인가 1/(N-1) 인가 ───────────────────────────────────
        1/N     : MLE. likelihood를 최대화. 분산을 약간 과소추정(biased).
                  np.cov(X.T, bias=True) 와 같음
        1/(N-1) : unbiased estimator (Bessel 보정). np.cov()의 기본값

    mu를 데이터에서 추정했기 때문에 데이터가 mu_hat 주위에 실제 mu 주위보다
    조금 더 모여 있다. 자유도를 하나 썼으므로 N-1로 나눠야 unbiased가 된다.

    과제가 "using MLE"라고 명시했으므로 1/N이 정답이다. np.cov(X.T)를 그냥
    썼으면 틀린 답이 된다. N=1797이라 실제 차이는 0.06%로 미미하지만,
    뭘 계산하고 있는지 아느냐의 문제다.
    """
    n = X.shape[0]
    mu = X.mean(axis=0)     # axis=0 : 샘플 방향으로 평균 -> (64,)
    Xc = X - mu             # 중심화(centering). broadcasting으로 각 행에서 mu를 뺀다

    # ── (Xc.T @ Xc) / n 이 왜 공분산인가 ──
    #   Xc.T @ Xc  : (64,1797) @ (1797,64) -> (64,64)
    #   (j,k) 원소 = SUM_i Xc[i,j] * Xc[i,k]
    #              = SUM_i (x_ij - mu_j)(x_ik - mu_k)
    #   정확히 공분산의 정의다. for문 없이 행렬곱 하나로 736만 번의 곱셈이
    #   BLAS에서 처리된다 (파이썬 루프보다 수백 배 빠름).
    #
    # Sigma가 담고 있는 것:
    #   대각   Sigma_jj = 픽셀 j의 분산. 배경 픽셀은 0에 가깝고 가운데는 크다
    #   비대각 Sigma_jk = 픽셀 j,k의 공분산. 양수면 "같이 진해진다"
    #
    # [중요] 만약 Sigma를 대각행렬로만 뒀다면(= 픽셀 독립 가정) 생성 결과는
    # TV 노이즈가 된다. 획이 이어지는 유일한 이유가 비대각 원소다.
    Sigma = (Xc.T @ Xc) / n

    # 수학적으로 Sigma는 반드시 대칭인데, 부동소수점 덧셈은 결합법칙이 성립하지
    # 않아서 Sigma[3,7]과 Sigma[7,3]이 1e-18 정도 차이날 수 있다.
    # 다음 단계의 np.linalg.eigh가 "대칭임"을 가정하고 하삼각만 읽으므로,
    # 미세한 비대칭이 있으면 조용히 잘못된 결과가 나올 수 있다. 방어적 코딩.
    Sigma = (Sigma + Sigma.T) / 2.0
    return mu, Sigma


# --------------------------------------------------------------------------- #
# [4단계] 샘플링 - 이 과제의 핵심 난관
# --------------------------------------------------------------------------- #
def build_sampler(mu, Sigma):
    """N(mu, Sigma)에서 샘플을 뽑는 함수를 만들어 반환한다.

    ── 왜 직접 뽑을 수 없는가 ─────────────────────────────────────────
    "64차원 Gaussian에서 뽑아라"를 하드웨어가 직접 수행할 방법은 없다.
    가진 것은 균등난수 생성기 하나뿐이고, 경로는 이렇다:

        U(0,1) --[Box-Muller/Ziggurat]--> N(0,1) --[affine]--> N(mu, Sigma)

    rng.standard_normal이 두 번째 화살표를 해준다. 우리가 할 일은 세 번째뿐.

    ── affine 변환의 핵심 정리 ────────────────────────────────────────
    z ~ N(0, I) 에서 시작해 x = mu + A z 로 두면

        E[x]   = mu + A E[z] = mu                              (OK)
        Cov[x] = Cov[Az] = A Cov[z] A^T = A I A^T = A A^T

        (유도: Cov(Az) = E[(Az)(Az)^T] = E[A z z^T A^T]
                       = A E[z z^T] A^T = A Cov(z) A^T,  A는 상수라 밖으로 나감)

    그리고 Gaussian의 affine 변환은 다시 Gaussian이다. 따라서
    ** A A^T = Sigma 인 A를 하나만 찾으면 끝 ** 이다.
    A는 유일하지 않다 (임의의 직교행렬 Q에 대해 AQ도 답). 아무거나 하나면 된다.

    [연결] 이 식이 나중에 VAE의 reparameterization trick과 정확히 같다:
           z = mu(x) + sigma(x) * eps,   eps ~ N(0, I)

    ── 왜 Cholesky를 못 쓰는가 ────────────────────────────────────────
    교과서 방법은 Sigma = L L^T (하삼각 L)로 분해하는 Cholesky이고 가장 빠르다.
    그런데 이 데이터에서는 실패한다:

        8x8 숫자 이미지에서 모서리 픽셀 3개는 1797장 전부에서 값이 0이다.
        -> Sigma_jj = Var(x_j) = 0
        -> Cauchy-Schwarz 에 의해 |Sigma_jk| <= sqrt(Sigma_jj * Sigma_kk) = 0
        -> 그 행과 열 전체가 0 -> rank가 61로 떨어짐 (singular)

        positive DEFINITE (PD)       : 모든 lambda > 0.  Cholesky 가능
        positive SEMI-definite (PSD) : 모든 lambda >= 0. Cholesky 불가능

    공분산 행렬은 항상 PSD다 (v^T Sigma v = Var(v^T x) >= 0). PD인지는
    데이터에 달렸고, 여기서는 아니다.

    ── 해결: spectral theorem ─────────────────────────────────────────
    실수 "대칭" 행렬은 항상 직교 고유벡터로 대각화된다 (조건이 대칭뿐이다):

        Sigma = V diag(lambda) V^T
        A     = V diag(sqrt(lambda))
        => A A^T = V diag(sqrt L) diag(sqrt L)^T V^T
                 = V diag(L) V^T = Sigma                        (OK)

    lambda_j = 0 이면 sqrt(0) = 0 이라 그 열이 0이 될 뿐, 수식은 완벽히 성립한다.

    기하학적 해석: V의 열은 데이터 타원체의 주축 방향, sqrt(lambda)는 그 방향의
    반지름. 샘플링은 "구에서 뽑아서(z) -> 주축마다 sqrt(lambda)배 늘리고(diag)
    -> 회전시키고(V) -> mu만큼 평행이동".

    ── degenerate Gaussian ────────────────────────────────────────────
    rank가 61이면 이 분포는 R^64 전체가 아니라 61차원 affine 부분공간
    (mu + span{lambda_j > 0 인 고유벡터}) 위에만 존재한다.
    64차원 공간에서 61차원 평면은 부피가 0이므로:

        샘플링은 된다 (OK)   /   확률밀도 p(x)는 정의되지 않는다 (Sigma^-1 없음)

    PA2에서 log-likelihood를 계산할 때 이 문제가 정면으로 걸린다.
    """
    # eigh는 대칭/Hermitian 전용 고유값 분해. 일반 eig와 비교하면
    #   - 실수 고유값 보장 (eig는 복소수 반환 가능 -> sqrt에서 터짐)
    #   - 고유벡터 직교 보장
    #   - 오름차순 정렬되어 나옴
    #   - 2배 이상 빠름
    evals, evecs = np.linalg.eigh(Sigma)

    # 이론상 lambda >= 0 이지만 계산 오차로 -1e-19 같은 값이 나온다.
    # np.sqrt(음수)는 nan이고 nan은 전염되어 모든 샘플을 망친다. 반드시 잘라낸다.
    # (진짜 음수 고유값이면 Sigma가 공분산이 아니라는 뜻인데, 그 경우는 main의
    #  assertion `evals_all.min() > -1e-10`이 따로 잡는다. 즉 -1e-19는 0으로
    #  취급하고 -0.5는 에러로 구분해 놨다.)
    evals_clipped = np.clip(evals, 0.0, None)

    # broadcasting: (64,64) * (64,) 는 각 "열" j 에 sqrt(lambda_j)를 곱한다.
    # 이게 정확히 V @ np.diag(sqrt(lambda)) 인데, 대각행렬을 실제로 만들어
    # 곱하면 26만 번 곱셈이고 broadcasting은 4096번이다.
    # (행이 아니라 열에 곱해지는 게 맞는지는 main에서 A@A.T == Sigma 로 검증)
    A = evecs * np.sqrt(evals_clipped)

    # ── [실제로 버그를 잡은 한 줄] ────────────────────────────────────────
    # 수학: Sigma_jj = 0  =>  (A A^T)_jj = ||A의 j번째 행||^2 = 0
    #                     =>  그 행은 정확히 0벡터여야 한다.
    # 현실: eigh의 반올림 오차로 행 norm이 3.3e-9로 남는다.
    # 결과: 그 픽셀이 x_j = 0 +- 3e-9 로 흔들리는데, 상수값이 하필 경계인 0이라
    #       절반이 음수가 된다. PA2의 "범위 밖 픽셀" 통계가 오염됐다
    #       (단일 모델 1.56%p, conditional 모델 8.2%p 오차).
    #
    # 이건 오차를 숨긴 게 아니라 "결정론적 픽셀"이라는 수학적으로 참인 제약을
    # 명시한 것이다. 임계값 1e-12가 안전한 이유: 픽셀값이 1/16의 배수라
    # 1797장 중 1장만 값이 있어도 분산이 (1/16)^2 / 1797 = 2.2e-6 이다.
    # 1e-12와 6자릿수 차이라 진짜 분산을 실수로 죽일 위험이 없다.
    A[np.diag(Sigma) <= 1e-12, :] = 0.0

    def sample(rng, n):
        """n개 샘플을 (n, 64) 배열로 한 번에 생성."""
        z = rng.standard_normal((n, mu.size))
        # 수식은 x = mu + A z (열벡터)인데 numpy는 샘플을 행으로 쌓으므로
        # 전치하면 x^T = mu^T + z^T A^T. 즉 z @ A.T 가 된다.
        # 이 한 줄로 n개가 동시에 생성된다 (루프 없음).
        return mu + z @ A.T

    # 함수를 반환하는 구조(closure)인 이유: eigh는 비싼 연산인데 한 번만 하고
    # A를 클로저에 가둬 재사용한다. 클래스별 샘플러 10개를 만드는 PA2에서 유용.
    return sample, evals, A


# --------------------------------------------------------------------------- #
# 시각화 (모델 자체와는 무관 - 개념 이해에는 중요하지 않은 부분)
# --------------------------------------------------------------------------- #
def figure_real_vs_generated(X_real, X_gen, path):
    """왼쪽에 진짜 숫자 5x5, 오른쪽에 생성 샘플 5x5를 나란히 그린다."""
    # 11열 = 진짜 5열 + 간격 1열 + 생성 5열
    fig, axes = plt.subplots(5, 11, figsize=(11.5, 5.6))
    for row in range(5):
        for col in range(5):
            ax = axes[row, col]
            # vmin=0, vmax=1 로 고정해야 진짜/생성의 밝기 스케일이 같아져서
            # 공정한 비교가 된다 (생성값은 [0,1]을 벗어나므로 시각적으로는 잘림).
            ax.imshow(X_real[row * 5 + col].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 5].axis("off")           # 가운데 간격 열은 비움
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
    plt.close(fig)          # 닫지 않으면 figure가 메모리에 계속 쌓인다


def figure_diagnostics(mu, Sigma, evals, A, path):
    """모델이 실제로 뭘 학습했는지 보여주는 진단 그림.

    mean / per-pixel std / Sigma 히트맵 / 고유값 스펙트럼 / 상위 6개 주성분
    """
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.35)

    # (1) 평균 이미지: 0~9를 전부 평균낸 것이라 흐릿한 세로 획이 나온다.
    #     "아무 숫자도 아닌 곳"인데 Gaussian은 여기서 밀도가 가장 높다
    #     -> PA1이 실패하는 근본 원인의 시각적 증거
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(mu.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("mean $\\mu$", fontsize=10); ax.set_xticks([]); ax.set_yticks([])

    # (2) 픽셀별 표준편차: 가운데는 변동이 크고(노랑) 모서리는 0(보라).
    #     이 보라색 모서리가 singularity의 시각적 원인이다.
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(np.sqrt(np.diag(Sigma)).reshape(8, 8), cmap="viridis")
    ax.set_title("per-pixel std $\\sqrt{\\Sigma_{jj}}$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (3) 공분산 행렬 자체. 빨강=양의 상관, 파랑=음의 상관.
    #     대각선 근처의 블록 구조 = "가로로 인접한 픽셀끼리 상관이 높다"
    ax = fig.add_subplot(gs[0, 2:4])
    ax.imshow(Sigma, cmap="RdBu_r", vmin=-np.abs(Sigma).max(), vmax=np.abs(Sigma).max())
    ax.set_title("covariance $\\Sigma$  (64x64)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # (4) 고유값 스펙트럼 (log 스케일). 61번째 이후가 1e-18로 절벽처럼 떨어지는데
    #     이게 rank 61의 시각적 증거다.
    #     np.maximum(..., 1e-20): log 스케일에서 0은 그릴 수 없으므로 바닥을 깔아줌
    #     [::-1]: eigh가 오름차순으로 주므로 뒤집어 내림차순으로 그린다
    ax = fig.add_subplot(gs[0, 4:6])
    ax.semilogy(np.maximum(evals[::-1], 1e-20), "o-", ms=3)
    ax.axhline(1e-12, color="crimson", ls="--", lw=1, label="numerical zero")
    ax.set_title("eigenvalues of $\\Sigma$ (log scale)", fontsize=10)
    ax.set_xlabel("index"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (5) 상위 6개 주성분(principal component) = 데이터 변동이 가장 큰 방향들.
    #     획의 굵기, 좌우 기울기 같은 패턴이 보인다.
    order = np.argsort(evals)[::-1]        # 고유값 내림차순 인덱스
    for k in range(6):
        ax = fig.add_subplot(gs[1, k])
        v = A[:, order[k]]
        v = v / (np.abs(v).max() + 1e-12)  # [-1,1]로 정규화해 색 스케일 통일
        ax.imshow(v.reshape(8, 8), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"PC{k + 1}\n$\\lambda$={evals[order[k]]:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("PA1 diagnostics: what the single Gaussian actually learned", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main - 전체 파이프라인
# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = Log(OUT / "pa1_log.txt")

    # default_rng vs np.random.seed:
    #   np.random.seed는 "전역 상태"를 쓴다. 라이브러리 어딘가가 난수를 쓰면
    #   내 수열이 밀려서 재현이 깨진다. default_rng는 독립된 Generator 객체를
    #   만들어 그 문제가 없다. numpy 1.17부터 권장 방식이고 통계 품질도 더 좋다.
    rng = np.random.default_rng(SEED)

    log("=" * 74)
    log("PA1 - Fit ONE multivariate Gaussian to load_digits() and sample from it")
    log("=" * 74)

    # ── [1] 데이터 ──────────────────────────────────────────────────────── #
    X, y, images_raw, sklearn_flat = load_normalized_digits()
    log("\n[1] Load + normalize + flatten")
    log(f"    raw images shape          : {images_raw.shape}   (8x8 per image)")
    log(f"    raw pixel range           : [{images_raw.min():.1f}, {images_raw.max():.1f}]  (0-16)")
    log(f"    X shape after flatten     : {X.shape}   -> each x is in R^64")
    log(f"    X pixel range after /16   : [{X.min():.4f}, {X.max():.4f}]  (0-1)")
    log(f"    n samples / n classes     : {X.shape[0]} / {len(np.unique(y))}")

    # 내가 직접 편 결과가 sklearn이 이미 flatten해둔 digits.data와 같은지 대조.
    # 축 순서를 헷갈리는 건 흔한 실수인데(예: reshape(64,-1)) 이 한 줄로 즉시 걸린다.
    flatten_matches = np.allclose(X * 16.0, sklearn_flat)
    log(f"    flatten matches digits.data: {flatten_matches}")

    # ── [2] MLE fit ─────────────────────────────────────────────────────── #
    mu, Sigma = fit_gaussian_mle(X)

    # eigvalsh는 고유값만 계산 (고유벡터 생략 -> 더 빠름)
    evals_all = np.linalg.eigvalsh(Sigma)
    # "수치적 rank" = 0이 아닌 고유값의 개수. 정확히 0인 경우는 드물고
    # 1e-18 같은 값이 나오므로 임계값이 필요하다.
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

    # ── [3] 샘플러 구축 + 검증 ──────────────────────────────────────────── #
    sample, evals, A = build_sampler(mu, Sigma)

    # 검증 (a) 대수적: 행렬 분해가 맞나?
    #   machine epsilon(2.2e-16) 수준이면 완벽하다.
    recon_err = np.linalg.norm(A @ A.T - Sigma) / np.linalg.norm(Sigma)

    # 검증 (b) 통계적: 실제로 뽑은 게 그 분포를 따르나?
    #   대수의 법칙 - S가 진짜 N(mu, Sigma)에서 왔다면 N -> 무한대일 때
    #   표본평균 -> mu, 표본공분산 -> Sigma 로 수렴해야 한다.
    #
    #   [왜 2단계인가] (a)만 하면 "A는 맞는데 z @ A.T 를 A.T @ z 로 썼다" 같은
    #   실수를 못 잡는다. (b)가 end-to-end 검증이다.
    S = sample(rng, N_VERIFY)
    mu_hat = S.mean(axis=0)
    # bias=True(1/N): 비교 대상인 Sigma가 1/N 버전이므로 일관성을 맞춘다
    Sigma_hat = np.cov(S.T, bias=True)
    err_mu = np.linalg.norm(mu_hat - mu) / np.linalg.norm(mu)
    err_Sigma = np.linalg.norm(Sigma_hat - Sigma) / np.linalg.norm(Sigma)

    log("\n[3] Sampler check   x = mu + A z,  z ~ N(0, I)")
    log(f"    ||A A^T - Sigma|| / ||Sigma||                 : {recon_err:.3e}")
    log(f"    {N_VERIFY} samples -> rel. error of mean      : {err_mu:.3e}")
    log(f"    {N_VERIFY} samples -> rel. error of covariance: {err_Sigma:.3e}")
    log("    => the empirical moments converge to (mu, Sigma): the sampler is correct.")
    # 오차 "크기"까지 예측대로인 게 중요하다. 몬테카를로 오차는 O(1/sqrt(N))이고
    # 1/sqrt(100000) = 3.2e-3. 실측이 같은 자릿수면 정상이다.
    # 만약 0.3 같은 값이 나왔다면 샘플러가 틀린 것이다.

    # ── [4] 과제 결과물: 샘플 생성 ──────────────────────────────────────── #
    X_gen = sample(rng, N_SHOW)
    log(f"\n[4] Generated {N_SHOW} new samples (>= 20 required), each reshaped to 8x8")
    log(f"    generated value range     : [{X_gen.min():.4f}, {X_gen.max():.4f}]")
    # (A < 0) | (A > 1) 은 원소별 bool 배열.
    # np.mean(bool 배열)은 True를 1로 세므로 결과가 "비율"이 된다.
    frac_out = float(np.mean((X_gen < 0.0) | (X_gen > 1.0)))
    log(f"    fraction of pixels outside [0,1] : {frac_out:.3%}")
    log("    (the training data is bounded in [0,1] but a Gaussian has support on")
    log("     all of R^64, so out-of-range pixels are expected - analysed in PA2.)")

    # replace=False: 같은 이미지를 두 번 뽑지 않게
    idx_real = rng.choice(len(X), size=N_SHOW, replace=False)
    figure_real_vs_generated(X[idx_real], X_gen, OUT / "pa1_real_vs_generated.png")
    figure_diagnostics(mu, Sigma, evals, A, OUT / "pa1_diagnostics.png")
    log(f"\n[5] Figures written to {OUT}")
    log("    - pa1_real_vs_generated.png")
    log("    - pa1_diagnostics.png")

    np.savez_compressed(OUT / "pa1_model.npz", mu=mu, Sigma=Sigma, samples=X_gen)

    # ── assertion 블록 ──────────────────────────────────────────────────── #
    # 위에서 출력한 주장이 하나라도 틀리면 여기서 실행이 멈추고 exit code != 0.
    # 검증은 성격에 따라 종류가 나뉘고 허용오차도 다르다:
    #   (1) 수학적 항등식 -> 1e-12.  틀리면 코드 버그 확정
    #   (2) 통계적 수렴   -> 2e-2.   몬테카를로 오차를 감안한 값
    #   (3) 데이터 사실   -> 정확히 일치해야 하는 값
    assert X.shape == (1797, 64), "X must be (1797, 64)"
    assert flatten_matches, "manual flatten disagrees with digits.data"
    assert abs(X.min()) < 1e-12 and abs(X.max() - 1.0) < 1e-12, "X must be exactly in [0,1]"
    assert mu.shape == (64,) and Sigma.shape == (64, 64)
    assert np.allclose(Sigma, Sigma.T), "Sigma must be symmetric"
    # 진짜 음수 고유값이면 Sigma가 공분산이 아니라는 뜻 -> clip으로 덮지 않고 잡는다
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
