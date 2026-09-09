# 생성형 AI — Programming Assignment

`sklearn.datasets.load_digits()` (8×8 손글씨 숫자)에 multivariate Gaussian을 MLE로 fitting하고 샘플을 생성하는 과제입니다. 두 과제를 각각 **독립 실행 가능한** 폴더로 분리했습니다.

```
.
├── pa1_single_gaussian/          # PA1: 전체 데이터에 Gaussian 1개
│   ├── pa1.py
│   ├── REPORT_PA1.md             # ← 분석 리포트
│   └── outputs/                  # 그림 + 실행 로그
└── pa2_conditional_gaussian/     # PA2: 클래스별 Gaussian 10개 + 픽셀 범위 분석
    ├── pa2.py
    ├── REPORT_PA2.md             # ← 분석 리포트
    └── outputs/
```

## 실행

```bash
pip install -r requirements.txt
```

```bash
cd pa1_single_gaussian && python pa1.py
```

```bash
cd pa2_conditional_gaussian && python pa2.py
```

두 스크립트 모두 시드가 고정(`SEED = 0`)되어 있어 결과가 재현됩니다. 실행이 끝나면 `outputs/`에 그림과 전체 로그가 저장됩니다.

## 과제 요구사항 대응표

### PA1 — Fit a Multivariate Gaussian to Images

| 요구사항 | 결과 |
|---|---|
| `load_digits()` 로드 + 0–16 → 0–1 정규화 | `[0, 16]` → `[0.0000, 1.0000]` |
| 8×8 → x ∈ R⁶⁴ flatten | `(1797, 8, 8)` → `(1797, 64)` |
| MLE로 Gaussian 1개 fit | `μ ∈ R⁶⁴`, `Σ ∈ R⁶⁴ˣ⁶⁴` (rank 61, singular) |
| 20개 이상 샘플 생성 + 시각화 | 25개 → `pa1_real_vs_generated.png` |

### PA2 — Class-conditional Gaussians & pixel ranges

| 요구사항 | 결과 |
|---|---|
| `p(x\|y=c)` 10개 fit, c로 conditioning해 생성 | `pa2_conditional_samples.png` |
| 단일 Gaussian과 품질 비교 | 독립 지표 3개 전부 개선 (아래) |
| conditioning이 왜 도움 되는지 설명 | law of total covariance — 전체 분산의 **42.1%가 between-class** |
| 학습/생성 이미지의 픽셀 min·max 보고 | training `[0.0000, 1.0000]` / single `[−1.3625, 2.5253]` / conditional `[−1.1519, 2.0698]` |
| 차이 서술 + Gaussian 모델 기반 설명 | Gaussian의 support가 R⁶⁴ ⊅ 유계 데이터 — 이론 예측이 실측과 0.05%p 이내 일치 |

## 핵심 결과

**단일 Gaussian은 읽을 수 있는 숫자를 만들지 못하고, class-conditional은 만듭니다.**

| 지표 | 단일 Gaussian | class-conditional |
|---|---|---|
| held-out log-likelihood (높을수록 좋음) | 11.080 | **68.463** |
| 의도한 클래스로 분류된 비율 | — | **98.8%** |
| 분류기 predictive entropy (낮을수록 좋음) | 1.039 nats | **0.360 nats** |
| 실제 이미지까지 최근접 거리 (낮을수록 좋음) | 1.814 | **1.344** (진짜 데이터 기준선 1.089) |

이유는 *law of total covariance* 로 정확히 설명됩니다:

```
Σ_total  =  Σ_c π_c Σ_c   +   Σ_c π_c (μ_c − μ)(μ_c − μ)ᵀ
            within (57.9%)     between (42.1%)
```

단일 Gaussian이 학습한 분산의 42.1%는 "숫자 모양의 다양성"이 아니라 **"어떤 숫자인가"** 에서 오는 것이라, 샘플이 여러 숫자 사이의 빈 공간으로 흘러갑니다. Conditioning은 이 항을 통째로 제거합니다.

## 구현상 주의점

- **`Σ`가 singular합니다** (rank 61/64, 클래스별로는 48–54/64). 항상 0인 픽셀이 있어서입니다. `np.linalg.cholesky`는 실패하므로 eigendecomposition `Σ = V diag(λ) Vᵀ` 로 `A = V diag(√λ)` 를 만들어 `x = μ + Az` 로 샘플링합니다.
- **MLE 공분산은 `1/N`** 입니다 (`np.cov`의 기본값인 `1/(N−1)`은 unbiased estimator이지 MLE가 아님).
- 분산이 0인 픽셀에 대해 `eigh`가 남기는 `~3e-9` 반올림 오차가 픽셀 범위 통계를 오염시켜서, 해당 행을 정확히 0으로 강제했습니다. 자세한 추적은 [REPORT_PA2.md](pa2_conditional_gaussian/REPORT_PA2.md) B.3 참고.

## 검증

두 스크립트 모두 마지막에 assertion 블록이 있고, 통과하면 `ALL CHECKS PASSED` 를 출력합니다 (실패 시 exit code ≠ 0). 검사 항목:

- flatten 결과가 sklearn의 `digits.data` 와 일치
- 정규화 후 데이터가 정확히 `[0, 1]`
- `Σ` 가 대칭이고 positive semi-definite
- `A Aᵀ = Σ` (상대오차 `1.9e-15`)
- **10만 개 샘플의 표본 평균/공분산이 `μ`, `Σ` 로 수렴** ← 샘플러가 맞다는 증거
- law of total covariance 항등식 성립 (`4.2e-16`)
- conditional 모델이 held-out likelihood·entropy·최근접거리 3개 모두에서 우세
- **범위 밖 픽셀 비율의 이론 예측값이 실측값과 일치** ← 설명이 맞다는 증거

## 환경

Python 3.12.10 / numpy 2.5.1 / scikit-learn 1.9.0 / matplotlib 3.11.1 / scipy
