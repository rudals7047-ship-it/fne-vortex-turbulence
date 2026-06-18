"""
FNE-II 민감도 검사 (S3 counting artifact + S5 GBM robustness)
=============================================================================
실행: SciServer Jupyter — 전체 붙여넣기 → Shift+Enter
전제: Phase 2 캐시 fne2_full_128.npz 가 같은 폴더에 있어야 함
      (없으면 fne2_phase2_full.py 먼저 실행)
소요: 캐시만 사용. S5(GBM)가 무거워 5~15분.

검사 항목:
  S3. counting artifact 차단 — degree를 baseline에 넣고도
      higher-order 4개(clustering/2-hop/betweenness/coord.var)가
      추가 신호를 주는가? (FNE 논문의 동일 방어 논리)
  S5. GBM robustness — 비선형 모델에서도 ΔR²(R|Y,O)>0 유지되는가?
      (선형 모델 한계 때문이 아님을 입증)
=============================================================================
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

# ════════════ CONFIG (Phase 2와 동일) ════════════
NFRAMES  = 100
HORIZONS = [1, 5, 10, 22, 45]
CACHE    = 'fne2_full_128.npz'
NBOOT    = 300        # 민감도용 (Phase 2의 500보다 줄여 속도 확보)
BLOCK    = 10
# ════════════════════════════════════════════════

if not os.path.exists(CACHE):
    raise SystemExit(f"{CACHE} 없음 — fne2_phase2_full.py 먼저 실행하세요")

z = np.load(CACHE)
Ys, Os, Rs, TOPO = z['Y'], z['O'], z['R'], z['TOPO']
NCELLS = Ys.shape[1]
print(f"캐시 로드: 셀 {NCELLS}, 프레임 {NFRAMES}")
print(f"R features 순서: [0]degree [1]clustering [2]2-hop [3]betweenness [4]coord.var\n")

# ── 회귀 함수 ─────────────────────────────────────────────────
def cv_r2(X, y, nsplit=5, seed=0, model='linear'):
    nf = X.shape[0]; nsplit = min(nsplit, max(2, nf//4))
    kf = KFold(n_splits=nsplit, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(np.arange(nf)):
        Xtr = X[tr].reshape(-1, X.shape[-1]); ytr = y[tr].ravel()
        Xte = X[te].reshape(-1, X.shape[-1]); yte = y[te].ravel()
        if model == 'gbm':
            m = HistGradientBoostingRegressor(max_iter=80, max_depth=3,
                                              learning_rate=0.1, random_state=0)
        else:
            m = LinearRegression()
        m.fit(Xtr, ytr); pred = m.predict(Xte)
        ss = ((yte-yte.mean())**2).sum()
        r2s.append(1-((yte-pred)**2).sum()/ss if ss>0 else 0.0)
    return float(np.mean(r2s))

def block_ci(func, *arrays, B=NBOOT):
    """func(idx)를 block-bootstrap. arrays[0]의 길이로 재표집."""
    nf = arrays[0].shape[0]; rng = np.random.default_rng(0); vals = []
    for _ in range(B):
        nblocks = int(np.ceil(nf/BLOCK))
        starts = rng.integers(0, max(1, nf-BLOCK+1), nblocks)
        idx = np.concatenate([np.arange(s,s+BLOCK) for s in starts])[:nf]
        vals.append(func(idx))
    return np.percentile(vals, [2.5, 97.5])

# ════════════════════════════════════════════════════════════
# 기준값 재현 (Phase 2 선형)
# ════════════════════════════════════════════════════════════
print("="*62)
print("=== 기준값 재현 (선형, Phase 2 확인) ===")
base = {}
for H in HORIZONS:
    ts  = np.arange(1, NFRAMES-H); tgt = Ys[ts+H]
    P   = Ys[ts][...,None]
    PO  = np.concatenate([P, Os[ts]], axis=-1)
    POR = np.concatenate([PO, Rs[ts]], axis=-1)
    base[H] = cv_r2(POR, tgt) - cv_r2(PO, tgt)
    print(f"  H={H:3d}: ΔR²(R|Y,O) = {base[H]:+.5f}")

# ════════════════════════════════════════════════════════════
# S3. counting artifact 차단
#   baseline = P + O + degree
#   추가 = clustering, 2-hop, betweenness, coord.var (4개)
#   이게 양수면 → degree 외 위상 정보 존재 (단순 count 아님)
# ════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("=== S3. counting artifact 차단 ===")
print("  baseline=P+O+degree, 추가=higher-order 4개")
print("  양수+CI>0 이면 → degree 카운팅 아닌 진짜 위상 구조 정보\n")
for H in HORIZONS:
    ts  = np.arange(1, NFRAMES-H); tgt = Ys[ts+H]
    P   = Ys[ts][...,None]
    PO_deg = np.concatenate([P, Os[ts], Rs[ts,:,0:1]], axis=-1)      # +degree
    PO_deg_high = np.concatenate([PO_deg, Rs[ts,:,1:]], axis=-1)     # +나머지4

    dR_high = cv_r2(PO_deg_high, tgt) - cv_r2(PO_deg, tgt)
    ci = block_ci(lambda idx: cv_r2(PO_deg_high[idx], tgt[idx])
                              - cv_r2(PO_deg[idx], tgt[idx]),
                  PO_deg_high)
    sig = "***" if ci[0] > 0 else ("?" if ci[1] > 0 else "")
    print(f"  H={H:3d}: ΔR²(higher-order | Y,O,degree) = {dR_high:+.5f}  "
          f"CI[{ci[0]:+.5f},{ci[1]:+.5f}]  {sig}")

# ════════════════════════════════════════════════════════════
# S5. GBM robustness
#   선형 대신 gradient boosting으로 ΔR²(R|Y,O) 재측정
#   양수 유지되면 → 선형 모델 한계 때문이 아님
# ════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("=== S5. GBM robustness (비선형) ===")
print("  선형 vs GBM 방향 일치하면 → 신호가 모델 무관\n")
for H in HORIZONS:
    ts  = np.arange(1, NFRAMES-H); tgt = Ys[ts+H]
    P   = Ys[ts][...,None]
    PO  = np.concatenate([P, Os[ts]], axis=-1)
    POR = np.concatenate([PO, Rs[ts]], axis=-1)
    dR_gbm = cv_r2(POR, tgt, model='gbm') - cv_r2(PO, tgt, model='gbm')
    match = "일치" if (dR_gbm > 0) == (base[H] > 0) else "불일치"
    print(f"  H={H:3d}: 선형 {base[H]:+.5f}  vs  GBM {dR_gbm:+.5f}   [{match}]")

# ════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("판독:")
print("  S3 모든 H 양수 → 위상 신호가 degree 카운팅이 아님 (구조 정보)")
print("  S5 선형·GBM 방향 일치 → 신호가 선형모델 한계 때문이 아님")
print("이 출력 전체를 공유해주세요.")
print("="*62)
