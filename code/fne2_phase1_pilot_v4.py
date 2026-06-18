"""
FNE-II Phase 1 (pilot) v4: 지평 확장 + block bootstrap
=============================================================================
실행: SciServer Jupyter — 전체 붙여넣기 → Shift+Enter
첫 실행: 100프레임 다운로드 (쿼리 200회, 10~20분 예상) → 캐시 저장
재실행: 캐시 재사용, 수 분 내

v3 → v4 변경:
  * NFRAMES 30 → 100 (testing 토큰으로 가능 — 한도는 쿼리당 포인트 수)
  * HORIZONS [1,3,5] → [1,5,10,22,45]
    (H=22 ≈ 1 Kolmogorov 시간 τ_η, H=45 ≈ 2τ_η)
    v3 발견: ΔR²(R|Y,O)가 H와 함께 단조 증가 (0→0.0008→0.0025)
    → 위상 신호가 더 긴 지평에서 계속 자라는지 검증
  * bootstrap → moving-block bootstrap (블록 길이 10)
    프레임 자기상관으로 인한 CI 과소평가 보정
=============================================================================
"""

import os
import numpy as np

# ════════════ CONFIG ════════════════════════════════════════════
NX       = 16
CELL     = 4
F0       = 500
NFRAMES  = 100          # v4: 30 → 100
X0       = 256
DATASET  = 'isotropic1024coarse'
TOKEN = os.environ.get('JHTDB_TOKEN', 'edu.jhu.pha.turbulence.testing-201406')  # set JHTDB_TOKEN env var; falls back to public testing token
LCI_TH   = 1.0
ALIGN_TH = 0.7
CACHE    = 'fne2_phase1_frames_100.npz'   # v4: 새 캐시 파일명
HORIZONS = [1, 5, 10, 22, 45]             # v4: τ_η(≈22프레임)까지 확장
NBOOT    = 200
BLOCK    = 10                              # block bootstrap 블록 길이
# ════════════════════════════════════════════════════════════════

DX = 2*np.pi/1024
NU = 0.000185
NC = NX // CELL
NCELLS = NC**3

# ── 0. 데이터 수신 ────────────────────────────────────────────
def _normalize(result, nvals):
    if hasattr(result, 'data_vars'):
        arr = result[list(result.data_vars)[0]].values
    else:
        arr = getattr(result, 'values', result)
    arr = np.squeeze(np.asarray(arr))
    want = (NX,NX,NX,nvals) if nvals > 1 else (NX,NX,NX)
    if arr.shape == want:
        return arr
    if nvals > 1 and arr.shape == (nvals,NX,NX,NX):
        return np.moveaxis(arr, 0, -1)
    raise RuntimeError(f"예상 밖 shape: {arr.shape}")

def fetch(var, frame_index, nvals):
    from giverny.turbulence_dataset import turb_dataset
    from giverny.turbulence_toolkit import getCutout
    ds = turb_dataset(dataset_title=DATASET, output_path='./fne2_output',
                      auth_token=TOKEN)
    ar = np.array([[X0, X0+NX-1]]*3, dtype=np.int64)
    st = np.array([1,1,1], dtype=np.int64)
    return _normalize(getCutout(ds, var, int(frame_index), ar, st), nvals)

# ── 1. 물리량 ─────────────────────────────────────────────────
def gradient_tensor(u):
    G = np.zeros(u.shape[:3]+(3,3))
    for i in range(3):
        for j in range(3):
            G[...,i,j] = np.gradient(u[...,i], DX, axis=j)
    return G

def cg(field):
    if field.ndim == 3:
        return field.reshape(NC,CELL,NC,CELL,NC,CELL).mean(axis=(1,3,5))
    return np.stack([cg(field[...,k]) for k in range(field.shape[-1])], axis=-1)

def process_frame(fi):
    u = fetch('velocity', fi, 3)
    p = fetch('pressure', fi, 1)
    G = gradient_tensor(u)
    S = 0.5*(G + np.swapaxes(G,-1,-2))
    eps = 2*NU*np.einsum('...ij,...ij->...', S, S)
    eig = np.linalg.eigvals(G.reshape(-1,3,3))
    lci = np.abs(eig.imag).max(axis=1).reshape(NX,NX,NX)
    w = np.stack([G[...,2,1]-G[...,1,2],
                  G[...,0,2]-G[...,2,0],
                  G[...,1,0]-G[...,0,1]], axis=-1)
    Y = cg(eps)
    O = np.stack([
        cg(np.linalg.norm(u, axis=-1)),
        cg(np.sqrt(np.einsum('...ij,...ij->...',G,G))),
        cg(np.sqrt(np.einsum('...ij,...ij->...',S,S))),
        cg(p),
        cg(0.5*np.einsum('...k,...k->...', u, u)),
    ], axis=-1)
    return Y, O, cg(lci), cg(w)

# ── 2. 골격 그래프 ────────────────────────────────────────────
NBR6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

def skeleton_edges(LCI, W, lci_rms):
    vortex = LCI > LCI_TH*lci_rms
    Wn = W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-12)
    edges = set()
    idx = lambda i,j,k: (i*NC + j)*NC + k
    for i in range(NC):
        for j in range(NC):
            for k in range(NC):
                if not vortex[i,j,k]: continue
                for di,dj,dk in NBR6:
                    a,b,c = i+di, j+dj, k+dk
                    if not (0<=a<NC and 0<=b<NC and 0<=c<NC): continue
                    if not vortex[a,b,c]: continue
                    if abs(np.dot(Wn[i,j,k], Wn[a,b,c])) > ALIGN_TH:
                        edges.add(tuple(sorted((idx(i,j,k), idx(a,b,c)))))
    return vortex, edges

def graph_features(edges, ncells):
    import networkx as nx
    Gx = nx.Graph(); Gx.add_nodes_from(range(ncells)); Gx.add_edges_from(edges)
    deg  = np.array([Gx.degree(n) for n in range(ncells)], float)
    clus = np.array(list(nx.clustering(Gx).values()), float)
    hop2 = np.array([len(set(nb for n1 in Gx[n] for nb in Gx[n1]) - {n})
                     for n in range(ncells)], float)
    btw  = np.array(list(nx.betweenness_centrality(Gx).values()), float)
    cvar = np.array([np.std([Gx.degree(m) for m in Gx[n]]) if Gx.degree(n)>0 else 0.0
                     for n in range(ncells)], float)
    return np.stack([deg, clus, hop2, btw, cvar], axis=-1)

def topo_change(edges_a, edges_b, ncells):
    diff = edges_a ^ edges_b
    cnt = np.zeros(ncells)
    for (m,n) in diff:
        cnt[m] += 1; cnt[n] += 1
    return cnt

# ── 3. 프레임 루프 (캐시 사용) ────────────────────────────────
if os.path.exists(CACHE):
    print(f"캐시 {CACHE} 발견 — 다운로드 생략")
    z = np.load(CACHE, allow_pickle=True)
    Ys, Os, Rs, TOPO = z['Y'], z['O'], z['R'], z['TOPO']
else:
    Ys  = np.zeros((NFRAMES, NCELLS))
    Os  = np.zeros((NFRAMES, NCELLS, 5))
    Rs  = np.zeros((NFRAMES, NCELLS, 5))
    TOPO = np.zeros((NFRAMES-1, NCELLS))
    prev_edges = None
    for t in range(NFRAMES):
        Y, O, LCI, W = process_frame(F0 + t)
        lci_rms = np.sqrt((LCI**2).mean())
        vortex, edges = skeleton_edges(LCI, W, lci_rms)
        Ys[t] = Y.ravel()
        Os[t] = O.reshape(NCELLS, 5)
        Rs[t] = graph_features(edges, NCELLS)
        if prev_edges is not None:
            TOPO[t-1] = topo_change(prev_edges, edges, NCELLS)
        prev_edges = edges
        if (t+1) % 10 == 0:
            print(f"  frame {t+1}/{NFRAMES} 처리")
    np.savez_compressed(CACHE, Y=Ys, O=Os, R=Rs, TOPO=TOPO)
    print(f"캐시 저장: {CACHE}")

# ── 4. Gate 1 ────────────────────────────────────────────────
print("\n=== Gate 1: 골격 그래프 sanity ===")
mean_deg = Rs[...,0].mean(); frac_v = (Rs[...,0] > 0).mean()
print(f"  평균 degree {mean_deg:.2f} / 참여 셀 {frac_v*100:.1f}% / 재배선 {TOPO.mean():.3f}/셀·프레임")
print(f"  [Gate 1] {'PASS' if (mean_deg>0.1 and 0.02<frac_v<0.95 and TOPO.sum()>0) else 'FAIL'}")

# ── 5. Incremental 분석 + block bootstrap ────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def cv_r2(X, y, nsplit=5, seed=0):
    nf = X.shape[0]
    kf = KFold(n_splits=nsplit, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(np.arange(nf)):
        Xtr = X[tr].reshape(-1, X.shape[-1]); ytr = y[tr].ravel()
        Xte = X[te].reshape(-1, X.shape[-1]); yte = y[te].ravel()
        m = LinearRegression().fit(Xtr, ytr)
        pred = m.predict(Xte)
        r2s.append(1 - ((yte-pred)**2).sum()/((yte-yte.mean())**2).sum())
    return float(np.mean(r2s))

def block_indices(nf, rng):
    """moving-block bootstrap: 길이 BLOCK 블록들을 이어붙여 nf개 인덱스 생성"""
    nblocks = int(np.ceil(nf / BLOCK))
    starts = rng.integers(0, nf - BLOCK + 1, nblocks)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:nf]
    return idx

def boot_ci_diff_block(Xbig, Xsmall, y, B=NBOOT):
    nf = Xbig.shape[0]; rng = np.random.default_rng(0); vals = []
    for _ in range(B):
        idx = block_indices(nf, rng)
        vals.append(cv_r2(Xbig[idx], y[idx]) - cv_r2(Xsmall[idx], y[idx]))
    return np.percentile(vals, [2.5, 97.5])

print("\n=== Incremental 예측 (block bootstrap, 블록=10) ===")
print(f"    τ_η ≈ 22프레임 기준: H=22 ≈ 1τ_η, H=45 ≈ 2τ_η")
summary = []
for H in HORIZONS:
    ts  = np.arange(1, NFRAMES - H)
    tgt = Ys[ts + H]
    P    = Ys[ts][..., None]
    PO   = np.concatenate([P, Os[ts]], axis=-1)
    POR  = np.concatenate([PO, Rs[ts]], axis=-1)
    POT  = np.concatenate([PO, TOPO[ts-1][...,None]], axis=-1)

    r2P, r2PO  = cv_r2(P, tgt),  cv_r2(PO, tgt)
    r2POR, r2POT = cv_r2(POR, tgt), cv_r2(POT, tgt)
    dR_R, dR_T = r2POR - r2PO, r2POT - r2PO
    ciR = boot_ci_diff_block(POR, PO, tgt)
    ciT = boot_ci_diff_block(POT, PO, tgt)
    summary.append((H, dR_R, ciR, dR_T, ciT))

    print(f"\n  [H={H}]  (n_t={len(ts)})")
    print(f"    P             R² = {r2P:+.3f}")
    print(f"    P+O           R² = {r2PO:+.3f}")
    print(f"    P+O+R         R² = {r2POR:+.3f}   ΔR²(R|Y,O)    = {dR_R:+.4f}  CI[{ciR[0]:+.4f},{ciR[1]:+.4f}]")
    print(f"    P+O+TOPO      R² = {r2POT:+.3f}   ΔR²(TOPO|Y,O) = {dR_T:+.4f}  CI[{ciT[0]:+.4f},{ciT[1]:+.4f}]")

# ── 6. CRU_turb + 지평 스케일링 요약 ─────────────────────────
r2_O_deg = cv_r2(Os[:-1], Rs[:-1,:,0])
print(f"\n=== CRU_turb ===")
print(f"  R²(O→degree) = {r2_O_deg:.3f}  →  CRU_turb = {1-r2_O_deg:.3f}")

print("\n=== 지평 스케일링 요약: ΔR²(R|Y,O) vs H ===")
for H, dR_R, ciR, _, _ in summary:
    sig = "***" if ciR[0] > 0 else ("?" if ciR[1] > 0 else "")
    print(f"  H={H:3d}: {dR_R:+.4f}  CI[{ciR[0]:+.4f},{ciR[1]:+.4f}]  {sig}")
print("\n판독: ΔR²가 H와 함께 단조 증가 + 긴 H에서 CI>0 이면")
print("'위상은 느린 구조 정보' 가설 지지 → 본 규모 진행 근거 확보")
print("="*60)
