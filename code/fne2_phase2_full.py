"""
FNE-II Phase 2 (본 규모): 128³ × 100프레임 전체 분석
=============================================================================
실행: SciServer Jupyter — 전체 붙여넣기 → Shift+Enter
토큰: <your JHTDB token> (개인 토큰, 16GB 한도)

규모:
  128³ = 2,097,152 포인트 → 8³ 셀 → 16³ = 4,096 셀
  100 프레임 × 2 변수 = 200 쿼리
  예상 다운로드: ~3-4 GB / 예상 시간: 30~60분

파일럿(16³×8위치) 대비:
  셀 수:    64 × 8 = 512  →  4,096  (×8배)
  해상도:   셀당 25η      →  셀당 12η  (vortex 구조 해상)
  통계:     6,400 obs     →  409,600 obs (×64배)

Note on 1-based indexing:
  JHTDB 2024-09-16부터 1-based. X0=1이 첫 번째 포인트.
  128³ 박스: [1, 128] × [1, 128] × [1, 128]
=============================================================================
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# ════════════ CONFIG ════════════════════════════════════════════
NX       = 128
CELL     = 8           # 셀당 8³=512 DNS 포인트 → 16³=4,096 셀
F0       = 500
NFRAMES  = 100
X0       = 1           # 1-based 시작 인덱스 (도메인 코너)
DATASET  = 'isotropic1024coarse'
TOKEN = os.environ.get('JHTDB_TOKEN', 'edu.jhu.pha.turbulence.testing-201406')  # set JHTDB_TOKEN env var; falls back to public testing token
LCI_TH   = 1.0
ALIGN_TH = 0.7
HORIZONS = [1, 5, 10, 22, 45]
NBOOT    = 500         # 본 규모: 부트스트랩 늘림
BLOCK    = 10
CACHE    = 'fne2_full_128.npz'
# ════════════════════════════════════════════════════════════════

DX = 2*np.pi/1024
NU = 0.000185
NC = NX // CELL        # 16
NCELLS = NC**3         # 4,096

print(f"설정 확인:")
print(f"  박스: {NX}³, 셀: {NC}³={NCELLS}개, CELL={CELL}³ DNS포인트/셀")
print(f"  프레임: {NFRAMES}, 토큰: {TOKEN[:20]}...")
print(f"  예상 데이터량: {NX**3 * 4 * 4 * NFRAMES / 1e9:.2f} GB")

# ── 데이터 수신 ────────────────────────────────────────────────
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

# ── 물리량 ────────────────────────────────────────────────────
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
    return (cg(eps),
            np.stack([cg(np.linalg.norm(u, axis=-1)),
                      cg(np.sqrt(np.einsum('...ij,...ij->...',G,G))),
                      cg(np.sqrt(np.einsum('...ij,...ij->...',S,S))),
                      cg(p),
                      cg(0.5*np.einsum('...k,...k->...', u, u))],
                     axis=-1),
            cg(lci), cg(w))

# ── 골격 그래프 ───────────────────────────────────────────────
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
    Gx = nx.Graph(); Gx.add_nodes_from(range(ncells))
    Gx.add_edges_from(edges)
    deg  = np.array([Gx.degree(n) for n in range(ncells)], float)
    clus = np.array(list(nx.clustering(Gx).values()), float)
    hop2 = np.array([len(set(nb for n1 in Gx[n] for nb in Gx[n1]) - {n})
                     for n in range(ncells)], float)
    btw  = np.array(list(nx.betweenness_centrality(Gx).values()), float)
    cvar = np.array([np.std([Gx.degree(m) for m in Gx[n]]) if Gx.degree(n)>0
                     else 0.0 for n in range(ncells)], float)
    return np.stack([deg, clus, hop2, btw, cvar], axis=-1)

def topo_change(edges_a, edges_b, ncells):
    diff = edges_a ^ edges_b
    cnt = np.zeros(ncells)
    for (m,n) in diff: cnt[m] += 1; cnt[n] += 1
    return cnt

# ── 프레임 루프 ───────────────────────────────────────────────
if os.path.exists(CACHE):
    print(f"\n캐시 {CACHE} 발견 — 다운로드 생략")
    z = np.load(CACHE)
    Ys, Os, Rs, TOPO = z['Y'], z['O'], z['R'], z['TOPO']
else:
    print(f"\n데이터 다운로드 시작 ({NFRAMES}프레임)...")
    Ys   = np.zeros((NFRAMES, NCELLS))
    Os   = np.zeros((NFRAMES, NCELLS, 5))
    Rs   = np.zeros((NFRAMES, NCELLS, 5))
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
            print(f"  frame {t+1}/{NFRAMES}  "
                  f"(vortex {vortex.mean()*100:.0f}%, 엣지 {len(edges)})")
    np.savez_compressed(CACHE, Y=Ys, O=Os, R=Rs, TOPO=TOPO)
    print(f"캐시 저장: {CACHE}")

# ── Gate 1 ────────────────────────────────────────────────────
print("\n=== Gate 1: 골격 그래프 sanity ===")
mean_deg = Rs[...,0].mean(); frac_v = (Rs[...,0]>0).mean()
print(f"  평균 degree: {mean_deg:.2f}")
print(f"  참여 셀 비율: {frac_v*100:.1f}%")
print(f"  재배선: {TOPO.mean():.3f}/셀·프레임")
print(f"  [Gate 1] {'PASS' if mean_deg>0.1 and 0.02<frac_v<0.95 and TOPO.sum()>0 else 'FAIL'}")

# ── 회귀 함수 ─────────────────────────────────────────────────
def cv_r2(X, y, nsplit=5, seed=0):
    nf = X.shape[0]
    nsplit = min(nsplit, max(2, nf//4))
    kf = KFold(n_splits=nsplit, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(np.arange(nf)):
        Xtr = X[tr].reshape(-1, X.shape[-1]); ytr = y[tr].ravel()
        Xte = X[te].reshape(-1, X.shape[-1]); yte = y[te].ravel()
        m = LinearRegression().fit(Xtr, ytr)
        pred = m.predict(Xte)
        ss_tot = ((yte-yte.mean())**2).sum()
        r2s.append(1-((yte-pred)**2).sum()/ss_tot if ss_tot>0 else 0.0)
    return float(np.mean(r2s))

def block_ci_diff(Xbig, Xsmall, y, B=NBOOT):
    nf = Xbig.shape[0]; rng = np.random.default_rng(0); vals = []
    for _ in range(B):
        nblocks = int(np.ceil(nf/BLOCK))
        starts = rng.integers(0, max(1, nf-BLOCK+1), nblocks)
        idx = np.concatenate([np.arange(s,s+BLOCK) for s in starts])[:nf]
        vals.append(cv_r2(Xbig[idx],y[idx]) - cv_r2(Xsmall[idx],y[idx]))
    return np.percentile(vals, [2.5, 97.5])

# ── 본 분석: Incremental ──────────────────────────────────────
print("\n=== Incremental 예측: Y(t+H) | P, P+O, P+O+R, P+O+TOPO ===")
print(f"    (셀 {NCELLS}개 × 프레임 {NFRAMES}개, block bootstrap B={NBOOT})")

summary = []
for H in HORIZONS:
    ts  = np.arange(1, NFRAMES-H)
    tgt = Ys[ts+H]
    P   = Ys[ts][...,None]
    PO  = np.concatenate([P, Os[ts]], axis=-1)
    POR = np.concatenate([PO, Rs[ts]], axis=-1)
    POT = np.concatenate([PO, TOPO[ts-1][...,None]], axis=-1)

    r2P  = cv_r2(P,   tgt)
    r2PO = cv_r2(PO,  tgt)
    r2POR= cv_r2(POR, tgt)
    r2POT= cv_r2(POT, tgt)
    dR   = r2POR - r2PO
    dT   = r2POT - r2PO
    ciR  = block_ci_diff(POR, PO, tgt)
    ciT  = block_ci_diff(POT, PO, tgt)
    summary.append((H, r2P, r2PO, r2POR, r2POT, dR, dT, ciR, ciT))

    sigR = "***" if ciR[0]>0 else ("?" if ciR[1]>0 else "")
    sigT = "***" if ciT[0]>0 else ("?" if ciT[1]>0 else "")
    print(f"\n  [H={H}]  (n_t={len(ts)}, obs={len(ts)*NCELLS:,})")
    print(f"    P(persistence)  R² = {r2P:+.4f}")
    print(f"    P+O             R² = {r2PO:+.4f}")
    print(f"    P+O+R           R² = {r2POR:+.4f}  "
          f"ΔR²(R|Y,O)    = {dR:+.5f}  CI[{ciR[0]:+.5f},{ciR[1]:+.5f}]  {sigR}")
    print(f"    P+O+TOPO        R² = {r2POT:+.4f}  "
          f"ΔR²(TOPO|Y,O) = {dT:+.5f}  CI[{ciT[0]:+.5f},{ciT[1]:+.5f}]  {sigT}")

# ── CRU_turb ─────────────────────────────────────────────────
r2_O_deg = cv_r2(Os[:-1], Rs[:-1,:,0])
CRU_turb = 1 - r2_O_deg
print(f"\n=== CRU_turb ===")
print(f"  R²(O→degree) = {r2_O_deg:.4f}  →  CRU_turb = {CRU_turb:.4f}")

# ── 지평 스케일링 요약 ────────────────────────────────────────
print("\n=== 지평 스케일링 요약 ===")
print(f"  {'H':>5}  {'ΔR²(R|Y,O)':>12}  {'CI_lo':>10}  {'CI_hi':>10}  sig")
for H, _, _, _, _, dR, dT, ciR, ciT in summary:
    sig = "***" if ciR[0]>0 else ("?" if ciR[1]>0 else "")
    print(f"  {H:>5}  {dR:>+12.5f}  {ciR[0]:>+10.5f}  {ciR[1]:>+10.5f}  {sig}")

# 저장
np.savez('fne2_full_results.npz',
         horizons=np.array(HORIZONS),
         summary=np.array([(H,dR,ciR[0],ciR[1],dT,ciT[0],ciT[1])
                           for H,_,_,_,_,dR,dT,ciR,ciT in summary]),
         CRU_turb=CRU_turb)

print("\n결과 저장: fne2_full_results.npz")
print("이 출력 전체를 복사해서 공유해주세요.")
print("="*60)
