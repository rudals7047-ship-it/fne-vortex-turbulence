"""
FNE-II Phase 1 multi-box v2: 1-based 인덱싱 수정 + giverny 업데이트
=============================================================================
v1 → v2 변경:
  * JHTDB 2024-09-16부터 1-based 인덱싱 적용
    → X0=0 무효 → POSITIONS [0,128,...] → [1,129,...]
  * giverny 최신버전으로 업데이트 (첫 셀에서 자동)
  * 기존 캐시(fne2_phase1_frames_100.npz, X0=256)는
    새 X0=257에 해당하지 않으므로 재다운로드
=============================================================================
"""

import os
import subprocess
import sys

# ── giverny 업데이트 (첫 실행 시 한 번) ──────────────────────
print("giverny 업데이트 중...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--upgrade', 'givernylocal', '-q'],
    capture_output=True, text=True
)
if result.returncode != 0:
    # 패키지명이 다를 수 있으므로 pip 방식으로도 시도
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--upgrade', 'giverny', '-q'],
        capture_output=True, text=True
    )
print("giverny 업데이트 완료\n")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# ════════════ CONFIG ════════════════════════════════════════════
NX       = 16
CELL     = 4
F0       = 500
NFRAMES  = 100
DATASET  = 'isotropic1024coarse'
TOKEN = os.environ.get('JHTDB_TOKEN', 'edu.jhu.pha.turbulence.testing-201406')  # set JHTDB_TOKEN env var; falls back to public testing token
LCI_TH   = 1.0
ALIGN_TH = 0.7
HORIZONS = [1, 5, 10, 22, 45]
NBOOT    = 200
BLOCK    = 10

# 1-based 인덱싱: 최솟값 1, 간격 128 (총 8위치)
POSITIONS = [1, 129, 257, 385, 513, 641, 769, 897]
# ════════════════════════════════════════════════════════════════

DX = 2*np.pi/1024
NU = 0.000185
NC = NX // CELL
NCELLS = NC**3

# ── 공통 함수 ─────────────────────────────────────────────────
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

def fetch(var, frame_index, nvals, X0):
    from giverny.turbulence_dataset import turb_dataset
    from giverny.turbulence_toolkit import getCutout
    ds = turb_dataset(dataset_title=DATASET, output_path='./fne2_output',
                      auth_token=TOKEN)
    # 1-based: [X0, X0+NX-1], 유효범위 1~1024
    ar = np.array([[X0, X0+NX-1]]*3, dtype=np.int64)
    st = np.array([1,1,1], dtype=np.int64)
    return _normalize(getCutout(ds, var, int(frame_index), ar, st), nvals)

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

def process_frame(fi, X0):
    u = fetch('velocity', fi, 3, X0)
    p = fetch('pressure', fi, 1, X0)
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
    for (m,n) in diff:
        cnt[m] += 1; cnt[n] += 1
    return cnt

def cv_r2(X, y, nsplit=5, seed=0):
    nf = X.shape[0]
    nsplit = min(nsplit, max(2, nf // 4))
    kf = KFold(n_splits=nsplit, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(np.arange(nf)):
        Xtr = X[tr].reshape(-1, X.shape[-1]); ytr = y[tr].ravel()
        Xte = X[te].reshape(-1, X.shape[-1]); yte = y[te].ravel()
        m = LinearRegression().fit(Xtr, ytr)
        pred = m.predict(Xte)
        ss_tot = ((yte-yte.mean())**2).sum()
        r2s.append(1 - ((yte-pred)**2).sum()/ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(r2s))

def analyze_box(X0):
    cache = f'fne2_box_x{X0}.npz'
    if os.path.exists(cache):
        print(f"  캐시 {cache} 재사용")
        z = np.load(cache)
        Ys, Os, Rs, TOPO = z['Y'], z['O'], z['R'], z['TOPO']
    else:
        Ys   = np.zeros((NFRAMES, NCELLS))
        Os   = np.zeros((NFRAMES, NCELLS, 5))
        Rs   = np.zeros((NFRAMES, NCELLS, 5))
        TOPO = np.zeros((NFRAMES-1, NCELLS))
        prev_edges = None
        for t in range(NFRAMES):
            Y, O, LCI, W = process_frame(F0 + t, X0)
            lci_rms = np.sqrt((LCI**2).mean())
            vortex, edges = skeleton_edges(LCI, W, lci_rms)
            Ys[t] = Y.ravel()
            Os[t] = O.reshape(NCELLS, 5)
            Rs[t] = graph_features(edges, NCELLS)
            if prev_edges is not None:
                TOPO[t-1] = topo_change(prev_edges, edges, NCELLS)
            prev_edges = edges
            if (t+1) % 20 == 0:
                print(f"    frame {t+1}/{NFRAMES}")
        np.savez_compressed(cache, Y=Ys, O=Os, R=Rs, TOPO=TOPO)
        print(f"  캐시 저장: {cache}")

    box_results = {}
    for H in HORIZONS:
        ts  = np.arange(1, NFRAMES - H)
        tgt = Ys[ts + H]
        P   = Ys[ts][..., None]
        PO  = np.concatenate([P, Os[ts]], axis=-1)
        POR = np.concatenate([PO, Rs[ts]], axis=-1)
        POT = np.concatenate([PO, TOPO[ts-1][...,None]], axis=-1)
        box_results[H] = {
            'r2P':   cv_r2(P,   tgt),
            'r2PO':  cv_r2(PO,  tgt),
            'r2POR': cv_r2(POR, tgt),
            'r2POT': cv_r2(POT, tgt),
        }
    r2_O_deg = cv_r2(Os[:-1], Rs[:-1,:,0])
    return box_results, r2_O_deg

# ── 메인 ─────────────────────────────────────────────────────
print("="*60)
print(f"FNE-II multi-box v2: {len(POSITIONS)}개 위치 × {NFRAMES}프레임")
print(f"위치: {POSITIONS}  (1-based)")
print("="*60)

all_results, all_cru = {}, {}

for pos_i, X0 in enumerate(POSITIONS):
    print(f"\n[{pos_i+1}/{len(POSITIONS)}] X0={X0} 처리 중...")
    try:
        br, cru = analyze_box(X0)
        all_results[X0] = br
        all_cru[X0] = cru
        print(f"  CRU_turb={1-cru:.3f} | " +
              " ".join(f"H{H}:{br[H]['r2POR']-br[H]['r2PO']:+.4f}"
                       for H in HORIZONS))
    except Exception as e:
        print(f"  X0={X0} 에러: {e}")
        continue

# ── 합산 ─────────────────────────────────────────────────────
print("\n" + "="*60)
valid = [p for p in POSITIONS if p in all_results]
print(f"=== 합산 결과: {len(valid)}개 위치 ===")

print("\n  지평 스케일링 ΔR²(R|Y,O):")
print("  " + "  ".join(f"X{p:04d}" for p in valid) + "  | mean")
for H in HORIZONS:
    vals = [all_results[p][H]['r2POR'] - all_results[p][H]['r2PO']
            for p in valid]
    sig = "***" if all(v > 0 for v in vals) else \
          ("**"  if sum(v > 0 for v in vals) >= len(vals)*0.75 else "")
    row = "  ".join(f"{v:+.4f}" for v in vals)
    print(f"  H={H:3d}: {row}  | {np.mean(vals):+.4f} {sig}")

cru_vals = [1-all_cru[p] for p in valid]
print(f"\n  CRU_turb: {[f'{v:.3f}' for v in cru_vals]}")
print(f"  평균 CRU_turb: {np.mean(cru_vals):.3f} ± {np.std(cru_vals):.3f}")

np.savez('fne2_multibox_results.npz',
         positions=np.array(valid),
         cru_vals=np.array(cru_vals))
print("\n결과 저장: fne2_multibox_results.npz")
print("이 출력 전체를 복사해서 공유해주세요.")
print("="*60)
