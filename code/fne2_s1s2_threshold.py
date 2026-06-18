"""
FNE-II 민감도 검사 S1/S2: 골격 임계값 의존성
=============================================================================
실행: SciServer Jupyter — 전체 붙여넣기 → Shift+Enter
토큰 필요 (velocity/pressure 재수신). 단 1회만 받아 9조합 재사용.

검사:
  S1. vortex 임계 LCI_TH ∈ {0.5, 1.0, 1.5} × λ_ci,rms
  S2. 엣지 정렬 ALIGN_TH ∈ {0.5, 0.7, 0.9}
  → 9개 조합에서 ΔR²(R|Y,O) 방향(양수)·지평증가가 유지되는가?
  유지되면 "결론이 임계값 선택에 robust" → reviewer 방어

핵심 설계: process_frame에서 LCI, W를 셀 단위로 미리 저장.
  골격(skeleton_edges)만 임계값 바꿔 재계산 → velocity 재다운로드 없음.
소요: 다운로드 ~10분(100프레임×2변수) + 9조합 그래프 재계산 ~15분
=============================================================================
"""
import os, time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# ════════════ CONFIG ════════════
NX, CELL, F0, NFRAMES, X0 = 128, 8, 500, 100, 1
DATASET = 'isotropic1024coarse'
TOKEN = os.environ.get('JHTDB_TOKEN', 'edu.jhu.pha.turbulence.testing-201406')  # set JHTDB_TOKEN env var; falls back to public testing token
HORIZONS_TEST = [22, 45]              # 신호 큰 지평만
LCI_THS   = [0.5, 1.0, 1.5]           # S1
ALIGN_THS = [0.5, 0.7, 0.9]           # S2
CACHE_RAW = 'fne2_s1s2_raw.npz'       # LCI/W 셀값 캐시 (골격 재계산용)
DX, NU = 2*np.pi/1024, 0.000185
NC = NX//CELL; NCELLS = NC**3
# ═════════════════════════════════

def _normalize(result, nvals):
    if hasattr(result, 'data_vars'):
        arr = result[list(result.data_vars)[0]].values
    else:
        arr = getattr(result, 'values', result)
    arr = np.squeeze(np.asarray(arr))
    want = (NX,NX,NX,nvals) if nvals>1 else (NX,NX,NX)
    if arr.shape == want: return arr
    if nvals>1 and arr.shape == (nvals,NX,NX,NX): return np.moveaxis(arr,0,-1)
    raise RuntimeError(f"shape {arr.shape}")

def fetch(var, fi, nvals):
    from giverny.turbulence_dataset import turb_dataset
    from giverny.turbulence_toolkit import getCutout
    ds = turb_dataset(dataset_title=DATASET, output_path='./fne2_output', auth_token=TOKEN)
    ar = np.array([[X0,X0+NX-1]]*3, dtype=np.int64); st = np.array([1,1,1],dtype=np.int64)
    return _normalize(getCutout(ds, var, int(fi), ar, st), nvals)

def gradient_tensor(u):
    G = np.zeros(u.shape[:3]+(3,3))
    for i in range(3):
        for j in range(3):
            G[...,i,j] = np.gradient(u[...,i], DX, axis=j)
    return G

def cg(field):
    if field.ndim==3:
        return field.reshape(NC,CELL,NC,CELL,NC,CELL).mean(axis=(1,3,5))
    return np.stack([cg(field[...,k]) for k in range(field.shape[-1])], axis=-1)

# ── 1회 수신: Y, O, 그리고 골격 재계산용 LCI/W 셀값 저장 ──
if os.path.exists(CACHE_RAW):
    print(f"캐시 {CACHE_RAW} 재사용", flush=True)
    d = np.load(CACHE_RAW)
    Ys, Os, LCIc, Wc = d['Y'], d['O'], d['LCI'], d['W']
else:
    print("velocity/pressure 수신 + LCI/W 셀값 저장 (1회)...", flush=True)
    Ys  = np.zeros((NFRAMES, NCELLS))
    Os  = np.zeros((NFRAMES, NCELLS, 5))
    LCIc= np.zeros((NFRAMES, NC, NC, NC))
    Wc  = np.zeros((NFRAMES, NC, NC, NC, 3))
    for t in range(NFRAMES):
        u = fetch('velocity', F0+t, 3); p = fetch('pressure', F0+t, 1)
        G = gradient_tensor(u); S = 0.5*(G+np.swapaxes(G,-1,-2))
        eps = 2*NU*np.einsum('...ij,...ij->...', S, S)
        eig = np.linalg.eigvals(G.reshape(-1,3,3))
        lci = np.abs(eig.imag).max(axis=1).reshape(NX,NX,NX)
        w = np.stack([G[...,2,1]-G[...,1,2], G[...,0,2]-G[...,2,0],
                      G[...,1,0]-G[...,0,1]], axis=-1)
        Ys[t] = cg(eps).ravel()
        Os[t] = np.stack([cg(np.linalg.norm(u,axis=-1)),
                          cg(np.sqrt(np.einsum('...ij,...ij->...',G,G))),
                          cg(np.sqrt(np.einsum('...ij,...ij->...',S,S))),
                          cg(p), cg(0.5*np.einsum('...k,...k->...',u,u))],
                         axis=-1).reshape(NCELLS,5)
        LCIc[t] = cg(lci); Wc[t] = cg(w)
        if (t+1)%20==0: print(f"  frame {t+1}/{NFRAMES}", flush=True)
    np.savez_compressed(CACHE_RAW, Y=Ys, O=Os, LCI=LCIc, W=Wc)
    print(f"저장: {CACHE_RAW}", flush=True)

# ── 골격 + R features (임계값 인자화) ──
NBR6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
def build_R(LCI, W, lci_th, align_th):
    import networkx as nx
    Rs = np.zeros((NFRAMES, NCELLS, 5))
    for t in range(NFRAMES):
        lci_rms = np.sqrt((LCI[t]**2).mean())
        vortex = LCI[t] > lci_th*lci_rms
        Wn = W[t]/(np.linalg.norm(W[t],axis=-1,keepdims=True)+1e-12)
        edges=set(); idx=lambda i,j,k:(i*NC+j)*NC+k
        for i in range(NC):
            for j in range(NC):
                for k in range(NC):
                    if not vortex[i,j,k]: continue
                    for di,dj,dk in NBR6:
                        a,b,c=i+di,j+dj,k+dk
                        if not(0<=a<NC and 0<=b<NC and 0<=c<NC): continue
                        if not vortex[a,b,c]: continue
                        if abs(np.dot(Wn[i,j,k],Wn[a,b,c]))>align_th:
                            edges.add(tuple(sorted((idx(i,j,k),idx(a,b,c)))))
        Gx=nx.Graph(); Gx.add_nodes_from(range(NCELLS)); Gx.add_edges_from(edges)
        deg=np.array([Gx.degree(n) for n in range(NCELLS)],float)
        clus=np.array(list(nx.clustering(Gx).values()),float)
        hop2=np.array([len(set(nb for n1 in Gx[n] for nb in Gx[n1])-{n}) for n in range(NCELLS)],float)
        btw=np.array(list(nx.betweenness_centrality(Gx).values()),float)
        cvar=np.array([np.std([Gx.degree(m) for m in Gx[n]]) if Gx.degree(n)>0 else 0.0 for n in range(NCELLS)],float)
        Rs[t]=np.stack([deg,clus,hop2,btw,cvar],axis=-1)
    return Rs

def cv_r2(X,y):
    kf=KFold(n_splits=5,shuffle=True,random_state=0); o=[]
    for tr,te in kf.split(np.arange(X.shape[0])):
        Xtr,ytr=X[tr].reshape(-1,X.shape[-1]),y[tr].ravel()
        Xte,yte=X[te].reshape(-1,X.shape[-1]),y[te].ravel()
        m=LinearRegression().fit(Xtr,ytr); p=m.predict(Xte)
        ss=((yte-yte.mean())**2).sum(); o.append(1-((yte-p)**2).sum()/ss if ss>0 else 0.0)
    return float(np.mean(o))

def dR2(Rs,H):
    ts=np.arange(1,NFRAMES-H); tgt=Ys[ts+H]
    P=Ys[ts][...,None]; PO=np.concatenate([P,Os[ts]],axis=-1)
    POR=np.concatenate([PO,Rs[ts]],axis=-1)
    return cv_r2(POR,tgt)-cv_r2(PO,tgt)

# ── 9조합 스캔 ──
print("\n" + "="*60, flush=True)
print("=== S1/S2: 9개 임계값 조합 × ΔR²(R|Y,O) ===", flush=True)
print(f"  (기준 조합: LCI_TH=1.0, ALIGN_TH=0.7)\n", flush=True)
print(f"  {'LCI_TH':>7} {'ALIGN':>6} | " + " ".join(f"H={H}" for H in HORIZONS_TEST), flush=True)
print("  " + "-"*50, flush=True)
results = {}
for lci_th in LCI_THS:
    for align_th in ALIGN_THS:
        t0=time.time()
        Rs = build_R(LCIc, Wc, lci_th, align_th)
        vals = [dR2(Rs,H) for H in HORIZONS_TEST]
        results[(lci_th,align_th)] = vals
        star = " (기준)" if (lci_th==1.0 and align_th==0.7) else ""
        print(f"  {lci_th:>7.1f} {align_th:>6.1f} | " +
              " ".join(f"{v:+.5f}" for v in vals) +
              f"  ({time.time()-t0:.0f}s){star}", flush=True)

# ── 판정 ──
print("\n" + "="*60, flush=True)
all_pos = all(all(v>0 for v in vals) for vals in results.values())
all_mono = all(results[k][1] > results[k][0] for k in results)  # H45 > H22
print(f"  모든 9조합에서 ΔR²>0 (양쪽 지평): {'예 ✓' if all_pos else '아니오'}", flush=True)
print(f"  모든 9조합에서 H45 > H22 (지평증가): {'예 ✓' if all_mono else '아니오'}", flush=True)
print(f"\n  → {'결론이 임계값 선택에 robust. S1/S2 통과.' if (all_pos and all_mono) else '일부 조합에서 흔들림 — 검토 필요.'}", flush=True)
print("이 출력 전체를 공유해주세요.", flush=True)
print("="*60, flush=True)
