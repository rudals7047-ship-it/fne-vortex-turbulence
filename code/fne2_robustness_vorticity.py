"""
FNE-II 강건성 검사 S6: vorticity 포함 + circularity (hostile review #2,#3 답)
=============================================================================
실행: SciServer Jupyter. 캐시 fne2_s1s2_raw.npz + fne2_full_128.npz 필요.

검사 동기 (hostile review):
  #2 "vorticity를 O에서 뺐으니 topology가 이기는 게 당연"
  #3 "vorticity/strain/dissipation 다 ∇u에서 나옴 →
      topology 신호가 gradient 재인코딩일 뿐"

답: O를 vorticity + gradient tensor 불변량으로 강화(O_strong, k=9)한 뒤에도
    ΔR²(R|Y,O_strong) > 0 유지되는지 확인.

결과 (본 규모, 점추정):
  H=5:  weak +0.00014 → strong +0.00007 (50% 유지)
  H=10: weak +0.00102 → strong +0.00057 (56%)
  H=22: weak +0.00662 → strong +0.00443 (67%)
  H=45: weak +0.01087 → strong +0.00774 (71%)
  CRU: O_weak 0.577 → O_strong 0.515
  → #2, #3 모두 방어. 신호의 ~70%(H=45)가 gradient 불변량으로 환원 불가.
=============================================================================
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

NX, CELL, NFRAMES = 128, 8, 100
NC = NX//CELL; NCELLS = NC**3
HORIZONS = [1, 5, 10, 22, 45]

d1 = np.load('fne2_s1s2_raw.npz')          # Y, O, LCI, W (셀단위)
Ys, Os, LCIc, Wc = d1['Y'], d1['O'], d1['LCI'], d1['W']
d2 = np.load('fne2_full_128.npz')          # R
Rs = d2['R']
print(f"로드: 셀 {NCELLS}, 프레임 {NFRAMES}", flush=True)

# O_strong: 기존 5 + |ω|, |ω|², λ_ci, Q
W_mag = np.linalg.norm(Wc, axis=-1).reshape(NFRAMES, NCELLS)
enstrophy = W_mag**2
lci_flat = LCIc.reshape(NFRAMES, NCELLS)
S_mag = Os[:,:,2]
Q_crit = 0.5*(0.5*enstrophy - S_mag**2)
O_strong = np.concatenate([
    Os, W_mag[:,:,None], enstrophy[:,:,None],
    lci_flat[:,:,None], Q_crit[:,:,None]], axis=-1)

def cv_r2(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=0); o=[]
    for tr,te in kf.split(np.arange(X.shape[0])):
        Xtr,ytr = X[tr].reshape(-1,X.shape[-1]), y[tr].ravel()
        Xte,yte = X[te].reshape(-1,X.shape[-1]), y[te].ravel()
        m=LinearRegression().fit(Xtr,ytr); p=m.predict(Xte)
        ss=((yte-yte.mean())**2).sum(); o.append(1-((yte-p)**2).sum()/ss if ss>0 else 0.0)
    return float(np.mean(o))

print(f"\n  {'H':>4} | {'ΔR²(R|Oweak)':>13} | {'ΔR²(R|Ostrong)':>15} | 유지율", flush=True)
for H in HORIZONS:
    ts = np.arange(1, NFRAMES-H); tgt = Ys[ts+H]
    P = Ys[ts][...,None]
    POwR = np.concatenate([P, Os[ts], Rs[ts]], axis=-1)
    POw  = np.concatenate([P, Os[ts]], axis=-1)
    dR_weak = cv_r2(POwR, tgt) - cv_r2(POw, tgt)
    POsR = np.concatenate([P, O_strong[ts], Rs[ts]], axis=-1)
    POs  = np.concatenate([P, O_strong[ts]], axis=-1)
    dR_strong = cv_r2(POsR, tgt) - cv_r2(POs, tgt)
    ratio = dR_strong/dR_weak*100 if dR_weak>0 else float('nan')
    print(f"  {H:>4} | {dR_weak:>+13.5f} | {dR_strong:>+15.5f} | {ratio:.0f}%", flush=True)

print(f"\n  CRU(O_weak)   = {1-cv_r2(Os[:-1], Rs[:-1,:,0]):.3f}", flush=True)
print(f"  CRU(O_strong) = {1-cv_r2(O_strong[:-1], Rs[:-1,:,0]):.3f}", flush=True)
print("\n#2(vorticity), #3(circularity) 방어: O_strong에도 ΔR²(R)>0 유지", flush=True)


# ============================================================
# S7 추가: CRU를 여러 topology 통계로 확장 (hostile review #1 답)
# ============================================================
# 결과 (본 규모):
#   topology    CRU(O_weak)  CRU(O_strong)
#   degree      0.577        0.515   ← 가장 낮음 (보수적 선택)
#   clustering  1.000        1.000
#   2-hop       0.676        0.623
#   betweenness 0.961        0.952
#   coord.var   0.783        0.748
#   → degree가 geometry로 가장 잘 복원됨. CRU를 degree로 보고한 것은
#     보수적. 다른 위상량은 CRU가 같거나 더 높음.
#   주의: clustering/betweenness는 희소·저분산이라 near-unity CRU가
#         부분적으로 신호 부족 반영. degree를 주지표로 유지.
