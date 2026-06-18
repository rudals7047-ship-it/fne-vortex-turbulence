"""
import os
FNE-II Phase 0: JHTDB cutout 수신 + 파이프라인 sanity test
(v3 — testing 토큰 한도 대응: 16^3 = 4,096 포인트)
=============================================================================
실행: SciServer Jupyter 노트북에 전체 붙여넣기 → Shift+Enter
통과 기준:
  [A] cutout 수신 (16,16,16,3) × 2프레임
  [B] <ε> 자릿수 ≈ 0.1 (README 공칭 0.103)
  [C] vortex 셀 비율 — 16^3은 박스가 작아 범위 벗어나도 무시 가능
  [D] 인접 프레임 상관 0.9~1.0
=============================================================================
"""

import numpy as np

# ── 설정 (v3: testing 토큰 한도 4096포인트에 맞춤) ──
NX = 16          # cutout 한 변: 16^3 = 4,096 (한도 딱 맞음)
CELL = 4         # 셀 한 변: 4^3 포인트/셀 → 4^3 = 64셀
X0 = 256         # cutout 시작 인덱스
T0 = 1.000       # 시작 시각
DT = 0.002       # coarse 저장 간격
DATASET = 'isotropic1024coarse'
TOKEN = os.environ.get('JHTDB_TOKEN', 'edu.jhu.pha.turbulence.testing-201406')  # set JHTDB_TOKEN env var; falls back to public testing token

# ── 데이터 수신 ────────────────────────────────
def _normalize(result):
    """giverny 반환형(xarray/ndarray) → (NX,NX,NX,3) ndarray로 통일"""
    arr = getattr(result, 'values', result)
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.shape == (NX, NX, NX, 3):
        return arr
    if arr.shape == (3, NX, NX, NX):
        return np.moveaxis(arr, 0, -1)
    raise RuntimeError(f"예상 밖 shape: {arr.shape} — 이 출력 그대로 공유해주세요")

def fetch_velocity(t):
    from giverny.turbulence_dataset import turb_dataset
    from giverny.turbulence_toolkit import getCutout
    dataset = turb_dataset(dataset_title=DATASET,
                           output_path='./fne2_output',
                           auth_token=TOKEN)
    axes_ranges = np.array([[X0, X0+NX-1],
                            [X0, X0+NX-1],
                            [X0, X0+NX-1]], dtype=np.int64)
    strides = np.array([1, 1, 1], dtype=np.int64)
    try:
        result = getCutout(dataset, 'velocity', t, axes_ranges, strides)
    except Exception as e:
        # 일부 버전은 시각(float) 대신 프레임 번호(int)를 요구
        if 'time' in str(e).lower():
            frame = int(round(t / DT))
            result = getCutout(dataset, 'velocity', frame, axes_ranges, strides)
        else:
            raise
    return _normalize(result)

print("=== [A] cutout 수신 테스트 ===")
u0 = fetch_velocity(T0)
u1 = fetch_velocity(T0 + DT)
okA = u0.shape == (NX,NX,NX,3) and u1.shape == (NX,NX,NX,3)
print(f"  frame t={T0}:    shape={u0.shape}")
print(f"  frame t={T0+DT}: shape={u1.shape}")
print(f"  [A] {'PASS' if okA else 'FAIL'}")

# ── 미분/물리량 ────────────────────────────────
DX = 2*np.pi/1024
NU = 0.000185    # isotropic1024 동점성계수 (README에서 재확인 필요)

def gradient_tensor(u):
    G = np.zeros(u.shape[:3]+(3,3))
    for i in range(3):
        for j in range(3):
            G[...,i,j] = np.gradient(u[...,i], DX, axis=j)
    return G

def dissipation(G):
    S = 0.5*(G + np.swapaxes(G, -1, -2))
    return 2*NU*np.einsum('...ij,...ij->...', S, S)

def swirling_strength(G):
    eig = np.linalg.eigvals(G.reshape(-1,3,3))
    return np.abs(eig.imag).max(axis=1).reshape(G.shape[:3])

print("\n=== [B] 소산율 sanity ===")
G0 = gradient_tensor(u0)
eps0 = dissipation(G0)
okB = 0.01 < eps0.mean() < 1.0
print(f"  <ε> = {eps0.mean():.4f}  (공칭 ≈ 0.103)")
print(f"  분포: min={eps0.min():.4f}, median={np.median(eps0):.4f}, max={eps0.max():.2f}")
print(f"  [B] {'PASS' if okB else 'FAIL'}")
print(f"  주의: 16^3 단일 박스는 국소 영역이라 공칭값과 수 배 차이 가능 (자릿수만 확인)")

print("\n=== [C] swirling strength ===")
lci0 = swirling_strength(G0)
lci_rms = np.sqrt((lci0**2).mean())
frac = (lci0 > 1.0*lci_rms).mean()
okC = 0.10 < frac < 0.70
print(f"  λ_ci,rms = {lci_rms:.3f}")
print(f"  vortex 셀 비율 (>1.0·rms): {frac*100:.1f}%")
print(f"  [C] {'PASS' if okC else 'FAIL'}  (16^3에서는 FAIL이어도 무시 가능)")

print("\n=== [D] 프레임 간 상관 ===")
corr = np.corrcoef(u0.ravel(), u1.ravel())[0,1]
okD = 0.9 < corr < 1.0
print(f"  corr(u_t, u_t+δt) = {corr:.5f}")
print(f"  [D] {'PASS' if okD else 'FAIL'}")

# ── 셀 coarse-graining + ΔY ───────────────────
NC = NX // CELL

def coarse_grain(field):
    return field.reshape(NC,CELL,NC,CELL,NC,CELL).mean(axis=(1,3,5))

print("\n=== 셀 coarse-graining ===")
Y0 = coarse_grain(eps0)
Y1 = coarse_grain(dissipation(gradient_tensor(u1)))
dY = Y1 - Y0
print(f"  셀 격자: {Y0.shape} (총 {Y0.size}셀)")
print(f"  ΔY: mean={dY.mean():.2e}, std={dY.std():.2e}")

print("\n" + "="*50)
results = {'A': okA, 'B': okB, 'C': okC, 'D': okD}
for k, v in results.items():
    print(f"  [{k}] {'PASS ✓' if v else 'FAIL ✗'}")
core = okA and okB and okD   # C는 16^3에서 비핵심
print(f"\n{'>>> 핵심 항목(A/B/D) 통과 — 배관 정상. 개인 토큰 받으면 Phase 1 진행' if core else '>>> 미통과 — 출력 전체를 공유해주세요'}")
print("="*50)
