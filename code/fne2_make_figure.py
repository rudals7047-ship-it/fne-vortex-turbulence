"""
FNE-II Figure 생성기
====================
Phase 2 결과(fne2_full_results.npz)가 나오면 이 스크립트로 논문 figure 생성.
SciServer에서 결과 npz를 다운로드한 뒤 로컬/Claude 환경에서 실행하거나,
SciServer에서 직접 실행해도 됨.

생성 figure:
  fig_fne2_main.pdf/png — 2-panel:
    (a) 지평 스케일링: ΔR²(R|Y,O) vs H, 멀티박스 8위치 + 본규모
    (b) CRU 비교: FNE 대형/소형/난류/유리전이 배치도
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── 색상 (FNE 논문과 통일) ───
C_O   = '#E07B39'   # geometry
C_R   = '#3A6EA5'   # relational
C_NET = '#5BAD72'

# ════════════════════════════════════════════════════════
# 멀티박스 파일럿 결과 (이미 확보됨 — 8위치 × 5지평)
# ════════════════════════════════════════════════════════
HORIZONS = [1, 5, 10, 22, 45]
MULTIBOX_dR = {  # ΔR²(R|Y,O), 위치별
    1:   [0.0000, 0.0000, 0.0000, 0.0001, 0.0000, 0.0001, 0.0000, 0.0001],
    5:   [0.0006, 0.0053, 0.0005, 0.0020, 0.0006, 0.0014, 0.0011, 0.0014],
    10:  [0.0037, 0.0231, 0.0006, 0.0082, 0.0028, 0.0031, 0.0054, 0.0046],
    22:  [0.0046, -0.0332, 0.0009, 0.0174, 0.0024, 0.0164, 0.0059, 0.0099],
    45:  [0.0397, 0.0096, 0.0005, 0.0234, 0.0089, 0.0028, 0.0026, 0.0068],
}
MULTIBOX_CRU = [0.462, 0.523, 0.473, 0.483, 0.593, 0.693, 0.521, 0.604]

# ════════════════════════════════════════════════════════
# 본 규모 결과 (Phase 2 — 나오면 여기 채움)
# ════════════════════════════════════════════════════════
FULL_RESULTS = None  # fne2_full_results.npz 로드 시 채워짐
import os
if os.path.exists('fne2_full_results.npz'):
    z = np.load('fne2_full_results.npz')
    FULL_RESULTS = z['summary']  # [(H, dR, ciR_lo, ciR_hi, dT, ciT_lo, ciT_hi), ...]
    FULL_CRU = float(z['CRU_turb'])
    print("본 규모 결과 로드됨")
else:
    print("본 규모 결과 아직 없음 — 멀티박스만으로 figure 생성")

# ─── Figure ───
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel (a): 지평 스케일링
ax = axes[0]
# 멀티박스 개별 위치 (연한 점)
for pos_i in range(8):
    yv = [MULTIBOX_dR[H][pos_i] for H in HORIZONS]
    ax.plot(HORIZONS, yv, 'o-', color=C_R, alpha=0.18, lw=0.8, ms=3)
# 멀티박스 평균 (진한 선)
mb_mean = [np.mean(MULTIBOX_dR[H]) for H in HORIZONS]
mb_se   = [np.std(MULTIBOX_dR[H])/np.sqrt(8) for H in HORIZONS]
ax.errorbar(HORIZONS, mb_mean, yerr=mb_se, fmt='s-', color=C_R,
            lw=2, ms=7, capsize=4, label='Multi-box mean (8 positions, 16³)', zorder=5)
# 본 규모 (나오면)
if FULL_RESULTS is not None:
    Hf   = FULL_RESULTS[:,0]
    dRf  = FULL_RESULTS[:,1]
    lo   = FULL_RESULTS[:,2]; hi = FULL_RESULTS[:,3]
    ax.errorbar(Hf, dRf, yerr=[dRf-lo, hi-dRf], fmt='D-', color='#B5341A',
                lw=2.2, ms=8, capsize=5, label='Full scale (128³, 4096 cells)', zorder=6)

ax.axhline(0, color='#999', lw=0.8, ls='--')
ax.axvline(22, color='#bbb', lw=0.8, ls=':')
ax.text(22, ax.get_ylim()[1]*0.93, r'$\tau_\eta$', fontsize=9, color='#888', ha='center')
ax.set_xlabel('Prediction horizon H (frames)', fontsize=10)
ax.set_ylabel(r'$\Delta R^2(R \,|\, Y, O)$', fontsize=10)
ax.set_title('(a) Topology adds predictive info, growing with horizon', fontsize=10)
ax.legend(fontsize=7.5, loc='upper left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel (b): CRU 비교 배치도
ax = axes[1]
systems = ['Glass\n(Kob-Andersen)', 'Turbulence\n(this work)',
           'FNE small\n(N=29)', 'FNE large\n(N=824)']
cru_means = [0.0, np.mean(MULTIBOX_CRU), 0.635, 0.961]
cru_errs  = [0.0, np.std(MULTIBOX_CRU),  0.075, 0.024]
colors_b  = ['#999', C_NET, C_O, C_R]
if FULL_RESULTS is not None:
    cru_means[1] = FULL_CRU
ypos = np.arange(len(systems))
ax.barh(ypos, cru_means, xerr=cru_errs, color=colors_b, alpha=0.85,
        capsize=4, height=0.6)
ax.set_yticks(ypos); ax.set_yticklabels(systems, fontsize=8.5)
ax.set_xlabel('CRU  (geometry underdetermination of contact/topology)', fontsize=9.5)
ax.set_title('(b) Where turbulence sits on the CRU axis', fontsize=10)
ax.set_xlim(0, 1.0)
for i, v in enumerate(cru_means):
    ax.text(v+0.02, i, f'{v:.2f}', va='center', fontsize=8.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_fne2_main.pdf', bbox_inches='tight', dpi=150)
plt.savefig('fig_fne2_main.png', bbox_inches='tight', dpi=150)
print("저장: fig_fne2_main.pdf / .png")
