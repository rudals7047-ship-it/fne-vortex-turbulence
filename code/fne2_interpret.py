"""
FNE-II Phase 2 결과 자동 판정 + Table 생성
=============================================================================
실행: fne2_full_results.npz 가 있는 곳에서 (SciServer 또는 다운로드 후)
      python3 fne2_interpret.py

기능:
  1. Gate 2 자동 판정 (H1 / H0 / 애매)
  2. Table 1 markdown 자동 생성
  3. figure 자동 갱신 (fne2_make_figure.py 호출)
  4. 어느 시나리오 텍스트를 쓸지 안내
=============================================================================
"""
import os
import numpy as np

if not os.path.exists('fne2_full_results.npz'):
    raise SystemExit("fne2_full_results.npz 없음 — Phase 2 먼저 실행")

z = np.load('fne2_full_results.npz')
summary = z['summary']  # [(H, dR, ciR_lo, ciR_hi, dT, ciT_lo, ciT_hi), ...]
CRU = float(z['CRU_turb'])
HORIZONS = [int(r[0]) for r in summary]

print("="*60)
print("FNE-II Phase 2 자동 판정")
print("="*60)

# ── Gate 2 판정 ──────────────────────────────────────────────
dR_all   = [r[1] for r in summary]
ciR_lo   = [r[2] for r in summary]
ciR_hi   = [r[3] for r in summary]
dT_all   = [r[4] for r in summary]
ciT_lo   = [r[5] for r in summary]

# 신호 강도 판정
n_sig_R   = sum(1 for lo in ciR_lo if lo > 0)      # CI 전체 > 0
n_pos_R   = sum(1 for d in dR_all if d > 0)        # 점추정 양수
monotonic = all(dR_all[i] <= dR_all[i+1] + 0.002   # 대략 단조 (노이즈 허용)
                for i in range(len(dR_all)-1))
max_dR    = max(dR_all)

print(f"\n[ΔR²(R|Y,O) 판정]")
print(f"  CI 전체>0 인 지평: {n_sig_R}/{len(HORIZONS)}")
print(f"  점추정 양수 지평:  {n_pos_R}/{len(HORIZONS)}")
print(f"  최대 ΔR²:          {max_dR:+.5f}")
print(f"  지평 단조성:        {'예' if monotonic else '아니오(노이즈 가능)'}")

# TOPO 판정
n_sig_T = sum(1 for lo in ciT_lo if lo > 0)
print(f"\n[ΔR²(TOPO|Y,O) 판정]")
print(f"  CI 전체>0 인 지평: {n_sig_T}/{len(HORIZONS)}")

# 종합 판정
print(f"\n[CRU_turb] {CRU:.4f}")

print("\n" + "="*60)
if n_sig_R >= 2 and n_pos_R >= len(HORIZONS)-1:
    verdict = "H1"
    print("판정: H1 성립 ✓✓✓")
    print("  위상이 기하 너머의 미래 소산 정보를 담음")
    print("  → 시나리오 A 텍스트 사용")
    print("  → 논문 제목: 'Vortex Topology Carries Predictive")
    print("     Information Beyond Field Geometry...'")
elif n_pos_R >= len(HORIZONS)-1:
    verdict = "H1_weak"
    print("판정: H1 약한 지지")
    print("  방향은 일관되나 일부 지평 CI가 0 포함")
    print("  → 시나리오 A (단, claim 톤 약화) 또는")
    print("     멀티박스 8/8 일관성을 주요 증거로")
else:
    verdict = "H0"
    print("판정: H0 / 경계")
    print("  → 시나리오 B 텍스트 사용")
    print("  → ontic/epistemic 경계 논문으로 전환")
    print("  → FNE-I과 묶어 cross-system 비교")
print("="*60)

# ── Table 1 markdown 생성 ────────────────────────────────────
print("\n=== Table 1 (markdown) ===\n")
lines = []
lines.append("| H | P | P+O | P+O+R | ΔR²(R\\|Y,O) | ΔR²(TOPO\\|Y,O) |")
lines.append("|---|-----|-----|-------|------------|---------------|")
# Phase 2 출력에서 P/PO/POR 값도 필요 — summary에 없으면 dR만
for r in summary:
    H, dR, lo, hi, dT, tlo, thi = r
    lines.append(f"| {int(H)} | — | — | — | "
                 f"{dR:+.5f} [{lo:+.4f},{hi:+.4f}] | "
                 f"{dT:+.5f} [{tlo:+.4f},{thi:+.4f}] |")
table_md = "\n".join(lines)
print(table_md)
print(f"\nCRU_turb = {CRU:.4f}")

with open('fne2_table1.md', 'w') as f:
    f.write(f"# FNE-II Table 1\n\n판정: {verdict}\n\n")
    f.write(table_md)
    f.write(f"\n\nCRU_turb = {CRU:.4f}\n")
print("\n저장: fne2_table1.md")

# ── figure 갱신 ──────────────────────────────────────────────
if os.path.exists('fne2_make_figure.py'):
    print("\nfigure 갱신 중...")
    os.system('python3 fne2_make_figure.py')
    print("fig_fne2_main.pdf/png 갱신 완료 (본규모 반영)")

print("\n" + "="*60)
print(f"최종: {verdict}")
print("다음: 해당 시나리오 텍스트로 논문 초안 작성")
print("="*60)
