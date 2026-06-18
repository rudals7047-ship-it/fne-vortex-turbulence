# FNE-II 논문 초안 골격 (v0.1)
## Vortex Topology Carries Predictive Information Beyond Field Geometry in Turbulent Dissipation

**상태:** 본 규모 결과 대기 중. 멀티박스 파일럿 결과 + 구조 확정.
**저자:** Kyeongmin Kim (Independent Researcher)
**자매논문:** Kim (2026) FNE-I (granular), Zenodo 10.5281/zenodo.20484747

---

## Title 후보
1. "Vortex Topology Carries Predictive Information Beyond Field Geometry in Turbulent Dissipation"
2. "When Does Geometry Underdetermine Interaction? A Cross-System Test in Turbulence"
3. "Relational Structure and Dissipation Forecasting in Isotropic Turbulence"

→ 1번이 FNE-I과 가장 평행 (geometry vs relation 축 명시)

---

## Abstract (구조)

```
[배경] 난류 소산 예측에서 장 기하(velocity gradient 등)가 표준 입력.
       그러나 vortex 위상 구조가 기하 너머의 정보를 담는지는 미정량화.

[방법] JHTDB isotropic1024, 128³ × 100프레임.
       Eulerian 셀을 노드로 하는 vortex 골격 그래프 구성.
       Incremental 예측: 미래 소산 Y(t+H)를
       persistence + geometry(O) 위에 relation(R)이 추가하는 ΔR² 측정.

[결과] ΔR²(R|Y,O) > 0, 예측 지평 H와 함께 증가
       (H=1~45 프레임, ~2 Kolmogorov 시간).
       8개 독립 공간 위치에서 방향 일관 (멀티박스).
       CRU_turb = [본규모 값] — geometry가 vortex 위상을
       부분적으로만(약 50%) 결정.

[해석] 위상 정보는 장 기하가 포착하지 못하는 "느린 구조" 정보.
       FNE-I(granular, CRU 0.94-0.99)과 대비:
       난류는 epistemic 미결정(CRU~0.5), granular는 ontic(CRU~0.95).
       두 시스템에서 서로 다른 메커니즘으로 relation이 정보를 담음.

[범위] 단일 데이터셋, 셀 기반 골격(vortex line tracking 아님).
       예측 association이지 인과 아님.
```

---

## 1. Introduction (차별화 핵심)

선행연구를 세 그룹으로 명시 후 빈칸 제시:

```
그룹 1 — 재연결-캐스케이드 (Yao-Hussain, Kerr):
  "재연결이 에너지 캐스케이드를 주도한다"
  → 단, 이상화된 초기조건(반평행 튜브, 매듭)
  → 정상상태 난류에서 예측력 정량화 아님

그룹 2 — ML 소산 예측 (성층/대기 DNS):
  "장 기하로 소산 예측 가능"
  → 단, O만 입력. R과의 complexity-matched 비교 없음

그룹 3 — 난류 복잡계 네트워크 (Taira 2016, Iacobello, Yeh):
  "vortical network로 구조 특성화 / 흐름 제어 / ROM"
  → 단, 예측력 비교·CRU·frame-to-frame ΔR 분석 없음

→ 빈칸: "위상 변화 사건이 국소 소산에 대해
         장 기하 너머의 예측 정보를 갖는가 —
         정상상태 난류에서 — 는 미정량화"
```

차별화 문장 (초안):
> "Network representations of turbulence have been used for structural
> characterization and flow control [Taira, Iacobello, Yeh]; vortex
> reconnection has been identified as a dominant cascade pathway in
> idealized configurations [Yao-Hussain, Kerr]. However, whether
> topology-change events carry predictive information for local
> dissipation beyond field-geometry features—in statistically
> stationary turbulence—has not been quantified. We address this by
> adapting the relational-advantage framework of [FNE-I] to turbulence."

FNE-I과의 연결:
> "In granular force-network ensembles, coordinate geometry was found to
> underdetermine realized contact structure (CRU = 0.56-0.99), and this
> gap was associated with the predictive advantage of relational
> features [FNE-I]. Granular FNE is an *ontic* underdetermination:
> equilibrium leaves contact forces genuinely undetermined. Here we ask
> whether an analogous—though *epistemic*—gap exists in turbulence,
> where vorticity is formally a derivative of velocity yet may be poorly
> recovered by low-order field-geometry features."

---

## 2. Methods

### 2.1 Data
- JHTDB isotropic1024coarse, Re_λ≈433
- 128³ cutout, 스냅샷 500-599 (100프레임)
- δt=0.002, τ_η≈0.045 → H=1~45 = 0.04~2 τ_η
- ν=0.000185 (※ JHTDB README 최종 확인)

### 2.2 Cell / Variable construction
- 8³ DNS 포인트 → 1 셀, 16³=4,096 셀
- Y = 셀 평균 소산 ε = 2ν⟨S_ij S_ij⟩
- O (k=5): |u|, |∇u|, |S|, p, ½|u|²
- R (k=5): degree, clustering, 2-hop, betweenness, coord.variance
  (vortex 골격 그래프에서)

### 2.3 Vortex skeleton graph
- 노드: λ_ci > 1.0·λ_ci,rms 인 셀 (swirling strength)
- 엣지: 인접(6-neighbor) + vorticity 정렬 |cosθ| > 0.7
- → Taira(2016) vortical network의 변형. 단 본 연구는
  예측력 분해 용도 (구조 특성화 아님). 명시 필요.

### 2.4 CRU_turb
```
CRU_turb ≡ 1 − R²(O → degree_skeleton)
```
- FNE-I과 동일 정의 구조
- 주의: 난류에서 ω=curl(u)이므로 R은 O로부터 원리상 유도 가능.
  CRU_turb는 "원리적 미결정"이 아니라 "low-order feature로의
  예측 실패" = epistemic 간극을 측정. (FNE-I과의 결정적 차이)

### 2.5 Incremental prediction (leakage 차단)
- 동시간 ΔO→ΔY는 target leakage (Y=2ν|S|², O에 |S| 포함). 폐기.
- 예측 형태: 미래 수준 Y(t+H), persistence Y(t) baseline 포함.
- 비교: P / P+O / P+O+R / P+O+TOPO
- 핵심 지표: ΔR²(R|Y,O) = r2(P+O+R) − r2(P+O)
- = FNE-I의 History-augmented (H+R > H+O) 테스트의 난류 버전
- block bootstrap (블록 10, B=500) — 프레임 자기상관 보정

---

## 3. Results

### 3.1 Topology adds information beyond geometry [본규모 표]
- 표: H별 P/PO/POR/POT R² + ΔR²(R|Y,O), ΔR²(TOPO|Y,O) + CI
- [멀티박스: 8/8 위치 방향 일관 — supplementary]

### 3.2 Horizon scaling [Figure a]
- ΔR²가 H와 함께 증가 → "위상은 느린 구조 정보"
- 멀티박스 + 본규모 동시 표시

### 3.3 CRU_turb [Figure b]
- 난류 ~0.5, granular 0.94-0.99, glass ~0 사이 배치
- ontic/epistemic 축 위의 위치

---

## 4. Discussion

### 4.1 Two kinds of underdetermination
```
ontic (FNE-I, granular):
  법칙(force balance)이 contact force를 genuinely 미결정
  → CRU 0.94-0.99, 즉각적 relational advantage

epistemic (this work, turbulence):
  ω는 u의 함수지만 low-order feature가 위상을 못 잡음
  → CRU ~0.5, 예측(forecast)에서만 relational advantage 발현
  → 지평 H와 함께 증가하는 서명
```
이 대비가 논문의 개념적 기여.

### 4.2 Connection to reconnection literature
- ΔR²(TOPO|Y,O) 결과가 Yao-Hussain "재연결→소산" 의
  예측력 버전 (성립 시) 또는 경계 (불성립 시)

### 4.3 Limitations
- 단일 데이터셋 (isotropic만; channel/MHD 확장 가능)
- 셀 기반 골격 ≠ vortex line tracking
- 예측 association이지 인과 아님
- TOPO(재배선 count)는 거친 proxy
- CRU_turb는 선형 예측기 기반 하한

### 4.4 Conceptual outlook
- FNE-I + FNE-II = "configuration underdetermines interaction"
  패턴의 2개 사례 (ontic + epistemic)
- 더 큰 프로그램(격자 게이지 center vortex, 양자중력 얽힘)으로의
  사다리 — 단 speculative로 명시

---

## 5. Conclusions
- 난류에서 vortex 위상이 장 기하 너머의 소산 예측 정보를 가짐
  (지평과 함께 증가, 8위치 일관)
- CRU_turb ~0.5: epistemic 미결정의 정량화
- FNE-I(ontic)과 대비되는 2번째 사례

---

## 확정 필요 (본 규모 결과 후)
- [ ] ΔR²(R|Y,O) 본규모 값 + CI (H1 성립 여부)
- [ ] ΔR²(TOPO|Y,O) 본규모 값 (재연결 신호 성립 여부)
- [ ] CRU_turb 본규모 정확값
- [ ] H1(CI>0) vs H0(CI 포함) → claim 강도 결정
