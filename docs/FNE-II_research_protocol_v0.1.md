# FNE-II: Vortex Reconnection as Realized Relational Structure in Turbulence
## Research Protocol v0.1 (Design Document)

**Author:** Kyeongmin Kim (Independent Researcher)
**Date:** 2026-06-11
**Status:** Design phase — pre-pilot
**Predecessor:** Kim (2026), "Geometry Poorly Predicts Contact Realization in Granular Force-Network Ensembles" (Zenodo 10.5281/zenodo.20484747)

---

## 0. One-line summary

FNE 논문의 ΔO/ΔR 프레임워크를 난류에 재적용하되, R을 일반 통계 feature가 아니라
**vortex 위상 사건(재연결)** 으로 재정의하여, 논문 §4.1의 turbulence null result가
"관계가 무관해서"인지 "관계 변수를 잘못 골라서"인지 판별한다.

---

## 1. Hypotheses

```
H0: 난류에서 국소 소산 변화(ΔY)는 국소 장 기하 변화(ΔO)로
    충분히 예측된다. vortex 위상 사건은 추가 예측력이 없다.
    (= FNE 논문의 turbulence null result가 본질적)

H1: ΔY는 ΔO보다 vortex 위상 사건(ΔR_topo)으로 더 잘 예측된다.
    특히 재연결 사건 근방에서 ΔO→ΔY 결합이 끊어진다.
    (= null result는 feature 선택의 문제였음)
```

판별 기준 (사전 등록):
- ΔR_topo→ΔY R²가 ΔO→ΔY R²를 bootstrap 95% CI 비겹침으로 상회하면 H1
- 두 R²가 CI 겹침이면 결론 유보, 둘 다 ≈0이면 H0 강화

---

## 2. Variable Operationalization

분석 단위: **Eulerian coarse-grained cell** (입자의 대응물)
- 도메인을 균일 격자 셀로 분할 (pilot: 8³ DNS 포인트/셀)
- FNE의 "입자 i" ↔ 난류의 "셀 i"
- FNE의 "사이클" ↔ "저장 프레임"

### Y (target) — 셀별 소산율
```
Y_i = <2ν S_jk S_jk>_cell_i        (S = strain-rate tensor)
```
- JHTDB getData로 속도 구배 직접 수신 → S 계산
- FNE의 입자별 총 접촉력의 대응물 (국소 에너지 "소비")

### O (geometry features, k=5) — 셀별 장 기하
1. 속도 크기 |u| (셀 평균)
2. 속도 구배 노름 |∇u|
3. 스트레인 크기 |S|
4. 압력 (셀 평균)
5. 국소 에너지 밀도 ½|u|²

원칙: FNE와 동일 — 좌표/장 값 기반의 low-order local descriptor.
와도(vorticity) 자체는 O에서 제외 (R 쪽 정보 누설 방지 — FNE에서
force-weighted feature를 R에서 제외한 것의 대칭).

### R (relational features, k=5) — vortex 골격 그래프
**골격 구성:**
- 셀별 swirling strength λ_ci (속도구배텐서 고유값 허수부) 계산
- λ_ci > 임계값(λ_rms 기반)인 셀 = vortex 셀 (노드)
- 인접 vortex 셀 간 vorticity 방향 정렬(|cosθ|>0.7)이면 엣지
  → "vortex 골격 그래프" G_t (프레임마다)

**R features (FNE와 1:1 대응):**
1. degree (vortex 골격 내 연결 수) ↔ FNE degree
2. clustering coefficient ↔ FNE clustering
3. 2-hop connectivity ↔ FNE 2-hop
4. betweenness centrality ↔ FNE betweenness
5. local coordination variance ↔ FNE coord. variance

### ΔR_topo (위상 사건) — 핵심 신규 변수
```
ΔR_topo(i, t) = 셀 i 주변(반경 r_n셀)에서
                프레임 t→t+1 사이 엣지 재배선(rewiring) 수
              = |E_t △ E_{t+1}| restricted to neighborhood(i)
```
- 재연결 proxy: 골격 그래프의 엣지가 끊기고 새 엣지가 생기는 사건
- 보조 지표: helicity density h = u·ω 음수 패치의 출몰
  (Yao-Hussain 계열 문헌에서 재연결 위치 표지로 확립)

### CRU 대응물
```
CRU_turb ≡ 1 − R²(O → degree_skeleton)
```
- FNE와 동일한 정의 구조. 장 기하가 vortex 골격의 국소 연결성을
  얼마나 결정하는가.
- 주의: FNE와 달리 난류에서 ω는 u의 미분이므로 O가 충분히 풍부하면
  R이 원리상 유도 가능 (epistemic 미결정성). CRU_turb가 낮게 나오는
  것 자체가 ontic/epistemic 구분 가설의 검증 데이터가 됨.

---

## 3. Data

| 항목 | 값 |
|---|---|
| 데이터셋 | JHTDB isotropic1024coarse |
| 프레임 | 5,028개, t = 0–10.056, δt = 0.002 |
| Re_λ | ~433 |
| 시간 분해능 | δt ≈ 0.002 << τ_η ≈ 0.045 → 프레임 간 재연결 추적 가능 (※ README에서 τ_η 재확인 필요) |
| 접근 | SciServer 무료 계정 + pyJHTDB/giverny (사전 설치) 또는 hdf5 cutout |

**Pilot 규모:**
- 공간: 128³ DNS 포인트 cutout 1개 (전체 1024³의 부분영역)
- 셀: 8³ 포인트/셀 → 16³ = 4,096 셀 (FNE 824P의 ~5배 규모)
- 시간: 연속 100 프레임 (≈ 4.4 τ_η)
- 데이터량 추정: 128³ × 4변수 × 100프레임 × 4byte ≈ 3.4 GB (cutout으로 처리 가능)

---

## 4. Analysis Pipeline (FNE 코드 재사용 맵)

```
단계                          FNE 대응 코드            신규 작업
────────────────────────────────────────────────────────────
1. cutout 다운로드             —                       pyJHTDB 스크립트
2. 셀 coarse-graining          —                       numpy reshape
3. λ_ci, S, h 계산             —                       속도구배 → 고유값
4. 골격 그래프 구성             contact network 로드     임계값+정렬 엣지
5. O/R feature 계산            그대로 재사용             —
6. ΔO→ΔY vs ΔR→ΔY            reproduce_results.py     ΔR_topo 추가
7. CRU_turb                   CRU 계산부 그대로         —
8. bootstrap CI               그대로 재사용             —
```

---

## 5. 사전 등록할 민감도 검사 (hostile review 선제 대응)

FNE 논문에서 배운 공격 지점을 미리 설계에 포함:

1. **임계값 의존성:** λ_ci 임계값 3종 (1.0, 1.5, 2.0 × λ_rms)에서
   결과 방향 불변 확인
2. **셀 크기 의존성:** 4³, 8³, 16³ 포인트/셀 3종
3. **counting artifact 대응물:** ΔR_topo가 단순히 "vortex 셀 개수
   변화"를 세는 것 아닌지 — 셀 수 고정 조건부 분석
4. **시간 스케일:** 프레임 간격 1, 5, 10배에서 ΔR_topo→ΔY 비교
   (재연결의 시간 스케일 식별 — FNE §5.4 speculation의 직접 검증)
5. **GBM robustness:** 선형 + GBM 양쪽 (FNE Table S1 대응)

---

## 6. Decision Gates

```
Gate 1 (pilot 후):
  골격 그래프가 물리적으로 말이 되는가?
  (vortex worm 구조 재현, 문헌의 길이 스케일과 일치)
  → 실패 시: 골격 정의 재설계, 통과 시 Gate 2

Gate 2 (본분석 후):
  ┌ H1 (ΔR_topo > ΔO, CI 비겹침)
  │   → FNE-II 논문화. "관계 우위 패턴의 2번째 사례,
  │      epistemic 시스템에서도 성립"
  │   → FNE 논문 §4.1 null을 명시적으로 뒤집는 후속 연구
  │
  ├ H0 (둘 다 ≈0 또는 ΔO 우위)
  │   → ontic/epistemic 구분 논문(방향 B)의 핵심 증거로 흡수.
  │      "ontic 미결정성(FNE)에서만 관계 우위,
  │       epistemic(난류)에서는 불성립" — 이것도 출판 가치 있음
  │
  └ 애매 (CI 겹침)
      → 규모 확대 또는 보류. observe 모드 유지.
```

**중요:** 어느 쪽이 나와도 정보가 된다. H0여도 "FNE 결과의 경계 조건"
이라는 scope delimiter 논문이 가능 (FNE 논문에서 null result를
자산으로 쓴 전략의 반복).

---

## 7. Timeline (observe-and-optionality 모드 내)

```
Phase 0 (1주, 여유 시간):  SciServer 계정, cutout 1개 수신 테스트
Phase 1 (3–4주):           pilot — 골격 추출 + Gate 1
Phase 2 (4–6주):           본분석 + 민감도 검사
Phase 3 (조건부):          논문화 여부 결정 (Gate 2)
```

RPM 업무와 병행 가능한 주말 프로젝트 규모로 설계. 마감 없음.

---

## 8. Known Risks (정직한 기록)

1. **골격 그래프 ≠ vortex line topology.** 셀 기반 골격은 진짜
   vortex line tracking의 거친 근사. 재연결을 "엣지 재배선"으로
   proxy하는 것의 타당성이 Gate 1의 핵심.
2. **ω = curl(u) 문제.** 난류에서 R은 O로부터 원리상 유도 가능
   (FNE와 결정적 차이). CRU_turb 해석에 이 점 명시 필수.
   단, "유도 가능"과 "low-order feature로 예측 가능"은 다름 —
   이 간극 자체가 epistemic 미결정성의 측정값.
3. **단일 데이터셋.** isotropic1024 하나로 시작. 성립 시
   channel flow 등으로 확장 (JHTDB 내 교차검증 가능 — FNE의
   same-lab 문제보다 양호).
4. **선행연구 중복 가능성.** "reconnection이 dissipation을
   주도한다"는 정성적 주장은 Yao-Hussain 등에 이미 있음.
   본 연구의 차별점은 FNE와 동일 — 예측력 프레임(ΔR vs ΔO R²
   비교)으로 정량화하는 것. Phase 0에서 선행연구 정밀 조사 필수.
```
