# Planning baselines

논문·방법별로 코드를 한 폴더에 모아 관리한다.

## 반복 실험: 이 파일 하나에서 설정

**`run/iterate_baseline.sh`**가 KnowNo·IntroPlan·Query-as-Action POMCP의 공통 진입점이다.
상단 설정 블록의 `BASELINES`, `DOMAINS`, `SCENES`, `ITERATIONS_PER_SCENE`, seed 로그,
방법별 파라미터와 로그 제어를 수정한다. 기본은 **3 × 2 × 5 × 40 = 1,200회**다.

```bash
# 02_BRL_POMDP_CODE에서
conda activate brl
bash run/iterate_baseline.sh --dry-run
bash run/iterate_baseline.sh
bash run/iterate_baseline.sh --baseline introplan
bash run/iterate_baseline.sh --baselines "knowno introplan" --resume
bash run/iterate_baseline.sh --dry-run --iter 2
```

- 세 방법은 동일한 Ours CSV의 domain·scene·iteration별 seed를 사용한다.
- KnowNo qhat은 `KNOWNO_TOMATO_QHAT` / `KNOWNO_WASTE_QHAT`, IntroPlan은
  `INTROPLAN_TOMATO_QHAT` / `INTROPLAN_WASTE_QHAT`으로 독립 설정한다.
- `ARCHIVE_EXISTING=true`는 선택한 방법·도메인·scene의 기존 결과만
  `LOG_ROOT/archive/baselines_<timestamp>/`로 옮긴다.
- `RESUME=true` / `--resume`는 보관 이동을 끄고 완료된 episode를 건너뛴다.
  새 완료 표식은 설정 해시를 확인한다. 기존 KnowNo/QaA의 빈 완료 표식도 인정하지만
  해당 과거 표식에는 설정 정보가 없어 변경 여부를 검증할 수 없다.
- 기존 IntroPlan 설정 해시 형식도 같은 선택·설정이면 인정한다.
- `--no-archive`는 기존 폴더에 쓴다. 같은 episode의 LLM 로그는 덮어쓴다.
- seed 누락·중복과 경로를 검사한 후 보관/실행을 시작한다. dry-run은 API 호출이나 파일 변경이 없다.
- 실패한 episode는 기록하고 다음 실행으로 진행한다. 실패가 있으면 종료 코드 1이다.
- 완료는 프로세스 정상 종료 기준이다. 과업 실패 결과도 정상 저장됐으면 완료로 취급한다.
- 기존 결과 폴더 `when_knowno_gpt4/`, `when_introplan/`, `query_as_action/`을 유지한다.
  배치의 seed·설정·상태는 `LOG_ROOT/baseline_seed_logs/`에 CSV/JSON으로 모은다.

기존 파일은 **설정이 없는 호환용 진입점**으로 정리했다:

| 기존 파일 | 공통 runner에 전달하는 기본 선택 |
| --- | --- |
| `iterate_knowno.sh` | KnowNo |
| `iterate_introplan.sh` | IntroPlan |
| `iterate_query_baselines.sh` | KnowNo + Query-as-Action POMCP |

설정은 이제 위 세 파일이 아닌 **`iterate_baseline.sh` 한 곳**에서 변경한다.
실행 엔진은 `scripts/baseline/iterate.py`이며 단일 episode는 기존 domain runner를 사용한다.

## KnowNo와 IntroPlan 비교

여기서 IntroPlan은 **conformal prediction(CP)을 결합한 버전**을 말한다.
원본에는 CP 없이 직접 예측하는 버전과 multi-label CP 버전도 있지만,
현재 BRL 어댑터는 설명 생성 후 A–E 행동 점수를 보정하는 CP 흐름을 사용한다.

| 비교 항목 | KnowNo | IntroPlan CP |
| --- | --- | --- |
| 공통 목표 | 행동 선택의 불확실성을 판단하고 필요하면 외부 도움 요청 | 동일 |
| 기본 흐름 | 후보 생성 → 행동 점수화 → CP 후보 집합 → 실행/질의 | 후보 생성 → 관련 추론 예시 검색 → 설명 생성 → 행동 점수화 → CP 후보 집합 → 실행/질의 |
| 언제 질문하는가 | BRL에서는 후보 집합이 singleton이 아니거나 fallback을 포함할 때 | 동일한 판단 규칙; 추론 후 점수가 달라지므로 질문 시점·빈도는 달라질 수 있음 |
| 무엇을 질문하는가 | 다음 행동 선택 | 다음 행동 선택; 상태 fact 질문으로 바뀌는 것은 아님 |
| qhat | 사용 | CP 버전도 사용 |
| 추가 자료 | 후보 생성·점수화 프롬프트와 calibration 자료 | 여기에 검색할 설명 예시인 knowledge base 추가 |
| BRL에서 decision당 기본 LLM 호출 | 후보 생성 1회 + 점수화 1회 | 후보 생성 1회 + 설명 생성 1회 + 점수화 1회 (재시도 제외) |
| 성공률·질의 수 | 현재 과업에서 실험으로 확인 | 개선을 목표로 하지만 BRL 성능 개선은 아직 검증되지 않음 |

### 현재 BRL 코드에서 공유하는 부분과 원본 대비 차이

- 두 방법은 KnowNo의 tomato/wastesorting 실행 루프, 환경 전이·관측,
  행동 파싱, CP 기반 실행/질의 규칙, action oracle을 공유한다.
  IntroPlan은 LLM 호출 함수를 주입해 점수화 전에 검색·설명을 추가한다.
- 두 shell runner의 환경 오류 확률은 동일하다. 같은 scene·seed로 비교할 수 있지만,
  정책의 행동 순서가 달라지면 난수 소비 순서도 달라질 수 있다.
- 공식 IntroPlan Mobile CP 노트북은 SBERT 임베딩의 내적으로 예시 3개를 검색하고,
  정답 행동을 제공해 LLM으로 작성한 설명 예시를 사용한다.
  현재 BRL 구현은 **단어 빈도 cosine 검색 + 수동 작성한 도메인별 4개 예시**를 사용한다.
  따라서 현재 구현은 BRL adaptation이며 원본 그대로의 재현으로 표기하지 않는다.
- BRL KnowNo 점수화 요청은 top-logprobs 5개, IntroPlan은 20개이다.
  A–E 중 반환되지 않는 토큰이 생길 수 있어 이 설정도 비교 시 기록한다.
- Waste는 같은 의미의 중복 후보 확률을 합산한다. 현재 재보정 도구는 후보 토큰별
  정답 점수를 계산하므로 **중복 없는 후보로 보정**하거나, 중복 후보를 허용하는
  정식 평가에서는 보정에도 동일한 합산을 적용해야 한다.

확인한 원본: [IntroPlan CP notebook](introplan/introplan_origin/IntroPlan_CP_Mobile.ipynb),
[CP utility](introplan/introplan_origin/cp_utils.py).
BRL 연결 세부사항은 [IntroPlan README](introplan/README.md)를 참고한다.

### qhat은 두 방법에서 같은 의미인가?

**같은 역할이다. 숫자는 방법별로 다시 산출해야 한다.**

1. 후보 점수는 `p(a) = softmax(logprob(a) / T)`로 계산한다.
2. 보정 샘플의 nonconformity score는 `s = 1 - max(p(정답 후보))`이다.
   여러 정답 후보가 있으면 그중 가장 높은 점수를 사용하며, 정답 토큰이 없으면 `s=1`로 처리한다.
3. 목표 coverage에 맞는 분위수로 `qhat`을 산출한다.
4. 배포 시 `p(a) >= 1 - qhat`인 후보를 prediction set에 포함한다.

점수가 고정되어 있다면 qhat이 커질수록 후보 집합은 넓어진다.
빈 집합·fallback도 질문을 유발하므로 실제 질문 수가 항상 단조 증가한다는 뜻은 아니다.
`target-success`는 보정의 목표 정답 포함률이며, BRL 다단계 과업의 최종 성공률 자체가 아니다.
IntroPlan은 설명 때문에 점수 분포가 바뀌므로 KnowNo의 qhat을 그대로 복사하지 않는다.

### qhat 외에 사용하는 값

아래 기본값은 **현재 BRL shell runner/보정 CLI 기준**이다.

| 값 | 현재 설정 | 의미 |
| --- | --- | --- |
| `qhat` | KnowNo: tomato 0.8404, waste 0.8704 / IntroPlan: tomato 0.98094750, waste 0.96153422 | 2026-09-15, gpt-4o, T=5, 목표 coverage 95%. 실행 시 `QHAT`으로 변경 |
| score `temperature` | 두 runner 5.0 | 후보 확률 정규화 온도. LLM 생성 sampling temperature와 별개. `SCORE_TEMPERATURE`로 변경 |
| `target-success` | 두 보정 CLI 0.8 | 목표 coverage. 원본 IntroPlan Mobile CP 노트북 예시는 0.85 |
| `epsilon` | 원본에서 `1 - target_success` | 별도의 추가 임계값이 아니라 목표 오류율의 다른 표기 |
| `q_level` | 표본 수와 target-success로 계산 | 분위수 위치. 사용자 독립 튜닝값이 아님 |
| `top-k` | IntroPlan 3 | 검색 예시 개수. `--top-k`로 변경 |
| knowledge base | IntroPlan `knowledge.json` | 검색할 설명 예시. `--knowledge`로 변경 |
| 설명 길이 제한 | IntroPlan 512 tokens | 추가 추론 호출의 현재 코드 고정값 |
| 모델·프롬프트·top-logprobs | 설정 파일 및 코드 | 점수 분포에 영향을 주므로 보정/평가에서 고정해야 함 |

IntroPlan에 belief entropy threshold나 fact 선택용 정보 이득 임계값이 추가된 것은 아니다.

### 재보정 코드 재사용

IntroPlan의 [compute_qhat.py](introplan/compute_qhat.py)는 다음 KnowNo 코드를 재사용한다.

- [scripts/calibration.py](knowno/scripts/calibration.py)의 `score_calibration_choices`:
  IntroPlan 설명 생성 함수를 주입하고 A–E logprob 추출을 공유한다.
- [compute_qhat.py](knowno/compute_qhat.py)의 `load_scored_json`, `add_scores`,
  `qhat_from_scores`, `write_csv`: 입력 로딩, 확률 변환, 정답 점수, 분위수 및 CSV 출력을 공유한다.
- KnowNo의 기본 호출 방식과 기존 분위수 결과는 유지한다.
  KnowNo의 기본은 `legacy_higher` (`np.quantile(..., method="higher")`),
  IntroPlan의 기존 방식은 `finite_sample` (정렬된 점수의 `ceil((n+1)*coverage)`번째 값)으로
  같은 함수의 명시적 옵션으로 구분했다. 표본이 부족해 순위가 n을 넘으면 IntroPlan은 1을 사용한다.
  **두 계산은 유한 표본에서 수치가 다를 수 있으므로 정식 비교 시 이 차이도 통제해야 한다.**
  두 보정 CLI 모두 `--quantile-method finite_sample` 또는 `--quantile-method legacy_higher`를
  지정할 수 있으므로, 동일한 방식을 명시해 비교할 수 있다.

IntroPlan은 정답 레이블을 추론 프롬프트에 넣지 않고, 설명 후 점수를 받은 다음 보정에만 사용한다.
실행 scene의 정답을 knowledge에 추가하지 말고 knowledge/calibration/evaluation 자료를 분리한다.

```bash
# 02_BRL_POMDP_CODE에서, 별도의 labeled BRL score prompts로 보정
python scripts/baseline/introplan/compute_qhat.py --domain tomato \
  --records /path/to/held_out_prompts.json \
  --output /tmp/introplan_calibration.json --output-csv /tmp/introplan_calibration.csv

# 저장한 IntroPlan 점수로 재계산 (LLM 호출 없음)
python scripts/baseline/introplan/compute_qhat.py --domain tomato \
  --scored-json /tmp/introplan_calibration.json \
  --target-success 0.85 --output /tmp/introplan_calibration_085.json
```

`--records`의 각 항목은 `score_prompt`와 `true_options`를 갖는다.
`--scored-json`은 **IntroPlan으로 이미 점수화한** 자료여야 하며,
모델·knowledge·top-k 변경 시에는 기존 점수를 재사용하지 말고 다시 점수화한다.
출력 qhat은 자동으로 실행 설정에 반영되지 않는다. 실행 시 `QHAT=<보정 결과>`로 전달한다.

| 폴더 | 방법 | 현재 상태 |
| --- | --- | --- |
| [knowno](knowno/README_KNOWNO_EXPERIMENTS.md) | KnowNo — Robots That Ask For Help | BRL tomato/wastesorting 실행 코드, calibration, 데이터, 원본 노트북 |
| [introplan](introplan/README.md) | IntroPlan — Introspective Planning | 원본은 introplan_origin/; BRL 도메인 실행 어댑터 포함 |
| [targeted_query_pomdp](targeted_query_pomdp/README.md) | Query-as-Action POMCP | 기존 BRL 구현 |

## 단일 실행 디버깅 GUI

[run_baseline_gui](run_baseline_gui/README.md)는 **baseline을 하나씩 실행하는 디버깅용 폴더**이다.
Domain → Scene → Baseline을 선택하고 실행 출력 확인·중지를 할 수 있다.

```bash
# 02_BRL_POMDP_CODE에서 실행
bash scripts/baseline/run_baseline_gui/run_baseline_gui.sh
```

KnowNo와 targeted_query_pomdp는 BRL scene 실행을 지원한다.
IntroPlan도 BRL scene 실행을 지원한다. 현재 기본값은 2026-09-15 고정 후보 데이터의 목표 coverage 95% 보정 결과이다.

## KnowNo 실행

IntroPlan 반복 실험은 `run/iterate_introplan.sh`를 사용한다.
공통 `run/iterate_baseline.sh`의 상단 전역 변수에서 도메인·scene·반복 수·paired seed·qhat·검색 설정·로그 제어를 지정한다.
기본 구성은 2개 도메인 × 5개 scene × 40회 = 400 episode다.
실행 전 `bash run/iterate_introplan.sh --dry-run`으로 확인할 수 있다.
자세한 옵션은 [IntroPlan 반복 실험 안내](introplan/README.md#반복-실험)를 참고한다.

`02_BRL_POMDP_CODE`에서:

```bash
bash run/run_knowno_baseline.sh
DOMAIN=wastesorting SCENE=03 bash run/run_knowno_baseline.sh
bash run/run_knowno_baseline_gui.sh
DRY_RUN=true bash run/run_knowno_baseline.sh
```

`run/run_query_baseline.sh`, `run/iterate_knowno.sh`,
`run/iterate_query_baselines.sh`도 같은 KnowNo runner로 연결된다.
기존 `scripts/baseline/` 직하의 KnowNo 파일은 모두 `scripts/baseline/knowno/`로 이동했다.
설정 파일은 프로젝트 루트의 `llm_setting.json`, 실험 로그는 기존
`experiments_logs/system_log/`를 사용한다.

## 실제 재보정 결과 (2026-09-15)

[보정 결과 및 전체 산출물](calibration/20260915_brl/README.md)을 저장했다. IntroPlan은 이번 95% 값을 사용하고, KnowNo는 기존 100개 보정의 0.8404/0.8704를 유지한다.
Tomato는 표본 18개로 95% 기준의 유한 표본 순위를 충족하지 못해 두 방법 모두 qhat=1이다.
80%·85% 값도 함께 저장했다. 이번 결과는 기존 고정 후보 데이터에 대한 보정이며, 독립 rollout 검증이나 최종 작업 성공률 보장이 아니다.

## 보정 산출물 보관

[calibration/README.md](calibration/README.md)에 기존 KnowNo와 이번 보정의 산출물·로그·출처를 모았다.
기존 기록은 `calibration/knowno_legacy/`, 이번 기록은 `calibration/20260915_brl/`에 보관한다.
현재 KnowNo 실행 기본값은 기존 보정값 tomato **0.8404**, wastesorting **0.8704**이다.

### 최신 IntroPlan 보정: 복구한 도메인별 100개

[전체 결과와 로그](calibration/introplan_recovered100_20260915/README.md).
과거 KnowNo와 동일한 입력·후보에 IntroPlan 설명을 적용하여 새로 점수화했다.
기본 qhat은 tomato **0.9809474992495626**, wastesorting **0.9615342162270937**이다.
gpt-4o, T=5, 목표 95%, 기존 KnowNo와 같은 legacy_higher를 사용했다.
앞서 보관한 18/23개 결과를 대체하며, KnowNo 값 0.8404/0.8704는 유지한다.
