# IntroPlan — BRL adapter

원본 논문: **Introspective Planning: Aligning Robots’ Uncertainty with Inherent Task Ambiguity**, NeurIPS 2024.

## 구조

- `introplan_origin/`: [공식 저장소](https://github.com/kevinliang888/IntroPlan)의 원본 코드·노트북·데이터·MIT 라이선스. commit `443075e14bdda0dc97b67fed1f7bae0a14918d5f` (2026-09-15 반입). 이전 `upstream/`에서 이동했다.
- `run_experiment.py`, `run.sh`: BRL scene 로딩 및 단일 실험 실행.
- `policy.py`: 관련 예시 검색 → 설명 생성 → 설명을 반영한 행동 점수화.
- `knowledge.json`: 수동 작성한 BRL 도메인별 예시. 현재 평가 scene의 true state나 oracle 응답을 검색에 넣지 않는다.
- `compute_qhat.py`: 별도 calibration prompt를 이용한 임계값 계산.

## BRL 실행

### 반복 실험

`run/iterate_baseline.sh` 상단의 전역 설정 블록에서 `BASELINES=(introplan)`을 지정한다.
기존 `run/iterate_introplan.sh`는 이 공통 runner를 호출하는 호환용 진입점이다.
기본값은 `DOMAINS=(tomato wastesorting)`, `SCENES=(1 2 3 4 5)`,
`ITERATIONS_PER_SCENE=40`, `MAX_STEPS=50`으로 총 400 episode다.
기존 Ours When+What seed CSV의 `ours` 행을 domain·scene·iteration별로 매칭한다.

```bash
# 02_BRL_POMDP_CODE에서
conda activate brl
bash run/iterate_introplan.sh --dry-run
bash run/iterate_introplan.sh
bash run/iterate_introplan.sh --resume
bash run/iterate_introplan.sh --dry-run --iter 2
```

- 설정: `PAIRED_SEED_LOG`, `PROMPT_VERSION`, `SCORE_TEMPERATURE`,
  `INTROPLAN_TOMATO_QHAT`, `INTROPLAN_WASTE_QHAT`, `TOP_K`, `KNOWLEDGE_FILE`, `LOG_ROOT`,
  `ARCHIVE_EXISTING`, `RESUME`, `DRY_RUN`.
- 모든 seed 매칭·중복·scene 경로를 검사한 뒤에 기존 결과 보관이나 실행을 시작한다.
- `ARCHIVE_EXISTING=true`: 선택한 domain/scene의 `when_introplan/`만
  `LOG_ROOT/archive/baselines_<timestamp>/`로 옮긴다.
- `RESUME=true` 또는 `--resume`: 보관 이동을 끄고 동일 설정의 완료 episode를 건너뛴다.
  완료는 프로세스 정상 종료 기준이며, 과업 실패 결과도 정상 기록됐다면 완료로 취급한다.
- `--no-archive`: 기존 폴더에서 실행한다. 동일 episode 파일은 덮어쓴다.
- 로그: `LOG_ROOT/<domain>/scene_<NN>_step<N>/when_introplan/` 아래 실행 로그,
  `.introplan.jsonl`, `.console.log`, `.completed/`.
- seed·설정·완료/실패 상태: `LOG_ROOT/baseline_seed_logs/`의 CSV와 JSON.
- 배치 엔진은 `scripts/baseline/iterate.py`. 실패한 episode를 기록하고 다음 episode로 진행하며,
  실패가 하나라도 있으면 배치 종료 코드는 1이다.
- dry-run은 전체 계획을 검증하고 도메인별 대표 명령을 출력하며 API 호출과 파일 변경을 하지 않는다.

`02_BRL_POMDP_CODE`에서:

```bash
conda activate brl
DOMAIN=tomato SCENE=01 SEED=42 bash scripts/baseline/introplan/run.sh
DOMAIN=wastesorting SCENE=03 SEED=42 bash scripts/baseline/introplan/run.sh
DRY_RUN=true bash scripts/baseline/introplan/run.sh
bash scripts/baseline/run_baseline_gui/run_baseline_gui.sh
```

GUI에서 IntroPlan을 선택하고 **한 번 실행**을 누르면 선택한 BRL domain/scene을 실행한다.
Jupyter나 원본 노트북을 거치지 않는다. LLM 설정은 프로젝트 루트 `llm_setting.json`을 사용한다.
원본의 구형 OpenAI 패키지를 설치하지 않고 기존 BRL LLM 연결을 재사용한다.

## 연결 방식 및 차이

KnowNo의 도메인 실행 루프에 LLM 호출 함수를 주입한다. 환경 전이, 관측, 행동 파싱,
CP 후보 선택 및 action oracle은 공유하며, KnowNo 자체의 기본 실행은 기존 함수를 사용한다.
`run.sh`의 환경 오류 확률도 KnowNo runner와 동일하게 설정했다.

IntroPlan은 행동 후보를 생성한 뒤 관련 예시 3개를 검색하고 추가 LLM 호출로 설명을 만든다.
설명에 `Prediction:`이 있으면 그 이후를 제외하고 점수화 프롬프트에 넣는다.
추가 추론 호출 토큰은 전체 사용량에 포함된다.

이 구현은 **BRL용 adaptation**이며 논문의 성능 재현을 완료한 구현은 아니다.
원본의 sentence-embedding 검색 대신 단어 빈도 cosine 검색을 사용하고,
원본의 LLM 생성 knowledge 대신 수동 BRL 예시를 사용한다.
Knowledge/calibration/evaluation 분리와 모델·프롬프트·온도 고정이 필요하다.

## 임계값과 로그

기본 qhat은 **복구한 도메인별 100개 자료 보정의 95% 값**으로 tomato=0.9809474992495626, wastesorting=0.9615342162270937이다.
기존 KnowNo와 같은 legacy_higher 방식으로 산출했다. [보정 기록](../calibration/introplan_recovered100_20260915/README.md)을 참고한다. 정식 비교에서 KnowNo의 임계값을 복사하지 말고
IntroPlan 추론을 포함한 점수로 다시 보정한다.

Calibration 입력은 JSON 배열이며 각 항목은 `score_prompt`(관측과 A–E 행동 후보를 포함한
실제 배포 형식의 프롬프트)와 `true_options`(예: `["A"]`)를 가진다.
동일 의미의 중복 선택지는 제거한 후보로 보정한다. 정답 레이블은 설명 생성 프롬프트에 전달하지 않는다.

```bash
python scripts/baseline/introplan/compute_qhat.py --domain tomato \
  --records /path/to/held_out_prompts.json --output /tmp/introplan_qhat.json
QHAT=0.9 bash scripts/baseline/introplan/run.sh  # 보정 결과의 qhat 값으로 변경
```

보정과 실행에서 동일 knowledge, top-k, 모델, temperature를 사용한다.
보정 계산은 KnowNo의 공통 함수를 재사용한다. 저장한 출력 JSON은 `--scored-json`으로
다시 입력해 LLM 호출 없이 재계산할 수 있다. 두 방법의 보정 CLI에서
`--quantile-method finite_sample`을 동일하게 지정하면 분위수 계산 방식도 맞출 수 있다.
`--knowledge`, `--top-k`는 `run.sh` 뒤 인자로 전달할 수 있다.
Calibration 도구는 후보별 정답 포함 기준을 계산하며, 배포 분포에서의 성공률 보장은 별도 검증 대상이다.

로그는 `experiments_logs/system_log/<domain>/scene_<scene>_step<N>/when_introplan/`에 저장한다.
일반 실행 로그와 함께 `.introplan.jsonl`에 검색 예시 ID, 설명, 점수화 프롬프트, 호출별 토큰을 기록한다.
