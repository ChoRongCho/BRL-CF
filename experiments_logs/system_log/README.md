# System Experiment Log Layout

새 실험은 `10_TODO_20260727.md`의 E1–E4 구성에 따라 아래 위치에 저장한다.

```text
experiments_logs/system_log/
├── e1_threshold/
│   └── {domain}/scene_XX_step{max_step}/ours/oracle/tau_{value}/
├── e2_ablation/
│   └── {domain}/scene_XX_step{max_step}/{method}/oracle/
├── e3_baselines/
│   └── {domain}/scene_XX_step{max_step}/{method}/oracle/
└── e4_feedback/
    └── {domain}/scene_XX_step{max_step}/{method}/{vlm|human}/
```

Canonical method 이름:

```text
ours
ours-random-when
ours-random-what
knowno
active-search
```

저장 root는 `run/common.sh`의 `EXPERIMENT_LOG_ROOT`로 한 번에 변경할 수 있다.
기본값은 이 디렉터리다. 기존 `00_ours`, `04_answer_mode`, `05_active_search`,
`06_knowno`는 이전 구성에서 생성된 legacy 결과이며 이동하거나 덮어쓰지 않는다.

---

## Legacy experiment index

아래 내용은 기존 로그와 분석 코드의 의미를 확인하기 위한 기록이다.

`run/iterate_*.sh` 기준 실험 인덱스 초안이다. 세부 값은 각 스크립트의 상단 설정 블록 또는 환경변수로 조정한다.

도메인은 기본적으로 전체 도메인을 대상으로 한다. 아래 `도메인` 항목은 현재 스크립트에 실제로 들어가 있는 값 기준이며, 미완성 domain integration 때문에 일부 실험은 아직 전체 도메인이 들어가지 않았다.

## Reference Experiments

### 00. Ours reference

- 실행: `run/iterate_ours.sh`
- 단일 실행: `run/run_ours_reference.sh`
- 실제 runner: `main.py`
- 의미: 여러 비교 실험에서 재사용할 공통 ours 기준 결과
- 도메인: [`tomato`, `wastesorting`]
- 설정: [`scene=01..05`, `threshold=0.8`, `answer_type=oracle`, `max_step=50`, `iterations=40`, `seed=random`]
- metric: [`success_rate`, `average_step`, `average_question`, `query_probability_per_step`, `average_reward`, `elapsed_time`, `prediction_set_size_when_asked`]
- 저장위치: `experiments_logs/system_log/00_ours`
- seed log: `experiments_logs/system_log/00_ours/ours_seed_logs/iterate_ours_*.csv`
- 사용처: [`When-to-query policy comparison`, `Answer mode comparison`, `Active Search baseline`, `KnowNo baseline`]

## Comparison Experiments

## 01. Threshold comparison

- 실행: `run/iterate_th.sh`
- 실제 runner: `run/run_threshold_experiment.sh` -> `main.py`
- 의미: planner threshold 변화에 따른 system 성능 비교
- 도메인: [`tomato`]
- 축: [`threshold=0.9, 1.0`]
- 설정: [`scene=1..5`, `iterations=40`, `seed=random`]
- metric: [`success_rate`, `average_step`, `average_question`, `query_probability_per_step`, `average_reward`, `elapsed_time`]
- 저장위치: `experiments_logs/system_log/01_threshold`
- seed log: `experiments_logs/system_log/01_threshold/threshold_seed_logs/iterate_th_*.csv`
- 후처리: `experiments/system_eval/analysis_experiment.py`, `experiments/system_eval/read_csv_experiment.py`

## 02. When-to-query policy comparison

- 실행: `run/iterate_when.sh`
- 실제 runner: `run/when_experiments.sh` -> `when_main.py`
- 의미: query 시점 결정 strategy별 성능 비교
- 도메인: [`tomato`, `wastesorting`]
- 축: [`strategy=random`]
- 설정: [`scene=1..5`, `random_query_prob=tomato:0.48/wastesorting:0.38`, `max_step=50`, `iterations=40`, `seed=random`]
- 기준: `experiments_logs/system_log/00_ours`
- metric: [`success_rate`, `average_step`, `average_question`, `query_probability_per_step`, `average_reward`, `elapsed_time`, `prediction_set_size_when_asked`]
- 저장위치: `experiments_logs/system_log/02_when`
- 기타: 동일한 log directory가 이미 있으면 실행 전에 backup 디렉터리로 이동. `use_ours_reference=true`이면 `strategy=ours` 조건은 `00_ours`를 사용하고 iterator에서 건너뛴다.

## 03. Scale comparison

- 실행: `run/iterate_scale.sh`
- 실제 runner: `run/run_scale_experiments.sh` -> `scale_main.py`
- 의미: scene scale 증가에 따른 성능과 비용 변화 비교
- 도메인: [`wastesorting`]
- 축: [`scene=1`]
- 설정: [`threshold=0.8`, `iterations=40`, `seed=random`]
- 기준: 없음. scene 자체가 비교 조건이라 reference를 별도로 재사용하지 않는다.
- metric: [`success_rate`, `average_step_success_only`, `average_question`, `query_probability_per_step`, `elapsed_time`, `search_time_avg`, `update_time_avg`, `step_total_time_avg`, `tree_nodes_expanded_this_step_avg`, `belief_frontier_size_avg`, `belief_frontier_size_max`]
- scene별 설정: [`scene=1..5: max_step=50/max_belief_particles=800`, `scene=6..10: max_step=90/max_belief_particles=800`, `scene=11..15: max_step=125/max_belief_particles=8000`, `else: max_step=50/max_belief_particles=8000`]
- 저장위치: `experiments_logs/system_log/03_scale`
- seed log: `experiments_logs/system_log/03_scale/scale_seed_logs/iterate_scale_*.csv`

## 04. Answer mode comparison

- 실행: `run/iterate_answer_modes.sh`
- 실제 runner: `run/run_answer_mode_experiment.sh` -> [`main.py`, `scripts/baseline/knowno/runners/knowno_baseline_experiment.py`]
- 의미: 질문 답변 소스가 system 성능에 미치는 영향 비교. `runner=knowno` 설정 시 KnowNo baseline의 answer mode도 같은 실험 인덱스에서 실행한다.
- 도메인: [`tomato`, `wastesorting`]
- 축: [`answer_type=human-proxy`]
- 설정: [`scene=01..05`, `threshold=0.8`, `max_step=50`, `runner=ours|knowno`, `iterations=40`, `seed=random`, `noisy_oracle_accuracy=0.5, 0.7, 0.9`]
- 기준: `experiments_logs/system_log/00_ours`
- metric: [`success_rate`, `average_step`, `average_question`, `query_probability_per_step`, `average_reward`, `elapsed_time`, `prediction_set_size_when_asked`]
- 저장위치: `experiments_logs/system_log/04_answer_mode`
- seed log: `experiments_logs/system_log/04_answer_mode/answer_mode_seed_logs/iterate_answer_modes_*.csv`
- 기타: `use_ours_reference=true`이면 `answer_type=oracle`, `noisy_oracle_error_rate=0.0`, `runner=ours` 조건은 `00_ours`를 사용하고 iterator에서 건너뛴다.

## 05. Active Search baseline

- 실행: `run/iterate_active_search.sh`
- 실제 runner: `run/e3_baselines.sh` -> `scripts/baseline/attr_pomdp/main.py`
- 의미: Active Search baseline과 ours 비교를 위한 baseline runner 항목
- 도메인: [`wastesorting`, `tomato`]
- 축: [`baseline=active_search`]
- 설정: [`scene=all scene_*.yaml per domain`, `threshold=0.8`, `max_step=50`, `iterations=40`, `seed=random`]
- 기준: `experiments_logs/system_log/00_ours`
- metric: [`success_rate`, `average_step`, `average_question`, `query_probability_per_step`, `average_reward`, `elapsed_time`, `prediction_set_size_when_asked`]
- 대상 scene: [`wastesorting=01..25`, `tomato=01..25`]
- 저장위치: `experiments_logs/system_log/05_active_search`
- seed log: `experiments_logs/system_log/05_active_search/active_search_seed_logs/iterate_active_search_*.csv`
- 상태: Adapted Attr-POMDP runner 구현 완료

## 06. KnowNo baseline

- 실행: `run/iterate_knowno.sh`
- 실제 runner: `scripts/baseline/knowno/runners/knowno_baseline_experiment.py`
- 의미: KnowNo baseline의 model과 qhat calibration target별 성능 비교
- 도메인: [`wastesorting`, `tomato`]
- 축: [`model=gpt-4o, gpt-3.5-turbo`, `qhat_target=95, 85, 75, raw98`]
- 설정: [`scene=1..5`, `prompt_version=v2`, `temperature=5.0`, `max_steps=50`, `auto_answer=true`, `answer_type=oracle`, `noisy_oracle_error_rate=0.1`, `iterations=10`, `seed=random`]
- 기준: `experiments_logs/system_log/00_ours`
- metric: [`success_rate`, `average_step`, `average_question`, `query_probability_per_step`, `elapsed_time`, `prediction_set_size_when_asked`, `token_overall`]
- qhat: [`raw98=0.98`, `tomato/gpt-3.5-turbo: 95=0.9243, 85=0.9082, 75=0.8938`, `tomato/gpt-4o: 95=0.8404, 85=0.7779, 75=0.7322`, `wastesorting/gpt-3.5-turbo: 95=0.9028, 85=0.8851, 75=0.8512`, `wastesorting/gpt-4o: 95=0.8704, 85=0.7369, 75=0.7084`]
- 저장위치: `experiments_logs/system_log/06_knowno`
- seed log: `experiments_logs/system_log/06_knowno/knowno_seed_logs/iterate_knowno_*.csv`
- seed 처리: 고정 seed 입력 시 `seed + global_index - 1`을 runner에 전달
