# KnowNo Baseline Experiments

이 문서는 `scripts/baseline`의 KnowNo baseline planner를 tomato harvesting과 waste sorting 도메인에서 실험하는 방법을 정리한다.

## Scope

도메인별 multi-step planner는 다음 파일을 사용한다.

- Unified runner: `knowno_baseline_experiment.py`
- Shell runner: `run/run_knowno_baseline.sh`
- Tomato harvesting planner: `scripts/knowno_multistep_tomato.py`
- Waste sorting planner: `scripts/knowno_multistep_wastesorting.py`
- Waste sorting wrapper: `scripts/knowno_multistep_waste.py`

`knowno.py`는 single-step/demo entrypoint에 가깝다. 도메인별 multi-step 실험은 `knowno_baseline_experiment.py` 또는 `run/run_knowno_baseline.sh`로 실행한다. 개별 planner 파일은 `scripts/baseline/scripts/` 아래에 보관한다.

## Important Scene Note

`knowno_baseline_experiment.py`는 `scripts/domain/tomato/scene_0N.yaml` 또는 `scripts/domain/wastesorting/scene_0N.yaml`을 읽고, `true_init`을 기존 planner 인자로 변환한다.

기존 planner는 초기 hidden state를 command-line 인자로 받는다.

- Tomato: `--labels`, `--locations`
- Waste sorting: `--labels`

따라서 직접 planner를 실행할 때만 아래 표의 값을 인자로 넣으면 된다. 권장 실행 방식은 `knowno_baseline_experiment.py` 또는 `run/run_knowno_baseline.sh`이다.

## Setup

저장소 루트에서 실행한다.

```bash
cd /home/changmin/PyProject/00_BRL-CF
```

필요 패키지:

```bash
pip install numpy openai
```

LLM 설정은 `scripts/baseline/llm_setting.json` 또는 환경 변수로 지정한다.

```json
{
  "openai_api_key": "your-api-key",
  "model": "gpt-3.5-turbo-instruct"
}
```

`gpt-3.5-turbo-instruct`는 completions 모델이다. baseline의 `scripts/llm.py`는 이 모델을 `completions.create()` 경로로 호출한다.

## Recommended Runner

도메인과 scene은 shell script에서 수정한다.

```bash
# run/run_knowno_baseline.sh
DOMAIN="tomato"        # tomato 또는 wastesorting
SCENE="01"             # 01, 02, 03, 04, 05
```

실행:

```bash
bash run/run_knowno_baseline.sh
```

Python runner를 직접 실행할 수도 있다.

```bash
python3 scripts/baseline/knowno_baseline_experiment.py --domain tomato --scene 01
python3 scripts/baseline/knowno_baseline_experiment.py --domain wastesorting --scene 03
```

## Tomato Experiments

기본 실행:

```bash
python3 scripts/baseline/knowno_baseline_experiment.py --domain tomato --scene 01
```

주요 인자:

- `--labels`: tomato별 true property. 값은 `ripe`, `unripe`, `rotten`.
- `--locations`: tomato별 true location. 값은 `stem_01`, `stem_02`.
- `--detect-success-prob`: detect 성공 확률.
- `--detect-label-error-prob`: detect label error 확률.
- `--scan-success-prob`: scan 성공 확률.
- `--scan-label-error-prob`: scan label error 확률.
- `--qhat`: KnowNo prediction set threshold.
- `--seed`: random seed.
- `--verbose`: prompt와 상세 로그를 함께 출력.
- `--log-file`: 로그 저장 경로. 비우면 기본 log 폴더를 사용한다.

### Tomato Scene Commands

`scripts/domain/tomato/scene_01.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --labels "tomato1:ripe,tomato2:rotten,tomato3:ripe,tomato4:unripe" \
  --locations "tomato1:stem_01,tomato2:stem_01,tomato3:stem_02,tomato4:stem_02"
```

`scripts/domain/tomato/scene_02.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --labels "tomato1:ripe,tomato2:unripe,tomato3:rotten,tomato4:ripe" \
  --locations "tomato1:stem_01,tomato2:stem_01,tomato3:stem_02,tomato4:stem_02"
```

`scripts/domain/tomato/scene_03.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --labels "tomato1:ripe,tomato2:unripe,tomato3:ripe,tomato4:rotten" \
  --locations "tomato1:stem_01,tomato2:stem_01,tomato3:stem_02,tomato4:stem_02"
```

`scripts/domain/tomato/scene_04.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --labels "tomato1:rotten,tomato2:ripe,tomato3:ripe,tomato4:unripe" \
  --locations "tomato1:stem_01,tomato2:stem_01,tomato3:stem_02,tomato4:stem_02"
```

`scripts/domain/tomato/scene_05.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --labels "tomato1:rotten,tomato2:unripe,tomato3:ripe,tomato4:ripe" \
  --locations "tomato1:stem_01,tomato2:stem_01,tomato3:stem_02,tomato4:stem_02"
```

Example with deterministic seed and verbose logging:

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --seed 1 \
  --verbose \
  --labels "tomato1:ripe,tomato2:rotten,tomato3:ripe,tomato4:unripe" \
  --locations "tomato1:stem_01,tomato2:stem_01,tomato3:stem_02,tomato4:stem_02"
```

## Waste Sorting Experiments

기본 실행:

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py
```

`knowno_multistep_waste.py`는 wrapper이므로 아래 명령과 동일한 planner를 실행한다.

```bash
python3 scripts/baseline/scripts/knowno_multistep_waste.py
```

주요 인자:

- `--scene-objects`: waste object list. 기본값은 `waste1, waste2, waste3, waste4`.
- `--labels`: waste별 true category. 값은 `general`, `plastic`, `paper`, `can`.
- `--detect-success-prob`: detect 성공 확률.
- `--detect-label-error-prob`: detect label error 확률.
- `--qhat`: KnowNo prediction set threshold.
- `--seed`: random seed.
- `--verbose`: prompt와 상세 로그를 함께 출력.
- `--log-file`: 로그 저장 경로. 비우면 기본 log 폴더를 사용한다.

### Waste Sorting Scene Commands

`scripts/domain/wastesorting/scene_01.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --labels "waste1:paper,waste2:general,waste3:plastic,waste4:can"
```

`scripts/domain/wastesorting/scene_02.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --labels "waste1:can,waste2:general,waste3:paper,waste4:can"
```

`scripts/domain/wastesorting/scene_03.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --labels "waste1:plastic,waste2:plastic,waste3:paper,waste4:general"
```

`scripts/domain/wastesorting/scene_04.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --labels "waste1:paper,waste2:paper,waste3:can,waste4:general"
```

`scripts/domain/wastesorting/scene_05.yaml`:

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --labels "waste1:paper,waste2:paper,waste3:can,waste4:can"
```

Example with deterministic seed and verbose logging:

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --seed 1 \
  --verbose \
  --labels "waste1:paper,waste2:general,waste3:plastic,waste4:can"
```

## Calibration

두 multi-step planner 모두 calibration 옵션을 제공한다.

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --run-calibration \
  --num-calibration 20 \
  --num-test 10 \
  --target-success 0.8
```

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --run-calibration \
  --num-calibration 20 \
  --num-test 10 \
  --target-success 0.8
```

Calibration dataset template을 만들려면 다음 옵션을 사용한다.

```bash
python3 scripts/baseline/scripts/knowno_multistep_wastesorting.py \
  --write-calibration-template /tmp/waste_knowno_calibration.txt
```

```bash
python3 scripts/baseline/scripts/knowno_multistep_tomato.py \
  --write-calibration-template /tmp/tomato_knowno_calibration.txt
```

## Computing qhat

`qhat=0.928`은 실행 중 바뀌는 값이 아니라, calibration set에서 한 번 구해 둔 conformal threshold이다. 각 calibration sample에서 정답 option의 확률을 구하고, nonconformity score를 다음처럼 계산한다.

```text
score = 1 - max probability assigned to a correct option
```

그 다음 score들의 quantile을 사용한다.

```text
q_level = ceil((n + 1) * target_success) / n
qhat = quantile(scores, q_level, method="higher")
```

이 저장소에서는 계산용 CLI를 제공한다.

```bash
python3 scripts/baseline/compute_qhat.py \
  --domain tomato \
  --calibration-file scripts/baseline/data/tomato-mc-gen-prompt.txt \
  --num-calibration 20 \
  --target-success 0.8 \
  --score-with-llm \
  --output-csv experiments_logs/system_log/tomato/qhat_rows.csv \
  --output-json experiments_logs/system_log/tomato/qhat_records.json
```

이미 LLM scoring 결과를 저장한 JSON이 있으면 API 호출 없이 계산할 수 있다.

```bash
python3 scripts/baseline/compute_qhat.py \
  --domain tomato \
  --scored-json experiments_logs/system_log/tomato/qhat_records.json \
  --target-success 0.8
```

현재 포함된 tomato/waste calibration text files는 template 수준이라 0.928을 재현할 만큼 충분한 calibration sample을 담고 있지 않다. 0.928을 재현하려면 해당 값을 만들 때 사용한 calibration records와 LLM top-token logprobs가 필요하다.

## Interpreting Output

각 step에서 planner는 다음 정보를 출력한다.

- current state summary
- generated multiple-choice action options
- option token log probabilities
- softmax scores
- prediction set
- selected/executed action

Prediction set이 singleton이면 baseline은 해당 option을 실행한다. Prediction set이 여러 개이거나 fallback option이 포함되면 `Help needed` 상태가 되고, 터미널에서 `A/B/C/D/E` 중 하나를 입력해야 한다.

## Current Limitation

`knowno_baseline_experiment.py`는 scene YAML의 `true_init`만 읽는다. 기존 planner의 내부 실행 로직은 그대로 두고, scene 정보를 `--labels`, `--locations` 인자로 변환해 전달한다.

즉, domain YAML의 `goal`, `facts`, action schema 전체를 baseline planner가 직접 사용하는 구조는 아직 아니다.
