# Baseline

이 폴더는 KnowNo 스타일의 LLM 기반 로봇 플래닝 baseline을 노트북 없이 실행할 수 있도록 정리한 코드입니다. LLM이 가능한 행동 후보를 multiple choice로 생성하고, 각 선택지의 로그 확률을 이용해 prediction set을 만든 뒤, 확신이 부족하면 도움을 요청하는 흐름을 구현합니다.

## 구성

- `knowno.py`: baseline 실행 엔트리포인트입니다.
- `scripts/prompt.py`: demo, mobile, tabletop, waste sorting용 few-shot prompt와 선택지 처리 유틸리티가 있습니다.
- `scripts/llm.py`: OpenAI API 호출, 설정 로딩, retry/timeout 처리를 담당합니다.
- `scripts/env.py`: mobile calibration/test 데이터 로딩 및 필요 시 다운로드를 담당합니다.
- `scripts/sim.py`: tabletop 모드에서 사용하는 PyBullet 시뮬레이션 코드입니다.
- `data/`: mobile manipulation calibration/test에 쓰는 prompt 및 task 데이터입니다.
- `outputs/`: tabletop 실행 결과 이미지/비디오가 저장되는 위치입니다.
- `ur5e/`, `robotiq_2f_85/`, `bowl/`: tabletop PyBullet 시뮬레이션에 필요한 에셋입니다.

## 지원 모드

`knowno.py`는 다음 모드를 지원합니다.

- `demo`: office kitchen 예제에서 행동 후보를 생성하고 prediction set을 출력합니다.
- `mobile`: mobile manipulation 예제를 실행합니다. `--run-calibration`을 주면 calibration/test 데이터로 `qhat`을 다시 계산합니다.
- `tabletop`: tabletop rearrangement prompt를 실행하고, 선택된 행동을 PyBullet 시뮬레이션으로 수행합니다.
- `wastesorting`: waste sorting station 예제에서 쓰레기 분류 행동 후보를 생성하고 선택합니다.
- `bimani`: 현재 placeholder이며 실제 동작은 구현되어 있지 않습니다.

## 설치

Python 3 환경에서 실행합니다. 최소 의존성은 다음과 같습니다.

```bash
pip install numpy openai pillow pybullet moviepy gdown
```

일부 의존성은 코드 실행 중 없으면 자동 설치를 시도합니다. 네트워크가 제한된 환경에서는 미리 설치하는 편이 안전합니다.

## OpenAI 설정

API 키는 환경 변수 또는 `baseline/llm_setting.json`으로 설정할 수 있습니다.

환경 변수 사용:

```bash
export OPENAI_API_KEY="your-api-key"
```

설정 파일 예시:

```json
{
  "openai_api_key": "your-api-key",
  "model": "gpt-4"
}
```

`--settings`로 다른 설정 파일 경로를 넘길 수 있습니다. API 키가 들어간 설정 파일은 저장소에 커밋하지 않는 것이 좋습니다.

## 실행

기본 실행:

```bash
python3 baseline/knowno.py
```

모드 지정:

```bash
python3 baseline/knowno.py --mode wastesorting
python3 baseline/knowno.py --mode demo
python3 baseline/knowno.py --mode mobile
python3 baseline/knowno.py --mode tabletop
```

주요 옵션:

```bash
python3 baseline/knowno.py \
  --mode wastesorting \
  --instruction "Discard all waste" \
  --scene-objects "news paper, empty coke can, toilet paper" \
  --qhat 0.928 \
  --user-option D
```

`--instruction`과 `--scene-objects`를 생략하면 `knowno.py`에 정의된 기본값을 사용합니다.

## 모드별 동작

`demo`, `mobile`, `wastesorting`은 공통적으로 다음 순서로 동작합니다.

1. 현재 scene과 instruction으로 multiple-choice 생성 prompt를 만듭니다.
2. LLM이 행동 후보를 생성합니다.
3. 후보를 `A`-`E` 형식으로 정규화하고 `an option not listed here` 선택지를 추가합니다.
4. LLM 로그 확률로 선택지를 scoring합니다.
5. conformal prediction threshold `qhat`으로 prediction set을 만듭니다.
6. prediction set이 하나가 아니거나 추가 선택지가 포함되면 도움 필요로 판단합니다.

`tabletop`은 선택된 행동을 파싱해 PyBullet에서 UR5e와 Robotiq gripper로 pick-and-place를 실행합니다. 결과는 `baseline/outputs/tabletop_before.png`, `tabletop_after.png`, `tabletop_rollout.mp4`에 저장됩니다.

## Calibration

`mobile` 모드는 calibration 실행을 지원합니다.

```bash
python3 baseline/knowno.py \
  --mode mobile \
  --run-calibration \
  --num-calibration 200 \
  --num-test 100 \
  --target-success 0.8
```

데이터 파일이 없으면 `scripts/env.py`가 Google Drive에서 다운로드를 시도합니다.

## 참고

이 baseline은 원 논문 `Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners`의 KnowNo 아이디어를 로컬 Python 스크립트로 실행하기 위한 코드입니다. 원 프로젝트 웹사이트는 https://robot-help.github.io/ 입니다.
