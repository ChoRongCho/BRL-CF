# Baseline Debug GUI

**baseline을 하나씩 실행하는 디버깅용 폴더**

## 실행

`02_BRL_POMDP_CODE`에서:

```bash
conda activate brl
bash scripts/baseline/run_baseline_gui/run_baseline_gui.sh
```

실행 스크립트는 활성화된 `brl` 환경을 사용한다. 활성화하지 않은 경우
`~/miniconda3/envs/brl/bin/python`을 사용하며, `PYTHON`을 지정하면 우선 적용한다.
기존 baseline runner도 같은 환경의 Python을 사용하도록 PATH를 설정한다.
현재 conda Tk는 시스템 CJK 폰트를 읽지 못하므로 GUI 창은 `/usr/bin/python3`의 Tk로
표시한다 (사용 가능할 때). baseline 계산과 LLM 호출은 PATH에 설정한 `brl` Python에서 실행된다.
IntroPlan도 BRL 실행 코드로 연결되어 Jupyter가 필요하지 않다.

다른 작업 디렉터리에서도 실행할 수 있다. Python 3.8 이상과 Tkinter,
화면을 표시할 데스크톱 환경이 필요하다.

1. **Domain → Scene → Baseline** 순서로 선택한다. Scene 목록은 domain의 YAML 파일에서 읽는다.
2. Seed와 Max steps를 지정한다. Seed를 비우면 새 값을 생성하며, 미리보기 시 생성된 값은 실행에도 사용한다.
3. **명령 미리보기**로 실행 설정을 확인하고 **한 번 실행**을 누른다.
4. 하단에서 표준 출력·오류·로그 경로를 확인한다. **중지** 또는 창 닫기로 실행 프로세스와 하위 프로세스를 종료한다.

| Baseline | 동작 |
| --- | --- |
| `knowno` | `run/run_knowno_baseline.sh`로 선택한 BRL scene을 1회 실행. Auto action oracle 사용. |
| `targeted_query_pomdp` | 해당 폴더의 `run.sh`로 선택한 BRL scene을 1회 실행. Auto fact oracle 사용. |
| `introplan` | `introplan/run.sh`로 BRL scene을 실행. 예시 검색·설명 생성 후 CP 점수화, auto action oracle 사용. |

KnowNo의 설정은 프로젝트 루트 `llm_setting.json`을 사용한다.
각 baseline의 나머지 파라미터와 로그 위치는 기존 shell runner 설정을 따른다.
GUI는 단일 실행을 위해 domain, scene, seed, 최대 step, auto answer, verbose,
dry-run 여부를 지정하며, 그 외 환경변수는 상속한다.

IntroPlan 구현 차이와 calibration 방법은 [IntroPlan README](../introplan/README.md)를 참고한다.
GUI는 Noto Sans CJK KR 등 한글 글꼴을 우선 사용하고 UI·출력 기본 크기를 12pt로 설정한다.
