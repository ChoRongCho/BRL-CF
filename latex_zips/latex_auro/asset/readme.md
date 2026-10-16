# 현재 논문 자산

- [`BRL_EXP/`](BRL_EXP/README.md): 논문 실험 구조에 맞춘 원자료·시행별 데이터·그림/표 입력·생성 코드. 실험 데이터와 수치의 작업 기준이다.
- `vlm_prompt_configs/`: TomatoHarvesting·WasteSorting의 VLM 질문/응답 설정 YAML 및 원본 SHA-256·모델 설정 manifest. 실험 로그와 구분해서 보관하는 설정 자료다.

사람 실험을 추가하면 `08_Delete_Me/00_real_human/summarize_human.py`, `aggregate_human_summary.py`를 순서대로 실행한다. 작업장 `figures`가 자동 갱신되며 완성본 패키지는 별도로 반영한다.

이전 Oracle/VLM 묶음, 날짜별 실험 정리본, 기존 analysis_recent 및 중복 실패 분류 CSV는 2026-10-14에 작업 폴더에서 분리했다. 보관 위치: `08_Delete_Me/06_latex_auro_archive/20261014_cleanup/asset` (프로젝트 루트 기준). 원본 경로·파일 수·크기는 보관 폴더의 `manifest.json`에 기록했다.

`BRL_EXP` 재설치는 `02_BRL_CODE/experiments_final`의 원본과 현재 사람 로그를 사용한다. 분리한 자산 폴더를 다시 가져올 필요가 없다. 변경 전 입력의 출처를 기록한 `BRL_EXP/*/00_raw` 하위 문서 및 CSV에는 당시 경로가 남아 있을 수 있다.

사람 로그를 추가한 후에는 `python3 08_Delete_Me/00_real_human/aggregate_human_summary.py` 하나만 실행해도 된다. 기본 경로에서는 최신 로그로 `human_summary.csv`를 먼저 재생성한 뒤 aggregate와 그림을 갱신한다. `--no-publish`는 두 CSV만 갱신한다. 별도 입력/출력 경로를 지정한 분석에서는 원자료 재요약과 자동 그림 적용을 수행하지 않는다.
