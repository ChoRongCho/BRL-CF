# 논문 그림 구성

단독 그림은 figures 루트에 두고, 같은 figure의 subplot만 같은 폴더에 둔다. 파일명은 유지하고 TeX에서 폴더 경로를 명시한다.

단독 그림: `fig_pomdp_preliminaries.pdf`, `fig2_method.pdf`, `fig3_implementation.pdf`, `exp_human_ui.pdf`. 원본 사진도 루트에 보관한다.

| 폴더 | 구성 |
| --- | --- |
| 03_SETUP | WasteSorting·TomatoHarvesting 환경 subplot 2개 |
| 04_ORACLE_BASELINE | Oracle baseline의 도메인별 4개 지표, subplot 8개 |
| 05_VLM_BASELINE | VLM baseline subplot 7개 |
| 06_HUMAN_BASELINE | Human baseline subplot 7개 |
| 07_RESPONSE_COMPARISON | Oracle·VLM·Human 비교 subplot 4개 |
| 08_ABLATION | 도메인별 성공률·질문 수, subplot 4개 |
| 09_THRESHOLD | 성공률·질문 수 subplot 2개 |
| 00_DATA | 그림 수치 CSV와 사용 데이터 manifest |
| 99_LEGACY | 현재 main.tex에서 사용하지 않는 이전 그림. 백업 TeX의 경로도 이곳으로 변경 |

`asset/BRL_EXP/04_CODE/figure_paths.py`에서 생성 그림의 배포 경로를 관리한다. 사람 데이터 자동 갱신과 전체 그림 재생성 모두 이 경로를 따르며 실험 subplot을 지정된 폴더에 생성한다. `BRL_EXP` 내부의 실험별 `03_figures`는 분석 결과 보관 위치이고, 이 폴더는 원고에서 사용하는 그림 위치다.

`00_DATA/experiment_panels_manifest.json`의 `file`은 이 figures 폴더 기준 상대 경로다. 원고 포장 스크립트는 TeX에서 참조하는 그림만 ZIP에 포함하므로 99_LEGACY와 원본 사진은 참조하지 않는 한 배포하지 않는다.

`exp_human_interaction_time_per_query_all.pdf`는 성공·실패 전체 시행의 질문 1회당 상호작용 시간(전체 상호작용 시간 합계 / 전체 질의 수 합계)을 나타내는 독립 그림이다. 질문 0회 시행은 제외하고 오차막대는 표시하지 않으며, 다른 Human 비용 subplot의 success-only 기준과 구분한다.
