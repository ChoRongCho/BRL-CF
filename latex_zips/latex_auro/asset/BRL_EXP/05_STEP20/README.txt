20회 제한 수정본 (2026-10-23)
물리 행동 20회까지, POMCP-Q 질문 30회까지 허용한다. 경계의 성공을 유지한다.
Oracle baseline / ablation / threshold의 기존 실행 경로를 재판정했다.
성공 시행의 질의 수·질의확률·계획 길이·기록된 계획 시간을 새 성공 집합에서 재계산했다.
VLM·Human 데이터와 원본 tex/그림/원자료는 수정하지 않는다.
수정 tex: section/05_exp_setting_step20.tex, section/05_exp_system_step20.tex
수정 부록: appendix/02_detailedEXP_step20.tex
수정 그림: figures/04_ORACLE_BASELINE_STEP20, 08_ABLATION_STEP20, 09_THRESHOLD_STEP20
main.tex가 수정본을 참조한다. 원본 main.tex/PDF와 실험 tex·그림 백업은
02_BRL_CODE/experiments_backups/20261023_manuscript_step20/에 있다.
재생성: python3 scripts/update_step20_results.py
