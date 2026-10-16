# csv 파일 안내


| 파일 | 의미 | 내용 |
|---|---|---|
| `query_baseline_paired_runs.csv` | 실제 비교에 사용한 1,600개 episode의 원자료. `domain`, `method`, `scene`, `seed`, 성공 여부, 질문 수, 시간, 종료 사유 등을 포함한다. | 개별 episode 추적, 원본 로그 확인, 재분석 |
| `query_baseline_paper_table.csv` | 도메인별·방법별 최종 요약표. 그림에 사용되는 전체 episode 질문 수와 성공 episode 한정 질문 수를 모두 포함하며, 성공률·평균 step·평균 시간도 한 행에 모은다. | 엑셀/스프레드시트, 논문 표 작성 |
| `query_baseline_scene_summary.csv` | 도메인·방법·scene별 `n`, 성공 수, 성공률. | 특정 scene에서 성능이 달라지는지 확인 |
| `query_baseline_summary.csv` | 도메인·방법·metric별 평균과 표준편차. `valid_n`도 함께 기록한다. | 평균/분산 그래프와 통계 재현 |

