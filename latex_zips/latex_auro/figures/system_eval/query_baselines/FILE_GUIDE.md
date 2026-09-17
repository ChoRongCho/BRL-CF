# 파일 안내

이 폴더의 파일은 같은 실험을 서로 다른 집계 수준으로 저장한다.

| 파일 | 의미 | 언제 사용하는가 |
|---|---|---|
| `query_baseline_paired_runs.csv` | 실제 비교에 사용한 1,600개 episode의 원자료. `domain`, `method`, `scene`, `seed`, 성공 여부, 질문 수, 시간, 종료 사유 등을 포함한다. | 개별 episode 추적, 원본 로그 확인, 재분석 |
| `query_baseline_paired_success_contrasts.csv` | 같은 `scene + seed`에서 두 방법의 성공/실패를 짝지어 비교한 McNemar 통계. | 방법 간 성공률 차이와 paired 검정 확인 |
| `query_baseline_paper_table.csv` | 도메인별·방법별 최종 요약표. 그림에 사용되는 전체 episode 질문 수와 성공 episode 한정 질문 수를 모두 포함하며, 성공률·평균 step·평균 시간도 한 행에 모은다. | 엑셀/스프레드시트, 논문 표 작성 |
| `query_baseline_paper_table.tex` | 위 논문 요약표의 LaTeX 버전. | LaTeX 문서에 직접 삽입 |
| `query_baseline_scene_summary.csv` | 도메인·방법·scene별 `n`, 성공 수, 성공률. | 특정 scene에서 성능이 달라지는지 확인 |
| `query_baseline_summary.csv` | 도메인·방법·metric별 평균과 표준편차. `valid_n`도 함께 기록한다. | 평균/분산 그래프와 통계 재현 |

## 빠른 선택

- 그림에 대응하는 모든 최종 숫자를 한 파일에서 볼 때: `query_baseline_paper_table.csv`
- 논문에 붙일 때: `query_baseline_paper_table.tex`
- scene별 차이를 볼 때: `query_baseline_scene_summary.csv`
- 통계 검정을 볼 때: `query_baseline_paired_success_contrasts.csv`
- 원본 episode를 다시 추적할 때: `query_baseline_paired_runs.csv`
- metric별 평균·표준편차를 볼 때: `query_baseline_summary.csv`
