# Development and Validation of the System Trustworthiness Scale

## 서지정보

| 항목 | 내용 |
| --- | --- |
| 저자 | Gene M. Alarcon, August Capiola, Michael A. Lee, Sasha Willis, Izz Aldin Hamdan, Sarah A. Jessup, Krista N. Harris |
| 출판사 / 학술지 | SAGE / Human Factors |
| 학회 | Human Factors and Ergonomics Society |
| 연도 / 권·호 / 페이지 | 2024 / 66(7) / 1893–1913 |
| 온라인 최초 출판 | 2023-07-17 |
| DOI | [10.1177/00187208231189000](https://doi.org/10.1177/00187208231189000) |
| 인용수 | [OpenAlex](https://openalex.org/W4384559207) 표시값 15회; [ResearchGate](https://www.researchgate.net/publication/372418058_Development_and_Validation_of_the_System_Trustworthiness_Scale) 표시값 20회 |
| 인용수 조회일 | 2026-09-13. 웹 검색에서 조회된 표시값으로 집계·갱신 시점이 다르며, Google Scholar나 실시간 총인용수와 같지 않음 |
| 검토 원문 | [SAGE 웹 본문](https://journals.sagepub.com/doi/10.1177/00187208231189000), [출판사 PDF의 웹 추출 본문](https://journals.sagepub.com/doi/pdf/10.1177/00187208231189000) |

기존 `.txt`는 빈 제목 메모다. 이번에는 웹에서 논문 본문과 최종 문항 표를 확인했으며, 로컬 PDF를 다운로드했다는 뜻은 아니다. PDF 첫 페이지의 Author(s) Note는 논문 내용을 미국 연방정부 저작물로서 public domain이라고 명시한다.

**인용:** Alarcon, G. M., Capiola, A., Lee, M. A., Willis, S., Hamdan, I. A., Jessup, S. A., & Harris, K. N. (2024). Development and validation of the system trustworthiness scale. *Human Factors, 66*(7), 1893–1913. https://doi.org/10.1177/00187208231189000

## 최종 제안 설문: System Trustworthiness Scale (STS)

논문이 최종 제안한 척도는 **Performance, Purpose, Process 각 5문항, 총 15문항**이다. 아래는 **Table 4, 논문 p. 1906 / PDF 14쪽**에 제시된 최종 척도의 영어 원문 전체다. 초기 개발 단계의 후보 문항은 포함하지 않았다.

Table 4는 Study 2의 통계 예측 모델을 평가할 때 사용한 문구이므로, 평가 대상인 **“statistical model”을 원문 그대로** 유지했다.

**응답 척도:** 5-point Likert scale — **1 = Strongly Disagree**, **5 = Strongly Agree**.

### Performance

| 번호 | English item — verbatim |
| --- | --- |
| 1 | The statistical model is effective. |
| 2 | The statistical model performs the task accurately. |
| 3 | The statistical model is reliable. |
| 4 | The statistical model is incompetent. (R) |
| 5 | The statistical model performs its job well. |

### Purpose

| 번호 | English item — verbatim |
| --- | --- |
| 6 | The statistical model is programmed specifically to complete this task. |
| 7 | This statistical model executes its designer’s intent. |
| 8 | The statistical model is utilized as its creators intended. |
| 9 | The statistical model is used as intended. |
| 10 | The statistical model is designed to assist users in this context. |

### Process

| 번호 | English item — verbatim |
| --- | --- |
| 11 | I understand how the statistical model is supposed to work. |
| 12 | The statistical model conveys its reasoning to users. |
| 13 | The statistical model’s operation/decision making is transparent. |
| 14 | It is clear to me how the statistical model makes decisions. |
| 15 | It is clear how the statistical model completes its tasks. |

## 역채점 및 점수

**R은 reverse-scored, 즉 역채점 표시다.** 부정문에 대한 동의 점수를 뒤집어 긍정문과 방향을 맞춘다. 예를 들어 4번 “The statistical model is incompetent.”에 5점(강하게 동의)을 답하면, 무능하다는 평가이므로 합계에는 1점을 넣는다. 1–5 척도에서는 1↔5, 2↔4로 바꾸고 3은 유지한다. 응답자는 그대로 답하고 분석자가 계산할 때 변환한다.

**4번 문항만 역채점(R)**한다. 1–5 응답에서 역채점 값은 `6 − 응답값`이다.

논문의 Practical Implications에서는 각 구성개념의 점수를 합산하여 사용하는 방법을 설명한다.

```text
Performance = Q1 + Q2 + Q3 + (6 − Q4) + Q5
Purpose     = Q6 + Q7 + Q8 + Q9 + Q10
Process     = Q11 + Q12 + Q13 + Q14 + Q15
```

각 하위척도 합계의 범위는 **5–25점**이다.

출처: [논문 Table 4 및 Practical Implications](https://journals.sagepub.com/doi/10.1177/00187208231189000), [출판사 PDF](https://journals.sagepub.com/doi/pdf/10.1177/00187208231189000).
