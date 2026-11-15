# AURO 원고 작업장

`latex_auro`에서 본문·그림·인용·실험 데이터를 편집하고 검증한다. 완료된 원고를 배포할 때만 패키지 폴더에 반영한다. 현재 디스크의 패키지 폴더 이름은 `../BRL_achitecture`이며, 대화에서 언급한 BRL_architecture에 해당한다.

## 문서 및 인용 형식

패키지의 형식에 맞춰 다음 설정을 사용한다.

- 진입 파일: `main.tex`, 본문: `section/*.tex`, 부록: `appendix/*.tex`.
- 클래스: Springer Nature `sn-jnl`, `pdflatex,iicol,sn-mathphys-ay` 옵션. 두 폴더의 클래스 내용은 같으며 파일 줄바꿈 형식만 달랐다.
- 인용: natbib 저자·연도 형식. 기존 `\cite{key}`와 BibTeX 인용 키를 유지한다.
- 참고문헌: `ref.bib`에 원문 메타데이터를 보관하고, `brl-mathphys-ay.bst`가 표시 형식을 결정한다. 이는 `sn-mathphys-ay.bst`를 조정한 프로젝트 스타일이며 패키지와 동일하다.
- 프로젝트 스타일: 참고문헌의 저자가 3명 이상이면 첫 저자와 `et al.`로 표시하고, 저널·책 제목은 이탤릭, 권 번호는 굵게 표시한다. 제목의 대소문자 변환 및 URL 출력도 같은 스타일을 사용한다. 원본 저자 목록을 `.bib`에서 잘라내지 않는다.
- 그림: `figures/` 아래에 두고 `\includegraphics`로 참조한다. 각 실험 패널은 PDF이며 기존 파일명을 유지한다. 같은 figure의 subplot은 같은 하위 폴더로 묶으며 구성은 `figures/README.md`에 기록한다.

2026-10-14 형식 정리에서는 누락된 프로젝트 BST, 활성 원고에서 인용하던 `qu2023stochastic` 항목, 이미 참조하고 있던 그림 5개만 패키지에서 보완했다. 본문·부록·실험 결과 및 기존 참고문헌 메타데이터는 덮어쓰지 않았다. 글꼴 크기 및 감사문·윤리 안내의 PDF 북마크 설정은 `main.tex`에서 명시했다.

`subcaption`이 불러오는 caption 패키지는 `sn-jnl`을 자동 인식하지 못하므로 기본 설정 경고가 남을 수 있다. 이 경고는 빌드를 중단하지 않는다.

## 빌드

이 폴더에서 실행한다:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

참고문헌 생성은 BibTeX를 사용한다. `main.bbl`, `.aux`, `.out`, `.fdb_latexmk` 등은 생성물이다. 이전 IEEE 형식의 `.bbl`을 새 문서에 함께 가져오면 저자·연도 인용 설정과 충돌할 수 있으므로 배포 파일에는 포함하지 않는다.

## 실험 그림 갱신

실험 원자료·정리 데이터·그림 생성 코드는 `asset/BRL_EXP`에서 관리한다. 사람 데이터를 추가한 뒤 프로젝트 루트에서 다음 두 명령을 실행하면 작업장 `figures`가 갱신된다.

```bash
python3 08_Delete_Me/00_real_human/summarize_human.py
python3 08_Delete_Me/00_real_human/aggregate_human_summary.py
```

상세 구조와 지표 정의는 `asset/BRL_EXP/README.md`를 참조한다.

## 완성본 포장

형식 기준을 참고하는 것과 원고 전체를 복사하는 것은 구분한다. 평소 편집과 자동 그림 갱신의 대상은 `latex_auro`다. 패키지에는 완료된 작업장의 원고와 실제로 참조하는 의존 파일만 반영한다.

배포용 ZIP 생성 예시:

```bash
python3 scripts/package_overleaf.py --output /tmp/BRL_architecture.zip
```

이 스크립트는 현재 `main.tex`가 참조하는 본문·부록·그림·BibTeX·클래스·BST를 추적하여 ZIP에 담는다. 실험 raw, Python 분석 코드, 검토 메모, 백업 원고, 이전 빌드 결과는 제외한다. ZIP 생성만 수행하며 기존 `BRL_achitecture` 폴더는 수정하지 않는다. Overleaf에서는 `main.tex`를 Main document, pdfLaTeX를 Compiler로 지정한다.

사람 로그를 추가한 후에는 `python3 08_Delete_Me/00_real_human/aggregate_human_summary.py` 하나만 실행해도 된다. 기본 경로에서는 최신 로그로 `human_summary.csv`를 먼저 재생성한 뒤 aggregate와 그림을 갱신한다. `--no-publish`는 두 CSV만 갱신한다. 별도 입력/출력 경로를 지정한 분석에서는 원자료 재요약과 자동 그림 적용을 수행하지 않는다.
