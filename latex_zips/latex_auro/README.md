# Autonomous Robots (AURO) LaTeX template

This directory contains the official Springer Nature journal article template,
version 3.1 (December 2024), downloaded on 2026-09-11.

The migrated manuscript entry point is `main.tex`. The active paper source,
sections, appendices, referenced figures, bibliography, and review notes were
copied from `../latex_tro/`. Duplicate assets, obsolete IEEE/ACM templates,
backup TeX files, sample PDFs, and generated build files were excluded.

The manuscript document class is configured for the current *Autonomous Robots*
author instructions:

- `iicol` for the requested two-column layout
- `sn-mathphys-ay` for author--year citations
- `pdflatex` for submission-system compatibility

Official sources:

- Journal instructions: https://link.springer.com/journal/10514/submission-guidelines
- Template download: https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940

Compile from this directory with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
