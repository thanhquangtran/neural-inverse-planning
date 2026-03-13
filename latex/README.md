# LaTeX Report Scaffold

Files:

- `main.tex`: starter project report.
- `refs.bib`: starter bibliography.

Compile with the repo's local virtualenv:

```bash
cd ..
.venv/bin/inverse-planning-compile-latex
```

Legacy manual flow if you have a system TeX install:

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current draft mirrors the code and notebook structure:

- exact inverse-planning baseline,
- dataset generation,
- RNN variants,
- results placeholders,
- discussion of future `memo`-based extensions.
