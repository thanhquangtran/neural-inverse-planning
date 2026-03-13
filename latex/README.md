# LaTeX Report Scaffold

Files:

- `main.tex`: starter project report.
- `refs.bib`: starter bibliography.

Suggested compile flow:

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
