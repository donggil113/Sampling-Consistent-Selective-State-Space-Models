# Building the manuscript

- Source: `paper/main.tex` (manuscript), `paper/references.bib`, `paper/generated/*.tex` (every table, figure and number macro; generated from `results/raw` by `scripts/make_paper_assets.py`, included in the source zip so that no experiment is needed to build).
- Style: the official ICML 2026 kit (`icml2026.zip`, sha256 `8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c`), used unmodified in anonymous review mode. It is not vendored: `scripts/build_paper.sh` downloads it from `https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip` (or copies `$ICML_KIT_ZIP`) into `paper/build/` and verifies the hash. `TARGET_YEAR=2027`, `TEMPLATE_YEAR=2026`; no official 2027 style had been published when this was built.
- Toolchain used: TeX Live (pdfTeX 1.40.25), `latexmk`, poppler (`pdfinfo`, `pdftotext`) for the checks.
- Command (from the repository root):

```bash
ICML_KIT_ZIP=/path/to/icml2026.zip scripts/build_paper.sh   # or omit ICML_KIT_ZIP to download the kit
python3 scripts/check_pdf.py                                # pages, body-end page, log warnings, metadata, anonymity strings
```

`build_paper.sh` runs `latexmk -pdf -interaction=nonstopmode -halt-on-error -pdflatex='pdflatex -no-shell-escape %O %S' main.tex` inside `paper/build/` and copies the result to `paper/main.pdf`.
