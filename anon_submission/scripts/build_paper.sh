#!/usr/bin/env bash
# Build paper/main.pdf with the OFFICIAL ICML 2026 style kit (unmodified).
# The kit is not vendored: it is downloaded (or copied from $ICML_KIT_ZIP) into paper/build/,
# its sha256 is verified, and the build runs there. Output: paper/main.pdf, paper/build/main.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$ROOT/paper/build"
URL="https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip"
SHA="8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c"
mkdir -p "$B"
ZIP="$B/icml2026.zip"
if [ -n "${ICML_KIT_ZIP:-}" ]; then cp "$ICML_KIT_ZIP" "$ZIP"; fi
if [ ! -f "$ZIP" ]; then curl -sS --max-time 120 -o "$ZIP" "$URL"; fi
echo "$SHA  $ZIP" | sha256sum -c -
python3 - "$ZIP" "$B" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for n in ("icml2026.sty", "icml2026.bst", "fancyhdr.sty", "algorithm.sty", "algorithmic.sty"):
    open(f"{sys.argv[2]}/{n}", "wb").write(z.read(n))
PY
rm -rf "$B/generated"
cp -r "$ROOT/paper/generated" "$B/generated"
cp "$ROOT/paper/main.tex" "$ROOT/paper/references.bib" "$B/"
cd "$B"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > latexmk.out 2>&1 || { tail -40 latexmk.out; exit 1; }
cp main.pdf "$ROOT/paper/main.pdf"
echo "built $ROOT/paper/main.pdf"
