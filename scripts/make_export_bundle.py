"""Assemble export_bundle/ for review: manuscript source and PDF, configs, code, small raw
results and aggregates, manifests, plus a MANIFEST.sha256 over every bundled file.

Excluded on purpose (see export_bundle/README.md): model checkpoints (*.pt), cached synthetic
data (data_cache.json, testset_cache.json; regenerable and hash-checked), third-party code
(third_party/), the ICML style kit and LaTeX build intermediates (paper/build/), and caches.
The bundle is rebuilt from scratch on every run; files are copied byte-for-byte.
"""

import glob
import hashlib
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "export_bundle")

INCLUDE = [
    "paper/main.tex", "paper/main.pdf", "paper/references.bib", "paper/claims.csv",
    "paper/bib_provenance.json", "paper/generated/*.tex",
    "configs/*.json",
    "src/scssm/*.py", "src/scssm/real/*.py",
    "scripts/*.py", "scripts/build_paper.sh",
    "tests/*.py",
    "results/paper_assets.json", "results/paper_check.json", "results/pdf_check.json",
    "results/first_run_summary.json", "results/first_run_summary.md",
    "results/raw/*.jsonl", "results/raw/*.json", "results/raw/*.log",
    "results/raw/P1-REAL-02/*.json",
    "results/raw/P1-REAL-02/*/train.json", "results/raw/P1-REAL-02/*/train.log",
    "results/raw/P1-REAL-02/*/eval.jsonl", "results/raw/P1-REAL-02/*/eval_summary.json",
    "run_manifest.json", "STATUS.md", "RESEARCH_PACKET.md", "RELATED_WORK.md",
]
EXCLUDE_NAMES = {"data_cache.json", "testset_cache.json"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.isdir(OUT):
        keep = os.path.join(OUT, "README.md")
        readme = open(keep).read() if os.path.exists(keep) else None
        shutil.rmtree(OUT)
    else:
        readme = None
    os.makedirs(OUT)
    files = []
    for pat in INCLUDE:
        for src in sorted(glob.glob(os.path.join(ROOT, pat))):
            rel = os.path.relpath(src, ROOT)
            if os.path.basename(rel) in EXCLUDE_NAMES or rel.endswith(".pt") or "__pycache__" in rel:
                continue
            if rel in files:
                continue
            dst = os.path.join(OUT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            files.append(rel)
    if readme is not None:
        with open(os.path.join(OUT, "README.md"), "w") as f:
            f.write(readme)
        files.append("README.md")
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    lines = [f"{sha256(os.path.join(OUT, r))}  {r}" for r in sorted(files)]
    with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
        f.write(f"# sha256 of every bundled file; bundle built from git HEAD {head} plus the working tree at build time\n")
        f.write("\n".join(lines) + "\n")
    total = sum(os.path.getsize(os.path.join(OUT, r)) for r in files)
    print(f"{len(files)} files, {total / 1e6:.2f} MB -> {OUT}")


if __name__ == "__main__":
    main()
