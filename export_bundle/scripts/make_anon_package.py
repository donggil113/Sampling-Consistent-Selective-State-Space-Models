"""Assemble anon_submission/: the ANONYMOUS review package, kept separate from the internal
evidence bundle (export_bundle/).

Contents: manuscript source and PDF, configurations, code (without the internal checker and
packaging scripts, which carry identifying search patterns), tests, and the raw records from
which scripts/make_paper_assets.py regenerates every table, figure and number.

Sanitization of text files (bytes of the internal originals are not changed):
  * commit identifiers of THIS repository (full hashes and 7-40 character prefixes) -> <commit>
    (third-party pins such as the TIDES commit are kept)
  * absolute local paths of the checkout and of the session scratch area -> <repo> / <tmp>
  * the version-history comment line at the top of main.tex is replaced
Excluded: model checkpoints, cached synthetic data, third-party code, internal documents
(STATUS, research packet, related-work log, claims ledger, notes) and internal scripts.

Checks (written to results/anon_check.json, exit 1 on any failure):
  * no identifying pattern in any packaged text file or in the PDF text/metadata; patterns are
    derived from the git remote (owner, repository name), git author/committer identities,
    this repository's commit hashes, local paths, the patterns of scripts/check_pdf.py and
    optional extra patterns in $ANON_EXTRA_PATTERNS (comma-separated; never written to disk)
  * every path the package README lists as included exists
  * the table/number generator run on a scratch copy of the package reproduces
    paper/generated/ byte for byte
"""

import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "anon_submission")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from check_pdf import ANON as PDF_ANON  # noqa: E402

INCLUDE = [
    "paper/main.tex", "paper/main.pdf", "paper/references.bib", "paper/generated/*.tex",
    "configs/*.json",
    "src/scssm/*.py", "src/scssm/real/*.py",
    "scripts/*.py", "scripts/build_paper.sh",
    "tests/*.py",
    "results/raw/*.jsonl", "results/raw/*.json", "results/raw/*.log",
    "results/raw/P1-REAL-02/*.json",
    "results/raw/P1-REAL-02/*/train.json", "results/raw/P1-REAL-02/*/train.log",
    "results/raw/P1-REAL-02/*/eval.jsonl", "results/raw/P1-REAL-02/*/eval_summary.json",
    "results/raw/P1-REAL-02/h8_decomposition/*",
    "results/raw/P1-HAR-01/*",
    "results/raw/P1-HAR-CT-01/*",
    "run_manifest.json",
]
EXCLUDE_NAMES = {"data_cache.json", "testset_cache.json", "check_paper.py", "check_pdf.py",
                 "make_export_bundle.py", "make_anon_package.py"}
BINARY_EXT = (".pdf", ".gz")
TMP_ROOT = "/tmp/claude-0"


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def patterns():
    # third-party identifiers (the TIDES repository owner, generic github.com) are allowed in the
    # package because the code must name where the pinned TIDES commit is obtained
    pats = set(w.lower() for w in PDF_ANON) - {"github.com", "taylansoydan"}
    url = git("remote", "get-url", "origin").strip()
    m = re.search(r"[/:]([^/:]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        pats.update({m.group(1).lower(), m.group(2).lower()})
    for line in git("log", "--format=%an%n%ae%n%cn%n%ce").splitlines():
        line = line.strip().lower()
        if line and line not in ("claude",):
            pats.add(line)
    pats.update({"claude.ai/code", "claude-session", "co-authored-by", ROOT.lower(), TMP_ROOT})
    for w in os.environ.get("ANON_EXTRA_PATTERNS", "").split(","):
        if w.strip():
            pats.add(w.strip().lower())
    return sorted(p for p in pats if len(p) >= 4)


def main():
    commits = [h.strip() for h in git("log", "--all", "--format=%H").splitlines() if h.strip()]
    hexre = re.compile(r"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])")

    def redact_hashes(t):
        return hexre.sub(lambda m: "<commit>" if any(c.startswith(m.group(0)) for c in commits) else m.group(0), t)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    files, n_hash, n_path = [], 0, 0
    for pat in INCLUDE:
        for src in sorted(glob.glob(os.path.join(ROOT, pat))):
            rel = os.path.relpath(src, ROOT)
            if os.path.basename(rel) in EXCLUDE_NAMES or (rel.endswith(".pt") and "P1-HAR-01" not in rel) or "__pycache__" in rel or rel in files:
                continue
            dst = os.path.join(OUT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if rel.endswith(BINARY_EXT) or rel.endswith(".pt"):
                shutil.copy2(src, dst)
            else:
                t = open(src, encoding="utf-8").read()
                if rel == "paper/main.tex":
                    first = t.split("\n", 1)
                    t = "%%%%%%%% Manuscript v4 (anonymous review copy) %%%%%%%%\n" + first[1]
                t2 = redact_hashes(t)
                n_hash += len(hexre.findall(t)) - len(hexre.findall(t2))
                t3 = t2.replace(ROOT, "<repo>")
                t3 = re.sub(re.escape(TMP_ROOT) + r"[^\s\"']*", "<tmp>", t3)
                n_path += (t2 != t3)
                open(dst, "w", encoding="utf-8").write(t3)
            files.append(rel)
    readme = os.path.join(ROOT, "paper", "anon_README.md")
    shutil.copy2(readme, os.path.join(OUT, "README.md"))
    files.append("README.md")
    with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
        f.write("# sha256 of every file in this anonymous package\n")
        f.write("\n".join(f"{sha256(os.path.join(OUT, r))}  {r}" for r in sorted(files)) + "\n")

    # ------------------------------------------------------------------ checks
    pats = patterns()
    hits = {}
    for rel in files:
        p = os.path.join(OUT, rel)
        if rel.endswith(".gz"):
            raw = gzip.open(p, "rb").read()
            t = raw.decode("utf-8") if rel.endswith(".json.gz") else raw.decode("latin-1")
        elif rel.endswith(".pt"):
            t = open(p, "rb").read().decode("latin-1")
        elif rel.endswith(".pdf"):
            t = subprocess.run(["pdftotext", p, "-"], capture_output=True, text=True).stdout
            t += subprocess.run(["pdfinfo", "-meta", p], capture_output=True, text=True).stdout
            t += subprocess.run(["pdfinfo", p], capture_output=True, text=True).stdout
            t += open(p, "rb").read().decode("latin-1")
        else:
            t = open(p, encoding="utf-8").read()
        low = t.lower()
        found = [q for q in pats if q in low]
        found += ["<repository commit hash>"] if any(
            any(c.startswith(m) for c in commits) for m in hexre.findall(low)) else []
        if found:
            hits[rel] = len(found)
    rd = open(os.path.join(OUT, "README.md")).read()
    included = rd.split("## Included", 1)[1].split("## Not included", 1)[0]
    listed = re.findall(r"`([A-Za-z0-9_./{}*-]+)`", included)
    missing = []
    for item in listed:
        if "/" not in item or item.startswith("--") or item.startswith("$"):
            continue
        g = item.replace("{1..4}", "1").replace("{seed}", "seed1")
        if not glob.glob(os.path.join(OUT, g)):
            missing.append(item)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(OUT, os.path.join(tmp, "pkg"))
        r = subprocess.run([sys.executable, os.path.join(tmp, "pkg", "scripts", "make_paper_assets.py")],
                           capture_output=True, text=True)
        regen_ok = r.returncode == 0
        diffs = []
        for f in sorted(os.listdir(os.path.join(ROOT, "paper", "generated"))):
            a = os.path.join(ROOT, "paper", "generated", f)
            b = os.path.join(tmp, "pkg", "paper", "generated", f)
            if not os.path.exists(b) or open(a, "rb").read() != open(b, "rb").read():
                diffs.append(f)
    report = {"package": "anon_submission/", "n_files": len(files),
              "bytes": sum(os.path.getsize(os.path.join(OUT, x)) for x in files),
              "n_patterns_checked": len(pats), "pattern_hits_by_file": hits,
              "commit_hashes_redacted": n_hash, "files_with_local_paths_redacted": n_path,
              "readme_listed_paths_missing": missing,
              "generator_rerun_ok": regen_ok, "generated_files_differing": diffs,
              "checkpoints_included": any(x.endswith(".pt") for x in files),
              "third_party_code_included": any(x.startswith("third_party") for x in files)}
    with open(os.path.join(ROOT, "results", "anon_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(1 if (hits or missing or not regen_ok or diffs) else 0)


if __name__ == "__main__":
    main()
