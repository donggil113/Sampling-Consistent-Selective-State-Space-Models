"""Write deliverables/: <ID>_<version>_<commit7>.pdf (byte copy of paper/main.pdf), <ID>_source.zip
(main.tex, references.bib, generated tables/figures, BUILD.md, build script) and <ID>_delivery.json
(source commit, PDF sha256/bytes/pages, body_end_page, style, build command, pages viewed,
research status, file list with hashes). Usage: python3 scripts/make_deliverables.py --id P1 --version v5.2
--viewed "pages 1-19 rendered at 96 dpi and inspected" [--built-at <commit7>]
"""

import argparse
import hashlib
import json
import os
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "deliverables")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--viewed", required=True)
    ap.add_argument("--status", default="")
    a = ap.parse_args()
    head7 = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    os.makedirs(OUT, exist_ok=True)
    pdf_src = os.path.join(ROOT, "paper", "main.pdf")
    pdf = os.path.join(OUT, f"{a.id}_{a.version}_{head7}.pdf")
    open(pdf, "wb").write(open(pdf_src, "rb").read())
    assert sha(pdf) == sha(pdf_src)
    info = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout.splitlines() if ":" in l}
    pc = json.load(open(os.path.join(ROOT, "results", "pdf_check.json")))
    # source zip
    zp = os.path.join(OUT, f"{a.id}_source.zip")
    files = ["paper/main.tex", "paper/references.bib", "paper/BUILD.md", "scripts/build_paper.sh", "scripts/make_paper_assets.py", "scripts/check_paper.py", "scripts/check_pdf.py"]
    files += sorted(os.path.relpath(os.path.join(ROOT, "paper", "generated", f), ROOT) for f in os.listdir(os.path.join(ROOT, "paper", "generated")))
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(ROOT, f), f)
    listing = {f: sha(os.path.join(ROOT, f)) for f in files}
    d = {"project": a.id, "manuscript_version": a.version, "source_commit": head7, "working_tree_dirty_at_delivery": dirty,
         "pdf": os.path.basename(pdf), "pdf_sha256": sha(pdf), "pdf_bytes": os.path.getsize(pdf), "pdf_pages": int(info["Pages"]),
         "pdf_signature": open(pdf, "rb").read(8).decode("latin-1"), "body_end_page": pc["body_end_page"],
         "style": "official ICML 2026 kit (icml2026.zip sha256 8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c), anonymous review mode, unmodified .sty/.bst; TARGET_YEAR=2027, TEMPLATE_YEAR=2026, SUBMISSION_READY=false",
         "build_command": "scripts/build_paper.sh: copies the official kit (sha256-checked) into paper/build/, then `latexmk -pdf -interaction=nonstopmode -halt-on-error -pdflatex='pdflatex -no-shell-escape %O %S' main.tex`",
         "checks": {k: pc[k] for k in ("pages_total", "overfull_boxes", "undefined_references_or_citations", "multiply_defined_labels", "anonymity_hits_in_pdf_text", "metadata")},
         "pages_viewed": a.viewed, "research_status": a.status,
         "source_zip": os.path.basename(zp), "source_zip_sha256": sha(zp), "files_in_source_zip": listing,
         "delivered_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip()}
    json.dump(d, open(os.path.join(OUT, f"{a.id}_delivery.json"), "w"), indent=2)
    print(json.dumps({k: d[k] for k in ("pdf", "pdf_sha256", "pdf_bytes", "pdf_pages", "body_end_page", "source_zip")}, indent=1))


if __name__ == "__main__":
    main()
