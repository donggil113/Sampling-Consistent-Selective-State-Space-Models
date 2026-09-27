"""Checks of the BUILT paper/main.pdf and paper/build/main.log (run after scripts/build_paper.sh).

Records page count, the pages where the main body, references and appendix start or end,
LaTeX warnings (overfull/underfull boxes, undefined or multiply defined references),
PDF metadata, and anonymity strings in the extracted PDF text. Uses pdfinfo/pdftotext
(poppler). Writes results/pdf_check.json. A visual render check is separate.
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "paper", "main.pdf")
LOG = os.path.join(ROOT, "paper", "build", "main.log")
ANON = ("donggil", "pusan", "github.com", "TaylanSoydan", "Sampling-Consistent-Selective", "dgkang")


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def page_text(p):
    return run(["pdftotext", "-f", str(p), "-l", str(p), PDF, "-"])


def main():
    info = {}
    for line in run(["pdfinfo", PDF]).splitlines():
        k, _, v = line.partition(":")
        info[k.strip()] = v.strip()
    pages = int(info["Pages"])
    texts = {p: page_text(p) for p in range(1, pages + 1)}

    def first_page(pattern):
        for p in range(1, pages + 1):
            if re.search(pattern, texts[p], re.M):
                return p
        return None

    log = open(LOG, encoding="latin-1").read()
    full = "\n".join(texts.values())
    report = {
        "pdf": "paper/main.pdf",
        "pages_total": pages,
        "conclusion_page": first_page(r"^\s*\d+\.\s+Conclusion\s*$"),
        "impact_statement_page": first_page(r"^\s*Impact Statement\s*$"),
        "references_page": first_page(r"^\s*References\s*$"),
        "appendix_start_page": first_page(r"^\s*A\.\s+\S"),
        "icml_limit": "8 pages of main body; references, impact statement and appendix do not count",
        "overfull_boxes": len(re.findall(r"^Overfull", log, re.M)),
        "underfull_boxes": len(re.findall(r"^Underfull", log, re.M)),
        "undefined_references_or_citations": len(re.findall(r"undefined", log, re.I)),
        "multiply_defined_labels": len(re.findall(r"multiply defined", log, re.I)),
        "unresolved_question_marks_in_text": full.count("??"),
        "metadata": {k: info.get(k) for k in ("Title", "Author", "Subject", "Keywords", "Creator", "Producer")},
        "anonymity_hits_in_pdf_text": [w for w in ANON if w.lower() in full.lower()],
        "anonymity_hits_in_metadata": [w for w in ANON if w.lower() in json.dumps(info).lower()],
    }
    with open(os.path.join(ROOT, "results", "pdf_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    bad = (report["overfull_boxes"] or report["undefined_references_or_citations"]
           or report["multiply_defined_labels"] or report["anonymity_hits_in_pdf_text"]
           or report["anonymity_hits_in_metadata"]
           or (report["impact_statement_page"] or 99) > 9)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
