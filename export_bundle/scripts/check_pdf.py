"""Checks of the BUILT paper/main.pdf and paper/build/main.log (run after scripts/build_paper.sh).

Records page count, the page of the LAST SENTENCE of the main body (the \\label{body-end}
placed after the last Conclusion sentence, read from main.aux and cross-checked against the
last words of that sentence in the PDF text; a section-heading page is not used), the pages
where the references and appendix start,
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

    aux = open(os.path.join(ROOT, "paper", "build", "main.aux"), encoding="latin-1").read()
    m = re.search(r"\\newlabel\{body-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    body_end = int(m.group(1)) if m else None
    tex = open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
    last = tex[:tex.index("\\label{body-end}")].rstrip().split("\n")[-1]
    words = re.findall(r"[A-Za-z]+", re.sub(r"\\[A-Za-z]+", " ", last))[-4:]

    def norm(t):
        return " ".join(re.sub(r"-\n", "", t).split())
    match_pages = [p for p in range(1, pages + 1) if " ".join(words) in norm(texts[p])]
    log = open(LOG, encoding="latin-1").read()
    full = "\n".join(texts.values())
    report = {
        "pdf": "paper/main.pdf",
        "pages_total": pages,
        "conclusion_heading_page": first_page(r"^\s*\d+\.\s+Conclusion\s*$"),
        "body_end_page": body_end,
        "body_end_last_words": " ".join(words),
        "body_end_last_words_found_on_pages": match_pages,
        "impact_statement_page": first_page(r"^\s*Impact Statement\s*$"),
        "references_page": first_page(r"^\s*References\s*$"),
        "appendix_start_page": first_page(r"^\s*A\.\s+\S"),
        "icml_limit": "8 pages of main body (counted to the page of its last sentence); references, impact statement and appendix do not count",
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
           or body_end is None or body_end > 8 or match_pages != [body_end])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
