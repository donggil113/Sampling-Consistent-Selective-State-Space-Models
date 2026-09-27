"""Static checks of paper/main.tex (no LaTeX compiler is available).

Checks: citation keys vs references.bib, \\ref/\\cref labels vs \\label, number macros
vs generated/numbers.tex, \\input targets, environment balance, brace balance,
anonymity strings, remaining TODOs, and a word-count estimate of the main body.
This is NOT a substitute for compiling: page count, overfull boxes, float
placement and style-file errors are unverified.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def main():
    main_tex = strip_comments(read(os.path.join(PAPER, "main.tex")))
    inputs = re.findall(r"\\input\{([^}]+)\}", main_tex)
    full = main_tex
    missing_inputs = []
    for i in inputs:
        p = os.path.join(PAPER, i if i.endswith(".tex") else i + ".tex")
        if os.path.exists(p):
            full += "\n" + strip_comments(read(p))
        else:
            missing_inputs.append(i)
    bib = read(os.path.join(PAPER, "references.bib"))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cites = set()
    for grp in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", full):
        cites.update(k.strip() for k in grp.split(","))
    labels = set(re.findall(r"\\label\{([^}]+)\}", full))
    refs = set()
    for grp in re.findall(r"\\(?:c|C)?ref\{([^}]+)\}|\\eqref\{([^}]+)\}", full):
        for g in grp:
            if g:
                refs.update(k.strip() for k in g.split(","))
    numbers = read(os.path.join(PAPER, "generated", "numbers.tex"))
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", numbers + main_tex))
    used_macros = set(re.findall(r"\\(E(?:one|two|three)\w+|Fac\w+|Comp\w+|N(?:one|two|three|four)\w+|H(?:four|seven)\w+|Nunits\w+|FirstRun\w+|Real\w+|Conv\w+|Adapter\w+)", full))
    envs_open = re.findall(r"\\begin\{(\w+\*?)\}", full)
    envs_close = re.findall(r"\\end\{(\w+\*?)\}", full)
    env_balance = {e: envs_open.count(e) - envs_close.count(e) for e in set(envs_open + envs_close)
                   if envs_open.count(e) != envs_close.count(e)}
    brace = 0
    for ch in re.sub(r"\\[{}]", "", full):
        brace += (ch == "{") - (ch == "}")
    body = main_tex.split("\\begin{abstract}")[1].split("\\section*{Impact Statement}")[0]
    body_text = re.sub(r"\\input\{[^}]+\}", " ", body)
    body_text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body_text)
    body_text = re.sub(r"\$[^$]*\$", " x ", body_text)
    words = len(re.findall(r"[A-Za-z][A-Za-z\-']+", body_text))
    anon_hits = [w for w in ("github.com", "donggil", "pusan", "TaylanSoydan", "Sampling-Consistent-Selective")
                 if w.lower() in main_tex.lower()]
    todos = re.findall(r"\\todo\{([^}]*)\}", main_tex)
    report = {
        "compile_status": "static check only; the PDF is built by scripts/build_paper.sh (see paper/build/main.log)",
        "missing_inputs": missing_inputs,
        "undefined_citations": sorted(cites - bib_keys),
        "unused_bib_entries": sorted(bib_keys - cites),
        "undefined_refs": sorted(refs - labels),
        "unreferenced_labels": sorted(labels - refs),
        "undefined_number_macros": sorted(used_macros - defined),
        "environment_imbalance": env_balance,
        "brace_balance": brace,
        "anonymity_hits_in_main_tex": anon_hits,
        "todo_count": len(todos),
        "todos": todos,
        "main_body_word_estimate": words,
        "main_body_page_estimate": "UNVERIFIED (compile required); ICML limit is 8 pages excluding references and appendix",
    }
    out = os.path.join(ROOT, "results", "paper_check.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    bad = report["missing_inputs"] or report["undefined_citations"] or report["undefined_refs"] or \
        report["undefined_number_macros"] or report["environment_imbalance"] or report["brace_balance"] or \
        report["anonymity_hits_in_main_tex"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
