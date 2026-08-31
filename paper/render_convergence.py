"""Render the convergence-paper submission PDF (PyMuPDF Story pipeline).

Extends render_submission.py: handles markdown tables and indented formal-expression
blocks as aligned monospace, and places the landscape Figure 1 on its own page.

Usage:  python render_convergence.py
Inputs: convergence_paper_draft.md, fig_designlaw.png
Output: When_Identity_Is_Load_Bearing_v1_submission.pdf
Regenerate the figure first if it changed:  python fig_designlaw.py
"""
import re
from pathlib import Path

import fitz

HERE = Path(__file__).parent
MD = HERE / "convergence_paper_draft.md"
OUT = HERE / "When_Identity_Is_Load_Bearing_v1_submission.pdf"
FIG = HERE / "fig_designlaw.png"

SANITIZE = {
    "—": " - ", "–": "-", "⟨": "<", "⟩": ">", "→": " -> ", "×": "x",
    "≈": "~", "−": "-", "…": "...", "“": '"', "”": '"', "‘": "'", "’": "'", " ": " ",
}


def sanitize(s: str) -> str:
    for k, v in SANITIZE.items():
        s = s.replace(k, v)
    return s


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    return s


def mono_line(s: str) -> str:
    """Preformatted line: escape, then preserve spacing with non-breaking spaces."""
    return '<div class="mono">' + esc(s).replace(" ", "&nbsp;") + "</div>"


def render_table(rows):
    """rows: list of raw '| a | b |' lines (incl. the |---| separator). -> mono divs."""
    cells = []
    for r in rows:
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        cells.append(parts)
    # drop the separator row (all dashes)
    body = [c for c in cells if not all(set(x) <= set("-: ") and x for x in c)]
    if not body:
        return []
    ncol = max(len(r) for r in body)
    body = [r + [""] * (ncol - len(r)) for r in body]
    widths = [max(len(r[i]) for r in body) for i in range(ncol)]
    out = ['<div class="tbl">']
    for j, r in enumerate(body):
        line = "  ".join(r[i].ljust(widths[i]) for i in range(ncol))
        out.append(mono_line(line))
        if j == 0:
            out.append(mono_line("  ".join("-" * widths[i] for i in range(ncol))))
    out.append("</div>")
    return out


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out, in_list = [], False
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        # table block
        if stripped.startswith("|") and stripped.endswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            if in_list:
                out.append("</ul>"); in_list = False
            out.extend(render_table(block))
            continue

        # indented formal-expression block (4+ leading spaces)
        if raw.startswith("    ") and stripped:
            block = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                if lines[i].strip():
                    block.append(lines[i][4:].rstrip())
                elif block:
                    break
                i += 1
            if in_list:
                out.append("</ul>"); in_list = False
            out.append('<div class="eq">')
            out.extend(mono_line(b) for b in block)
            out.append("</div>")
            continue

        i += 1
        if in_list and not line.startswith("- "):
            out.append("</ul>"); in_list = False
        if not stripped:
            continue
        if re.match(r"!\[.*?\]\((.+?)\)", line):
            continue                                  # figure -> dedicated page
        if re.match(r"^\*Figure \d+:", line):         # caption -> figure page
            while not line.rstrip().endswith("*") and i < len(lines):
                line = lines[i]; i += 1               # consume the whole italic block
            continue
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line == "---":
            out.append("<hr/>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        else:
            out.append(f"<p>{inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


CSS = """
body { font-family: serif; font-size: 10px; line-height: 1.42; text-align: justify; color: #111; }
h1 { font-size: 15px; text-align: center; margin: 2px 0 8px 0; line-height: 1.25; }
h2 { font-size: 11.5px; margin: 12px 0 4px 0; }
p  { margin: 0 0 6px 0; }
blockquote { margin: 4px 22px 8px 22px; font-size: 10px; font-style: italic; }
ul { margin: 0 0 6px 18px; }
li { margin: 0 0 3px 0; }
hr { border: 0.5px solid #999; margin: 8px 0; }
.eq  { margin: 4px 0 6px 14px; }
.tbl { margin: 6px 0 8px 6px; }
.mono { font-family: monospace; font-size: 8.5px; line-height: 1.4; margin: 0; color: #1a1a1a; }
"""


def insert_figure_page(doc, md_text):
    mlines = md_text.splitlines()
    start = next(i for i, l in enumerate(mlines) if re.match(r"^\*Figure 1:", l))
    capbuf = []
    for l in mlines[start:]:
        capbuf.append(l.strip())
        if l.rstrip().endswith("*"):
            break
    cap = " ".join(capbuf).strip("*")
    ref_page = next((i for i, p in enumerate(doc)
                     if p.search_for("instantiate (Figure 1)")), 0)
    page = doc.new_page(pno=ref_page + 1, width=612, height=792)
    # landscape image 1872x1440 (1.30:1); width 500 -> height ~385
    page.insert_image(fitz.Rect(56, 150, 556, 535), filename=str(FIG))
    page.insert_textbox(fitz.Rect(70, 552, 542, 660), sanitize(cap),
                        fontname="tiit", fontsize=9, align=3, color=(0.07, 0.07, 0.07))


def main():
    md_text = sanitize(MD.read_text(encoding="utf-8"))
    story = fitz.Story(html=md_to_html(md_text), user_css=CSS,
                       archive=fitz.Archive(str(HERE)))
    rect = fitz.paper_rect("letter") + (58, 54, -58, -54)
    writer = fitz.DocumentWriter(str(OUT))
    more = 1
    while more:
        dev = writer.begin_page(fitz.paper_rect("letter"))
        more, _ = story.place(rect)
        story.draw(dev)
        writer.end_page()
    writer.close()

    doc = fitz.open(str(OUT))
    insert_figure_page(doc, md_text)
    doc.saveIncr()
    doc.close()

    doc = fitz.open(str(OUT))
    missing = sum(page.get_text().count("�") for page in doc)
    words = sum(len(page.get_text("words")) for page in doc)
    print(f"{OUT.name}: {doc.page_count} pages, {words} words, {missing} missing glyphs")
    doc.close()


if __name__ == "__main__":
    main()
