"""Render the submission PDF from the arXiv-draft markdown (PyMuPDF Story pipeline).

Usage:  python render_submission.py
Inputs: life_before_language_arxiv_draft_v0_2.md, fig1_hierarchy.png
Output: Life_Before_Language_v1_submission.pdf (letter, base-14 fonts,
        glyph-sanitized so nothing renders as a replacement box).
Regenerate the figure first if it changed:  python fig1_hierarchy.py
"""
import re
from pathlib import Path

import fitz

HERE = Path(__file__).parent
MD = HERE / "life_before_language_arxiv_draft_v0_2.md"
OUT = HERE / "Life_Before_Language_v1_submission.pdf"

# base-14 fonts miss these glyphs; map to safe equivalents
SANITIZE = {
    "—": " - ", "–": "-", "⟨": "<", "⟩": ">",
    "→": " -> ", "×": "x", "≈": "~", "−": "-",
    "…": "...", "“": '"', "”": '"', "‘": "'",
    "’": "'", " ": " ",
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


def md_to_html(md: str) -> str:
    out, in_list = [], False
    for raw in md.splitlines():
        line = raw.rstrip()
        if in_list and not line.startswith("- "):
            out.append("</ul>")
            in_list = False
        if not line.strip():
            continue
        m = re.match(r"!\[.*?\]\((.+?)\)", line)
        if m:
            # figure is inserted post-render as a dedicated page (Story
            # shrink-fits inline images to leftover space); skip here
            continue
        if re.match(r"^\*Figure \d+:", line):
            continue                    # caption goes on the figure page
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
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        else:
            out.append(f"<p>{inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


CSS = """
body { font-family: serif; font-size: 10px; line-height: 1.42;
       text-align: justify; color: #111; }
h1 { font-size: 15px; text-align: center; margin: 2px 0 8px 0; line-height: 1.25; }
h2 { font-size: 11.5px; margin: 12px 0 4px 0; }
p  { margin: 0 0 6px 0; }
blockquote { margin: 4px 24px 8px 24px; font-size: 10px; }
ul { margin: 0 0 6px 18px; }
li { margin: 0 0 3px 0; }
hr { border: 0.5px solid #999; margin: 8px 0; }
.figwrap { text-align: center; margin: 8px 0 2px 0; }
.cap { font-size: 9px; text-align: justify; margin: 2px 10px 10px 10px; }
"""


def insert_figure_page(doc, md_text):
    """Dedicated figure page right after the page bearing the in-text
    reference ('Figure 1 summarizes'), image full-size + caption."""
    cap = next(line.strip("*") for line in md_text.splitlines()
               if re.match(r"^\*Figure 1:", line))
    ref_page = next(i for i, p in enumerate(doc)
                    if p.search_for("Figure 1 summarizes"))
    page = doc.new_page(pno=ref_page + 1, width=612, height=792)
    # image aspect 1824x2112 -> width 430 gives height ~498
    page.insert_image(fitz.Rect(91, 80, 521, 578),
                      filename=str(HERE / "fig1_hierarchy.png"))
    page.insert_textbox(fitz.Rect(80, 596, 532, 700), sanitize(cap),
                        fontname="tiit", fontsize=9, align=3,
                        color=(0.07, 0.07, 0.07))


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
    doc.saveIncr()      # in-place; avoids Windows/OneDrive rename locks
    doc.close()

    doc = fitz.open(str(OUT))
    missing = sum(page.get_text().count("�") for page in doc)
    words = sum(len(page.get_text("words")) for page in doc)
    print(f"{OUT.name}: {doc.page_count} pages, {words} words, "
          f"{missing} missing glyphs")


if __name__ == "__main__":
    main()
