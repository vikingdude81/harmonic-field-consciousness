"""Figure for the demo-LM result: full vs ablate-bearer vs wrong-bearer accuracy,
across two gradient-trained tasks (synthetic key-value map and natural-language
persona recall), rendered as a vector grouped bar chart -> high-res PNG."""
import fitz
from pathlib import Path

W, H = 820, 520
doc = fitz.open()
page = doc.new_page(width=W, height=H)

GREEN_F, GREEN_B, GREEN_T = (0.80, 0.93, 0.84), (0.24, 0.58, 0.46), (0.08, 0.40, 0.30)
SLATE_F, SLATE_B, SLATE_T = (0.86, 0.89, 0.93), (0.46, 0.53, 0.62), (0.24, 0.30, 0.39)
ROSE_F, ROSE_B, ROSE_T = (0.98, 0.87, 0.89), (0.78, 0.35, 0.45), (0.60, 0.14, 0.29)
RED = (0.86, 0.20, 0.18)
INK = (0.13, 0.13, 0.15)
MUT = (0.42, 0.42, 0.48)

# title
page.insert_text((56, 50), "The bearer carries it all", fontname="hebo", fontsize=20, color=INK)
page.insert_text((56, 72), "one architecture, two gradient-trained tasks: a synthetic key-value map and natural-language persona recall",
                 fontname="helv", fontsize=11.5, color=MUT)

x0, y_base, y_top = 130, 430, 118
def yv(v):
    return y_base - v * (y_base - y_top)

# axes + y ticks
page.draw_line(fitz.Point(x0, y_top), fitz.Point(x0, y_base), color=(0.6, 0.6, 0.65), width=1)
page.draw_line(fitz.Point(x0, y_base), fitz.Point(720, y_base), color=(0.6, 0.6, 0.65), width=1)
for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
    y = yv(tv)
    page.draw_line(fitz.Point(x0 - 5, y), fitz.Point(x0, y), color=(0.6, 0.6, 0.65), width=1)
    page.insert_text((x0 - 34, y + 4), f"{tv:.2f}", fontname="helv", fontsize=9.5, color=MUT)
page.insert_text((x0 - 40, y_top - 10), "query accuracy", fontname="helv", fontsize=10.5, color=INK)

# two groups: (label, [(name, value, fills), ...], chance, [bar centers], group-label center)
groups = [
    ("synthetic key-value map", 0.0625,
     [("Full model", 1.000, GREEN_F, GREEN_B, GREEN_T),
      ("Bearer ablated", 0.064, SLATE_F, SLATE_B, SLATE_T),
      ("Wrong bearer", 0.056, ROSE_F, ROSE_B, ROSE_T)],
     [212, 288, 364], 288),
    ("natural-language personas", 0.100,
     [("Full model", 1.000, GREEN_F, GREEN_B, GREEN_T),
      ("Bearer ablated", 0.129, SLATE_F, SLATE_B, SLATE_T),
      ("Wrong bearer", 0.094, ROSE_F, ROSE_B, ROSE_T)],
     [502, 578, 654], 578),
]
bw = 66
for glabel, chance, bars, centers, gcx in groups:
    for (label, v, f, b, t), cx in zip(bars, centers):
        top = yv(v)
        r = fitz.Rect(cx - bw / 2, top, cx + bw / 2, y_base)
        page.draw_rect(r, color=b, fill=f, width=1.5, radius=0.05)
        val = f"{v:.3f}"
        tw = fitz.get_text_length(val, fontname="hebo", fontsize=12)
        page.insert_text((cx - tw / 2, top - 8), val, fontname="hebo", fontsize=12, color=t)
    # per-group chance line
    yc = yv(chance)
    lft, rgt = centers[0] - bw / 2 - 8, centers[-1] + bw / 2 + 8
    page.draw_line(fitz.Point(lft, yc), fitz.Point(rgt, yc), color=RED, width=1.5, dashes="[5 4] 0")
    page.insert_text((rgt + 4, yc + 4), "chance", fontname="hebo", fontsize=8.5, color=RED)
    # group label
    lw = fitz.get_text_length(glabel, fontname="hebo", fontsize=12)
    page.insert_text((gcx - lw / 2, y_base + 24), glabel, fontname="hebo", fontsize=12, color=INK)

# legend (top-right)
lx, ly = 560, 96
for i, (name, f, b) in enumerate([("full model", GREEN_F, GREEN_B),
                                  ("bearer ablated", SLATE_F, SLATE_B),
                                  ("wrong bearer", ROSE_F, ROSE_B)]):
    yy = ly + i * 16
    page.draw_rect(fitz.Rect(lx, yy - 8, lx + 12, yy + 2), color=b, fill=f, width=1)
    page.insert_text((lx + 18, yy), name, fontname="helv", fontsize=9.5, color=MUT)

# caption
cap1 = "Fig. 2. The design law's positive cell under gradient descent, in two decoder transformers with a delta-rule fast-weight bearer."
cap2 = "Both recall a private per-sequence fact set perfectly; ablating the bearer's read or reading against another sequence's bearer"
cap3 = "collapses accuracy to chance. Left: an abstract key-value map (chance 1/16). Right: natural-language personas over real GPT-2 BPE"
cap4 = "text (chance 1/10). Load-bearing and non-transferable in both, in the substrate the term language model names."
for i, c in enumerate((cap1, cap2, cap3, cap4)):
    page.insert_text((56, 476 + i * 12.5), c, fontname="helv", fontsize=9, color=MUT)

out = Path(__file__).parent / "fig_demo_lm.png"
pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)
pix.save(str(out))
print(f"{out.name}: {pix.width}x{pix.height}")
