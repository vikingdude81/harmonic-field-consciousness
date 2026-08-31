"""Figure for the demo-LM result: full vs ablate-bearer vs wrong-bearer accuracy,
rendered as a vector bar chart -> high-res PNG. Matches the paper figure style."""
import fitz

W, H = 720, 480
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
page.insert_text((56, 72), "a gradient-trained language model recalling a private key-value map",
                 fontname="helv", fontsize=11.5, color=MUT)

x0, y_base, y_top = 118, 400, 92
def yv(v):
    return y_base - v * (y_base - y_top)

# y axis + ticks
page.draw_line(fitz.Point(x0, y_top), fitz.Point(x0, y_base), color=(0.6, 0.6, 0.65), width=1)
page.draw_line(fitz.Point(x0, y_base), fitz.Point(662, y_base), color=(0.6, 0.6, 0.65), width=1)
for tv in (0.0, 0.25, 0.5, 0.75, 1.0):
    y = yv(tv)
    page.draw_line(fitz.Point(x0 - 5, y), fitz.Point(x0, y), color=(0.6, 0.6, 0.65), width=1)
    page.insert_text((x0 - 34, y + 4), f"{tv:.2f}", fontname="helv", fontsize=9.5, color=MUT)
page.insert_text((x0 - 40, y_top - 8), "query accuracy", fontname="helv", fontsize=10.5, color=INK)

bars = [
    ("Full model", 1.000, GREEN_F, GREEN_B, GREEN_T),
    ("Bearer ablated", 0.064, SLATE_F, SLATE_B, SLATE_T),
    ("Wrong bearer", 0.056, ROSE_F, ROSE_B, ROSE_T),
]
centers = [210, 390, 570]
bw = 104
for (label, v, f, b, t), cx in zip(bars, centers):
    top = yv(v)
    r = fitz.Rect(cx - bw / 2, top, cx + bw / 2, y_base)
    page.draw_rect(r, color=b, fill=f, width=1.5, radius=0.04)
    val = f"{v:.3f}"
    tw = fitz.get_text_length(val, fontname="hebo", fontsize=13)
    page.insert_text((cx - tw / 2, top - 9), val, fontname="hebo", fontsize=13, color=t)
    lw = fitz.get_text_length(label, fontname="hebo", fontsize=12)
    page.insert_text((cx - lw / 2, y_base + 20), label, fontname="hebo", fontsize=12, color=INK)

# chance line
yc = yv(0.062)
page.draw_line(fitz.Point(x0, yc), fitz.Point(662, yc), color=RED, width=1.6, dashes="[5 4] 0")
page.insert_text((666, yc + 4), "chance", fontname="hebo", fontsize=10, color=RED)
page.insert_text((666, yc + 17), "(1/16)", fontname="helv", fontsize=9, color=RED)

# annotation bracket-ish: the gap
page.insert_text((210 + bw / 2 + 10, (yv(1.0) + yv(0.062)) / 2),
                 "the incorporated", fontname="helv", fontsize=10, color=GREEN_T)
page.insert_text((210 + bw / 2 + 10, (yv(1.0) + yv(0.062)) / 2 + 13),
                 "memory does all of it", fontname="helv", fontsize=10, color=GREEN_T)

# caption
cap1 = "Fig. Demo LM (delta-rule fast-weight bearer, K=16). The transformer recalls the private map perfectly."
cap2 = "Freezing the bearer's write or reading against another sequence's bearer collapses accuracy to chance:"
cap3 = "the incorporated memory is load-bearing (gap +0.936) and non-transferable (gap +0.944)."
for i, c in enumerate((cap1, cap2, cap3)):
    page.insert_text((56, 446 + i * 13), c, fontname="helv", fontsize=9.5, color=MUT)

from pathlib import Path
out = Path(__file__).parent / "fig_demo_lm.png"
pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)
pix.save(str(out))
print(f"{out.name}: {pix.width}x{pix.height}")
