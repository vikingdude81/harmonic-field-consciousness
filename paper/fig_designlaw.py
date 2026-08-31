"""Figure 1 for the convergence paper: the design law's three regimes as a 2D
design space (escape x bearer-adaptivity), rendered as vector art -> high-res PNG.
Matches the style of fig1_hierarchy.py."""
import fitz

W, H = 780, 600
doc = fitz.open()
page = doc.new_page(width=W, height=H)

# palette (coherent with fig1_hierarchy)
SLATE_F, SLATE_B, SLATE_T = (0.90, 0.92, 0.95), (0.46, 0.53, 0.62), (0.26, 0.32, 0.41)
GREEN_F, GREEN_B, GREEN_T = (0.88, 0.96, 0.90), (0.29, 0.62, 0.50), (0.10, 0.42, 0.32)
ROSE_F, ROSE_B, ROSE_T = (0.99, 0.91, 0.92), (0.80, 0.35, 0.45), (0.62, 0.15, 0.30)
RED = (0.86, 0.20, 0.18)
INK = (0.13, 0.13, 0.15)
MUT = (0.42, 0.42, 0.48)


def box(rect, fill, border, radius=12, width=1.5):
    page.draw_rect(rect, color=border, fill=fill, width=width, radius=radius / rect.width)


def ctext(rect, lines, fonts, dy=0.0):
    total = sum(sz * 1.4 for _, sz, _ in lines)
    y = rect.y0 + (rect.height - total) / 2 + dy
    for (txt, sz, col), fn in zip(lines, fonts):
        y += sz * 1.1
        tw = fitz.get_text_length(txt, fontname=fn, fontsize=sz)
        page.insert_text((rect.x0 + (rect.width - tw) / 2, y), txt, fontname=fn, fontsize=sz, color=col)
        y += sz * 0.3


# title
page.insert_text((60, 52), "The design law: three regimes of an incorporated bearer",
                 fontname="hebo", fontsize=19, color=INK)
page.insert_text((60, 74), "when does developmental state become causally load-bearing?",
                 fontname="helv", fontsize=12.5, color=MUT)

# plot frame
x0, x1, y0, y1 = 150, 726, 100, 486
xm = x0 + int((x1 - x0) * 0.46)          # foreclosure boundary
ym = y0 + int((y1 - y0) * 0.5)           # static/adaptive boundary

# regions
box(fitz.Rect(x0, y0, xm - 3, y1), SLATE_F, SLATE_B)                 # escape open (full height)
box(fitz.Rect(xm + 3, y0, x1, ym - 3), GREEN_F, GREEN_B)            # closed + adaptive
box(fitz.Rect(xm + 3, ym + 3, x1, y1), ROSE_F, ROSE_B)             # closed + static

# region labels
ctext(fitz.Rect(x0, y0, xm - 3, y1),
      [("Bearer idle", 20, SLATE_T), ("the escape is taken", 12.5, SLATE_T),
       ("", 6, SLATE_T), ("forager  .  FourD", 11.5, MUT)], ["hebo", "helv", "helv", "helv"])
ctext(fitz.Rect(xm + 3, y0, x1, ym - 3),
      [("Load-bearing", 19, GREEN_T), ("and non-transferable", 12.5, GREEN_T),
       ("", 5, GREEN_T), ("Constitutive GPT  .  gap 0.509", 11.5, GREEN_T)],
      ["hebo", "helv", "helv", "helv"])
ctext(fitz.Rect(xm + 3, ym + 3, x1, y1),
      [("Difficulty valley", 18, ROSE_T), ("the system fails", 12.5, ROSE_T),
       ("", 5, ROSE_T), ("forager  .  FourD", 11.5, MUT)], ["hebo", "helv", "helv", "helv"])

# foreclosure boundary (dashed red) + label
page.draw_line(fitz.Point(xm, y0 - 8), fitz.Point(xm, y1 + 8), color=RED, width=2.2, dashes="[6 5] 0")
fl = "foreclosure"
tw = fitz.get_text_length(fl, fontname="hebo", fontsize=11)
page.insert_text((xm - tw / 2, y0 - 14), fl, fontname="hebo", fontsize=11, color=RED)

# adaptivity split (only on the right, subtle)
page.draw_line(fitz.Point(xm + 3, ym), fitz.Point(x1, ym), color=(0.55, 0.55, 0.6), width=1.0, dashes="[3 4] 0")

# scaling-corollary arrow inside the idle region (capacity pushes deeper into escape)
ay = y0 + 40
page.draw_line(fitz.Point(x0 + 118, ay), fitz.Point(x0 + 24, ay), color=SLATE_T, width=1.6)
page.draw_polyline([fitz.Point(x0 + 36, ay - 6), fitz.Point(x0 + 24, ay), fitz.Point(x0 + 36, ay + 6)],
                   color=SLATE_T, width=1.6)
page.insert_text((x0 + 20, ay - 12), "capacity up  ->  escape exploited more", fontname="helv", fontsize=10, color=SLATE_T)

# X axis
page.insert_text((x0 - 4, y1 + 26), "OPEN", fontname="hebo", fontsize=11, color=MUT)
fw = fitz.get_text_length("FORECLOSED", fontname="hebo", fontsize=11)
page.insert_text((x1 - fw, y1 + 26), "FORECLOSED", fontname="hebo", fontsize=11, color=MUT)
xt = "Escape route  (a non-incorporated path to viability)"
tw = fitz.get_text_length(xt, fontname="helv", fontsize=12)
page.insert_text(((x0 + x1) / 2 - tw / 2, y1 + 48), xt, fontname="helv", fontsize=12, color=INK)

# Y axis (rotated title + adaptive/static markers on the right column)
page.insert_text((x1 + 16, y0 + 66), "ADAPTIVE", fontname="hebo", fontsize=10, color=GREEN_T, rotate=90)
page.insert_text((x1 + 16, y1 - 8), "STATIC", fontname="hebo", fontsize=10, color=ROSE_T, rotate=90)
page.insert_text((116, y1 - 6), "Bearer", fontname="helv", fontsize=12, color=INK, rotate=90)

# caption
cap1 = "Fig 1.  An incorporated bearer is load-bearing and non-transferable only when the task forecloses every"
cap2 = "non-incorporated route AND the bearer is adaptive (top right). Capacity alone deepens the escape (left)."
page.insert_text((60, H - 40), cap1, fontname="helv", fontsize=10, color=MUT)
page.insert_text((60, H - 26), cap2, fontname="helv", fontsize=10, color=MUT)

from pathlib import Path
out = Path(__file__).parent / "fig_designlaw.png"
pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)
pix.save(str(out))
print(f"{out.name}: {pix.width}x{pix.height}")
