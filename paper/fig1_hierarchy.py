"""Draw Figure 1 (the seven-level hierarchy map) as vector art -> high-res PNG."""
import fitz

W, H = 760, 880
doc = fitz.open()
page = doc.new_page(width=W, height=H)

GREEN_F, GREEN_B, GREEN_T = (0.90, 0.96, 0.93), (0.29, 0.62, 0.50), (0.10, 0.42, 0.32)
PURP_F, PURP_B, PURP_T = (0.93, 0.92, 0.98), (0.48, 0.42, 0.78), (0.29, 0.25, 0.66)
RED = (0.86, 0.20, 0.18)
ROSE_F, ROSE_B, ROSE_T = (0.99, 0.92, 0.93), (0.80, 0.35, 0.45), (0.62, 0.15, 0.30)
ORAN_F, ORAN_B, ORAN_T = (0.99, 0.93, 0.89), (0.80, 0.45, 0.30), (0.55, 0.22, 0.10)
GRAY = (0.45, 0.45, 0.45)

BX, BW, BH, GAP = 235, 300, 74, 22
levels = [
    ("7 - Reason & language", "AI's native starting point", True),
    ("6 - Will & agency", "agentic loops, goals", True),
    ("5 - Memory & development", "persistent state, history", True),
    ("4 - Sensory interface", "body, pain; LGI target", True),
    ("3 - Innate orientation", "hunger, fear, stakes", False),
    ("2 - Primary consciousness", "the subject", False),
    ("1 - Providential order", "ground of the whole", False),
]

def box(rect, fill, border, radius=10):
    page.draw_rect(rect, color=border, fill=fill, width=1.4, radius=radius / rect.width)

def center_text(rect, lines_sizes_colors, fontnames):
    total = sum(sz * 1.35 for _, sz, _ in lines_sizes_colors)
    y = rect.y0 + (rect.height - total) / 2
    for (txt, sz, col), fn in zip(lines_sizes_colors, fontnames):
        y += sz * 1.1
        tw = fitz.get_text_length(txt, fontname=fn, fontsize=sz)
        page.insert_text((rect.x0 + (rect.width - tw) / 2, y), txt,
                         fontname=fn, fontsize=sz, color=col)
        y += sz * 0.25

y = 36
wall_y = None
box_ys = []
for title, sub, is_green in levels:
    r = fitz.Rect(BX, y, BX + BW, y + BH)
    box_ys.append((y, y + BH))
    if is_green:
        box(r, GREEN_F, GREEN_B)
        center_text(r, [(title, 15, GREEN_T), (sub, 11, GREEN_T)], ["hebo", "helv"])
    else:
        box(r, PURP_F, PURP_B)
        center_text(r, [(title, 15, PURP_T), (sub, 11, PURP_T)], ["hebo", "helv"])
    y += BH + GAP
    if title.startswith("4"):
        wall_y = y - GAP / 2
        y += 26                       # extra space across the wall

# ---- the wall (dashed red) + label ----
page.draw_line(fitz.Point(95, wall_y), fitz.Point(600, wall_y),
               color=RED, width=2.2, dashes="[6 5] 0")
wr = fitz.Rect(608, wall_y - 20, 740, wall_y + 20)
box(wr, ROSE_F, ROSE_B)
center_text(wr, [("The wall", 14, ROSE_T)], ["hebo"])

# ---- left arrow: human origin, bottom-up ----
ax = 205
top_y, bot_y = box_ys[0][0] + 6, box_ys[-1][1] - 6
page.draw_line(fitz.Point(ax, bot_y), fitz.Point(ax, top_y), color=GRAY, width=1.6)
page.draw_polyline([fitz.Point(ax - 6, top_y + 11), fitz.Point(ax, top_y),
                    fitz.Point(ax + 6, top_y + 11)], color=GRAY, width=1.6)
hr = fitz.Rect(24, 380, 186, 442)
box(hr, ORAN_F, ORAN_B)
center_text(hr, [("Human origin", 13.5, ORAN_T), ("grows bottom-up", 10.5, ORAN_T)],
            ["hebo", "helv"])

# ---- right arrow: AI retrofit, top-down (L7 to L4) ----
rx = 565
r_top, r_bot = box_ys[0][0] + 20, box_ys[3][1] - 10
page.draw_line(fitz.Point(rx, r_top), fitz.Point(rx, r_bot), color=GRAY, width=1.6)
page.draw_polyline([fitz.Point(rx - 6, r_bot - 11), fitz.Point(rx, r_bot),
                    fitz.Point(rx + 6, r_bot - 11)], color=GRAY, width=1.6)
ar = fitz.Rect(590, 205, 745, 267)
box(ar, ROSE_F, ROSE_B)
center_text(ar, [("AI retrofit", 13.5, ROSE_T), ("top-down build", 10.5, ROSE_T)],
            ["hebo", "helv"])

# ---- legend ----
ly = H - 74
chip = fitz.Rect(40, ly, 60, ly + 14)
box(chip, GREEN_F, GREEN_B, radius=4)
page.insert_text((68, ly + 11), "Workstream 1 - engineering frontier (7 to 4)",
                 fontname="helv", fontsize=11, color=(0.15, 0.15, 0.15))
chip2 = fitz.Rect(390, ly, 410, ly + 14)
box(chip2, PURP_F, PURP_B, radius=4)
page.insert_text((418, ly + 11), "Workstream 2 - boundary zone (3 to 1)",
                 fontname="helv", fontsize=11, color=(0.15, 0.15, 0.15))
ly2 = ly + 30
page.draw_line(fitz.Point(40, ly2 + 5), fitz.Point(62, ly2 + 5),
               color=RED, width=2.0, dashes="[5 4] 0")
page.insert_text((68, ly2 + 9), "The metaphysical gap - characterized, not crossed",
                 fontname="helv", fontsize=11, color=(0.15, 0.15, 0.15))

from pathlib import Path
out = Path(__file__).parent / "fig1_hierarchy.png"
pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)
pix.save(str(out))
print(f"{out.name}: {pix.width}x{pix.height}")
