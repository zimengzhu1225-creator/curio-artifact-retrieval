"""Build the multi-condition test subset (eval/multi_test).

Methodology:
- 20 new Chinese queries, each carrying at least two independent visual
  conditions separated by a full-width comma, so the compositional Soft-AND
  clause splitter (re.split on [，,；;。]) actually triggers.
- Each query refines an existing visual_test query (base qid recorded).
- qrels derivation: an image graded 0 on the base query stays 0 (the new
  query is a strict conjunction over the base conditions). Images graded >=1
  on the base query are re-judged visually against the new conditions via
  contact sheets (label_source recorded as ai_visual_refinement).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "eval" / "visual_test"
OUT = ROOT / "eval" / "multi_test"
SHEETS = ROOT / "_tmp_sheets"
IMG_DIR = ROOT / "data" / "processed" / "images"

RATIONALE = (
    "Derived refinement of visual_test qrels: images graded 0 on the base query "
    "remain 0 because the new query is a strict conjunction over the base conditions. "
    "Images graded >=1 on the base query were re-judged visually against the added "
    "conditions on contact sheets. AI image judgments, not human gold."
)

# qid, base qid, type, zh, en
QUERIES: list[tuple[str, str, str, str, str]] = [
    ("mt001", "vt001", "shape_motif",
     "长细颈、圆鼓腹的瓶，瓶身以青花绘花卉或缠枝纹",
     "a vase with a long neck and bulbous body, painted in underglaze blue with floral or scrolling motifs"),
    ("mt002", "vt002", "category_color",
     "无柄的敞口碗，釉面呈深褐色或灰青色，无彩绘纹饰",
     "an open bowl without handle, its glaze dark brown or grayish cyan, without painted decoration"),
    ("mt003", "vt003", "category_motif",
     "长方形织物残片，表面织出重复的花卉或几何纹样",
     "a rectangular textile fragment with repeating floral or geometric patterns woven into the surface"),
    ("mt004", "vt004", "category_color",
     "陶塑的人像或马造型，身上施三彩釉或彩绘",
     "a pottery figure of a person or horse, covered in sancai glaze or painted pigments"),
    ("mt005", "vt005", "category_motif",
     "圆形铜镜，背面中央有钮，饰浮雕动物与藤蔓纹",
     "a round bronze mirror with a central knob on the back, relief-decorated with animals and vines"),
    ("mt006", "vt006", "shape_color",
     "带流嘴的壶，一侧有弧形柄，釉色为青白或绿色",
     "an ewer with a spout and a curved handle on one side, glazed bluish-white or green"),
    ("mt007", "vt007", "shape_motif",
     "带盖的圆腹罐，盖与罐身绘人物或动物纹饰",
     "a covered round-bellied jar with painted figures or animals on the lid and body"),
    ("mt008", "vt008", "shape_texture",
     "由三个中空袋状尖足支撑的陶鬲，器身有刻划纹",
     "a pottery li cauldron standing on three hollow bag-shaped pointed feet, with incised marks on the body"),
    ("mt009", "vt009", "category_motif",
     "青花瓷器，器身绘水禽、人物或动物等场景纹",
     "a blue-and-white porcelain vessel painted with waterfowl, figures, or animal scenes"),
    ("mt010", "vt010", "color_shape",
     "绿色或青釉的瓶或壶，颈部长而收窄",
     "a green-glazed or celadon vase or ewer with a long narrowing neck"),
    ("mt011", "vt011", "color_shape",
     "通体红釉的长颈圆腹瓶，器身不带彩绘",
     "a long-necked round-bellied vase covered overall in red glaze, without overglaze painting"),
    ("mt012", "vt012", "color_motif",
     "深色底的瓷瓶，器身密绘彩色花鸟纹饰",
     "a porcelain vase on a dark ground densely painted with colorful bird-and-flower motifs"),
    ("mt013", "vt013", "color_texture",
     "浅色圆罐，釉面布满褐色开片细纹，无彩绘图案",
     "a pale jar with a crackled glaze of fine brown lines, without painted decoration"),
    ("mt014", "vt014", "shape_color",
     "带小耳的高瓶，釉面有红或紫的流动窑变斑",
     "a tall vase with small lug handles, its glaze streaked with red or purple flambe flow marks"),
    ("mt015", "vt015", "color_texture",
     "矮形敞口碗，釉面有绿色与褐色的竖向流动纹",
     "a low open bowl whose glaze shows vertical green and brown flowing streaks"),
    ("mt016", "vt016", "color_shape",
     "素白浅盘或盏托，口沿呈花瓣状分瓣，盘内无纹饰",
     "a plain white shallow dish or lobed saucer with petal-lobed rim, without interior decoration"),
    ("mt017", "vt017", "color_motif",
     "白底瓷杯或碗，器身以红色珐琅绘龙纹",
     "a white-ground porcelain cup or bowl painted with a dragon in red enamel"),
    ("mt018", "vt018", "motif_layout",
     "浅色瓷盘，盘心绘鸟栖于果树枝头，边框为彩色花卉带",
     "a pale porcelain dish painted with a bird on a fruiting branch, bordered by a colorful floral band"),
    ("mt019", "vt019", "color_motif",
     "绿色系织物残片，其上织有龙纹或兽纹",
     "a green-toned textile fragment woven with dragon or beast motifs"),
    ("mt020", "vt020", "motif_shape",
     "扁平的深色墨锭，表面模压山水楼阁或龙纹浮雕",
     "a flat dark ink stick molded in relief with landscape, pavilion, or dragon designs"),
]


def main() -> None:
    base_queries = {
        row["qid"]: row
        for row in (json.loads(l) for l in (TEST / "queries.jsonl").read_text(encoding="utf-8").splitlines())
    }
    base_qrels = {
        row["qid"]: row["grades"]
        for row in (json.loads(l) for l in (TEST / "qrels.jsonl").read_text(encoding="utf-8").splitlines())
    }
    gallery = json.loads((TEST / "gallery_ids.json").read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(exist_ok=True)

    (OUT / "gallery_ids.json").write_text(json.dumps(gallery, ensure_ascii=False), encoding="utf-8")

    sheets: list[tuple[str, list[str]]] = []
    rows_out = []
    for qid, base, qtype, zh, en in QUERIES:
        grades = base_qrels[base]
        to_judge = [iid for iid, v in grades.items() if v >= 1]
        rows_out.append({
            "qid": qid, "base_qid": base, "text_zh": zh, "text_en": en, "type": qtype,
            "to_judge": to_judge,
        })
        sheets.append((qid, to_judge))

    # write contact sheets: 12 cells per sheet, one sheet per query
    CELL_W, CELL_H, BAR = 260, 240, 26
    COLS = 4
    for qid, items in sheets:
        rows_n = (len(items) + COLS - 1) // COLS
        W, H = COLS * CELL_W, max(1, rows_n) * (CELL_H + BAR)
        sheet = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, iid in enumerate(items):
            col, row = idx % COLS, idx // COLS
            x0, y0 = col * CELL_W, row * (CELL_H + BAR)
            draw.rectangle([x0, y0, x0 + CELL_W, y0 + BAR], fill="#dddddd")
            draw.text((x0 + 6, y0 + 5), f"{idx + 1}. {iid}", fill="black")
            try:
                im = Image.open(IMG_DIR / f"{iid}.jpg")
                im.thumbnail((CELL_W - 8, CELL_H - 8))
                sheet.paste(im, (x0 + 4, y0 + BAR + (CELL_H - BAR - im.height) // 2))
            except Exception as exc:  # noqa: BLE001
                draw.text((x0 + 8, y0 + BAR + 20), f"MISSING: {exc}", fill="red")
        sheet.save(SHEETS / f"{qid}.png")

    (SHEETS / "pending.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows_out)} queries, {sum(len(r['to_judge']) for r in rows_out)} images to judge")
    print(f"sheets -> {SHEETS}")


if __name__ == "__main__":
    sys.exit(main())
