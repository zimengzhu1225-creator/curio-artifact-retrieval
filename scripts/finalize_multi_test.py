"""Finalize eval/multi_test: write queries/qrels/manifest from judged sheets.

Grades below come from visual re-judgment of every image graded >=1 on the
base visual_test query (contact sheets, one sheet per query). Images graded 0
on the base query stay 0: the new query is a strict conjunction over the base
conditions, so a base-0 image cannot become relevant.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "eval" / "visual_test"
OUT = ROOT / "eval" / "multi_test"

RATIONALE = (
    "Derived multi-condition subset. Each query refines a visual_test query; "
    "images graded 0 on the base query remain 0 (strict conjunction). Images "
    "graded >=1 on the base query were re-judged visually against the added "
    "conditions on per-query contact sheets. AI image judgments, not human gold."
)

# grade overrides per qid: {item_id: grade}; anything >=1 on base not listed
# was re-judged and confirmed, listed here are the confirmed grades too.
JUDGED: dict[str, dict[str, int]] = {
    "mt001": {"met_42491": 3, "met_48641": 1, "cma_120165": 1, "cma_122408": 1,
              "met_48551": 1, "met_48744": 1, "met_48763": 1, "cma_125592": 1,
              "met_48172": 0, "met_52562": 0, "met_52813": 1, "met_48483": 0,
              "cma_154599": 0},
    "mt002": {"met_53932": 3, "cma_284745": 2, "cma_160454": 3, "cma_130357": 3,
              "met_46742": 1, "met_48232": 0, "met_46709": 0, "met_47859": 0},
    "mt003": {"cma_132197": 2, "cma_101200": 3, "cma_132176": 1, "cma_132166": 0,
              "cma_109131": 1, "cma_156882": 1},
    "mt004": {"met_52565": 3, "met_62931": 2, "met_44324": 0},
    "mt005": {"cma_159167": 3, "cma_159151": 3, "met_50041": 1, "cma_121906": 0},
    "mt006": {"met_52152": 0, "met_48483": 3, "cma_154599": 3, "met_52562": 1,
              "cma_125592": 0, "cma_136630": 0},
    "mt007": {"met_52778": 3, "met_49833": 3, "cma_135008": 3, "met_52034": 3,
              "cma_122388": 0, "met_48172": 0, "met_47859": 0, "cma_134997": 1},
    "mt008": {"met_49897": 3, "met_61117": 1, "cma_154599": 0, "cma_160454": 0},
    "mt009": {"met_42491": 1, "met_52778": 3, "met_74336": 3, "cma_135008": 1,
              "cma_122408": 0, "cma_136630": 1},
    "mt010": {"met_48551": 1, "met_48744": 1, "met_48483": 2, "cma_122408": 3,
              "cma_134997": 2, "met_52562": 1, "met_48641": 1, "met_48232": 0,
              "cma_284745": 0},
    "mt011": {"cma_120165": 3, "cma_125592": 1, "cma_122388": 0, "cma_125614": 0},
    "mt012": {"met_48763": 3, "met_42491": 1, "met_48744": 0, "met_52813": 1,
              "cma_122408": 2},
    "mt013": {"met_48172": 3, "met_48483": 1, "met_48641": 0, "met_48232": 0},
    "mt014": {"cma_125592": 3, "met_52562": 1, "cma_134997": 1, "met_48483": 0,
              "cma_154599": 0},
    "mt015": {"met_48232": 3, "met_53932": 1, "cma_284745": 0, "cma_130357": 0},
    "mt016": {"met_46742": 3, "met_50041": 0, "met_47551": 0, "cma_140229": 0},
    "mt017": {"met_46709": 3, "cma_101200": 0, "cma_160454": 0, "cma_130357": 0},
    "mt018": {"cma_140229": 3, "met_52778": 0, "met_50041": 0, "met_52961": 1},
    "mt019": {"cma_101200": 3, "met_46709": 0, "cma_132197": 1, "cma_109131": 0,
              "cma_156882": 0},
    "mt020": {"cma_121906": 3, "met_47551": 0, "met_49833": 0, "cma_159167": 0,
              "met_52034": 0},
}

QUERIES = [
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
     "浅色瓷盘，盘心绘鸟栖于果树枝头，边框描彩",
     "a pale porcelain dish with a bird perched on a fruiting branch and colored rim decoration"),
    ("mt019", "vt019", "color_motif",
     "绿色系织物残片，其上织有龙纹或兽纹",
     "a green-toned textile fragment woven with dragon or beast motifs"),
    ("mt020", "vt020", "motif_shape",
     "扁平的深色墨锭，表面模压山水楼阁或龙纹浮雕",
     "a flat dark ink stick molded in relief with landscape, pavilion, or dragon designs"),
]


def main() -> None:
    base_qrels = {
        row["qid"]: row["grades"]
        for row in (json.loads(l) for l in (TEST / "qrels.jsonl").read_text(encoding="utf-8").splitlines())
    }
    gallery = json.loads((TEST / "gallery_ids.json").read_text(encoding="utf-8"))
    bm = json.loads((TEST / "blind_map.json").read_text(encoding="utf-8"))
    blind = {row["item_id"]: row["alias"] for row in bm}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gallery_ids.json").write_text(json.dumps(gallery, ensure_ascii=False), encoding="utf-8")

    q_lines, r_lines, j_lines = [], [], []
    for qid, base, qtype, zh, en in QUERIES:
        assert qid in JUDGED and base in base_qrels, (qid, base)
        grades = {iid: 0 for iid in gallery}
        for iid, g in JUDGED[qid].items():
            grades[iid] = g
        assert all(0 <= v <= 3 for v in grades.values())
        q_lines.append({"qid": qid, "text_en": en, "text_zh": zh, "type": qtype,
                        "base_qid": base, "label_source": "ai_visual_refinement",
                        "rationale": RATIONALE, "annotator": "root_ai"})
        r_lines.append({"qid": qid, "grades": grades, "label_source": "ai_visual_refinement",
                        "annotator": "root_ai", "rationale": RATIONALE})
        groups = {str(v): [blind.get(iid, iid) for iid, g in grades.items() if g == int(v)]
                  for v in (3, 2, 1, 0)}
        j_lines.append({"qid": qid, "text_en": en, "text_zh": zh, "type": qtype,
                        "base_qid": base, "annotator": "root_ai", "rationale": RATIONALE,
                        "grade_groups": groups})

    (OUT / "queries.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in q_lines) + "\n", encoding="utf-8")
    (OUT / "qrels.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in r_lines) + "\n", encoding="utf-8")
    (OUT / "query_judgments.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in j_lines) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "split": "multi_test",
        "count": 60,
        "queries": len(QUERIES),
        "candidate_pool": "identical to eval/visual_test gallery (60 images)",
        "construction": "20 multi-condition Chinese queries, each refining a visual_test "
                        "query with at least one comma-separated added condition; qrels by "
                        "conjunction derivation with visual re-judgment of base->=1 images",
        "label_source": "ai_visual_refinement; AI-generated judgments, not human gold",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(QUERIES)} queries to {OUT}")


if __name__ == "__main__":
    main()
