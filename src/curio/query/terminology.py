"""Terminology repair for machine-translated Chinese artefact queries.

Two layers, both auditable:

1. **Ontology coverage** - the frozen query vocabulary already knows the English
   canonical form of every concept it recognises (``vase``, ``porcelain``,
   ``dragon`` ...).  Using the *parser's slots* (not raw substring matching, which
   would fire on single characters such as ``像`` inside ``像喇叭一样``), any concept
   the translation failed to carry is appended once, with synonyms treated as
   equivalent (``bottle`` covers ``vase``).
2. **Curated visual terms** - small-artefact vocabulary generic MT models mangle
   systematically (``葫芦形`` -> "two drums stacked", ``凸钮`` -> "contour",
   ``镂空`` -> "empty").  Every entry is a *visible* appearance description; no
   dynasty, provenance, or authenticity claim is introduced.

The repair never rewrites the translation - it only appends missing concepts, so a
good translation passes through unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from curio.query.ontology import SPECIALIST_GLOSSARY, VOCAB

MAX_APPENDED = 4
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# Concepts that are interchangeable for a text encoder's purposes: if the
# translation already says ``bottle``, appending ``vase`` adds noise, not signal.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "vase": ("bottle", "flask", "vessel", "jar"),
    "bowl": ("dish", "basin"),
    "plate": ("dish", "saucer", "tray"),
    "jar": ("pot", "vessel", "container", "bottle"),
    "cup": ("beaker", "goblet"),
    "figure": ("figurine", "statuette", "statue", "sculpture"),
    "textile": ("fabric", "cloth", "silk", "weav"),
    "ornament": ("pendant", "decoration", "adornment"),
    "lid": ("cover", "lidded"),
}


@dataclass(frozen=True)
class VisualTerm:
    """One curated term: Chinese surface forms, English phrase, coverage markers."""

    zh: tuple[str, ...]
    phrase: str
    markers: tuple[str, ...]


# Curated from the systematic errors observed on ``eval/visual_dev`` translations.
# ``markers`` are substrings that prove the concept already survived translation.
VISUAL_TERMS: tuple[VisualTerm, ...] = (
    VisualTerm(("葫芦",), "double-gourd shape", ("gourd",)),
    VisualTerm(("梨形",), "pear-shaped body", ("pear",)),
    VisualTerm(("凸钮",), "raised central boss", ("boss", "knob")),
    VisualTerm(("喇叭",), "flaring trumpet-shaped mouth", ("trumpet", "flar")),
    VisualTerm(("镂空", "鏤空"), "openwork pierced decoration", ("openwork", "pierced", "reticulat")),
    VisualTerm(("竖棱", "棱线", "凸棱"), "closely spaced vertical ribs", ("rib",)),
    VisualTerm(("盖缝",), "horizontal lid seam", ("seam",)),
    VisualTerm(("镜状", "圆镜", "铜镜"), "mirror-like round disc", ("mirror", "disc")),
    VisualTerm(("浮雕", "浅浮雕"), "raised relief carving", ("relief",)),
    VisualTerm(("楼阁", "亭台"), "buildings and pavilions", ("pavilion", "building", "architectur")),
    VisualTerm(("素净", "素面", "少装饰", "无纹"), "plain undecorated surface", ("plain", "undecorat", "unadorn")),
    VisualTerm(("光亮", "光润", "莹润"), "glossy lustrous surface", ("glos", "shiny", "lustr")),
    VisualTerm(("垂褶", "衣褶", "飘动"), "flowing pleated drapery", ("draper", "pleat", "fold")),
    VisualTerm(("折线",), "angular zigzag pattern", ("zigzag", "angular")),
    VisualTerm(("珠穗", "垂珠", "流苏"), "bead tassels", ("tassel", "bead")),
    VisualTerm(("临水", "水岸"), "waterfront scenery", ("waterfront", "waterside")),
    VisualTerm(("装订", "缝线"), "binding stitches", ("stitch", "binding")),
    VisualTerm(("盘旋",), "swirling carved pattern", ("swirl", "spiral")),
    VisualTerm(("分叉角", "鹿角"), "branching antlers", ("antler",)),
    VisualTerm(("斑点",), "pale speckles", ("speckl", "spot")),
    VisualTerm(("鞍",), "saddle-like mass on the back", ("saddle",)),
    VisualTerm(("尾羽",), "long sweeping tail feathers", ("tail feather", "plume")),
    VisualTerm(("冠", "冠饰"), "crest above the head", ("crest", "crown")),
    VisualTerm(("鼓腹", "圆鼓", "圆腹"), "bulbous rounded body", ("bulbous",)),
    VisualTerm(("敞口",), "open flaring mouth", ("flaring", "wide mouth", "open mouth")),
    VisualTerm(("站立", "立像"), "full-length standing figure", ("standing", "full-length")),
    VisualTerm(("坐姿", "坐像"), "seated figure", ("seated", "sitting")),
    VisualTerm(("卧",), "reclining animal", ("reclining", "lying")),
    VisualTerm(("书册", "册页"), "closed book", ("book",)),
    VisualTerm(("人像", "人物像", "俑"), "human figure", ("figure", "figurine", "statuette")),
    VisualTerm(("织物", "织锦", "刺绣"), "textile fabric", ("textile", "fabric", "silk", "embroider")),
)


def _contains_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


def _present(translation: str, markers: tuple[str, ...]) -> bool:
    lowered = translation.lower()
    return any(marker.lower() in lowered for marker in markers)


def _tokens(phrase: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+", phrase.lower()) if len(word) > 2}


def term_additions(source_zh: str, translation: str) -> list[str]:
    """Visual expansions for specialist terms the ontology does not cover."""
    additions = []
    for term in VISUAL_TERMS:
        if not any(word in source_zh for word in term.zh):
            continue
        if _present(translation, term.markers):
            continue
        additions.append(term.phrase)
    return additions


def glossary_additions(source_zh: str, translation: str) -> list[str]:
    additions = []
    for entry in SPECIALIST_GLOSSARY.values():
        if not any(alias in source_zh for alias in entry["aliases"]):
            continue
        expansion = str(entry["expansion"])
        if expansion.split()[0].lower() in translation.lower():
            continue
        additions.append(expansion)
    return additions


def ontology_additions(slots: dict[str, list[str]] | None, translation: str) -> list[str]:
    """Canonical English words for slot values the translation failed to carry."""
    if not slots:
        return []
    additions = []
    for slot, concepts in slots.items():
        for concept in concepts:
            aliases = VOCAB.get(slot, {}).get(concept)
            if not aliases:
                continue
            english = [alias for alias in aliases if not _contains_cjk(alias)]
            if not english:
                continue
            markers = tuple(english) + SYNONYMS.get(concept, ())
            if _present(translation, markers):
                continue
            additions.append(english[0])
    return additions


def repair_translation(source_zh: str, translation: str, slots: dict[str, list[str]] | None = None,
                       *, max_appended: int = MAX_APPENDED) -> str:
    """Append only the concepts the translation lost, preserving the original text."""
    if not translation or not _contains_cjk(source_zh):
        return translation
    candidates = (
        term_additions(source_zh, translation)
        + glossary_additions(source_zh, translation)
        + ontology_additions(slots, translation)
    )
    appended: list[str] = []
    covered_tokens = _tokens(translation)
    for phrase in candidates:
        tokens = _tokens(phrase)
        if not tokens or tokens & covered_tokens:
            continue
        covered_tokens |= tokens
        appended.append(phrase)
        if len(appended) >= max_appended:
            break
    if not appended:
        return translation
    body = translation.rstrip().rstrip(".")
    return f"{body}, {', '.join(appended)}"
