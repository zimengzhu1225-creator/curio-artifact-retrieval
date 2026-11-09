"""Terminology repair must add signal without rewriting good translations."""

from curio.query.bridge import _is_nllb
from curio.query.terminology import MAX_APPENDED, repair_translation


def test_missing_specialist_term_is_appended():
    repaired = repair_translation("找蓝绿色的葫芦形瓶", "Find a blue green bottle with two drums stacked")
    assert "double-gourd shape" in repaired
    assert repaired.startswith("Find a blue green bottle")


def test_covered_term_is_left_alone():
    original = "a blue-green double-gourd shaped bottle"
    assert repair_translation("找蓝绿色的葫芦形瓶", original) == original


def test_single_character_simile_does_not_trigger_a_concept():
    """``像`` inside ``像喇叭一样`` is a simile, not a human figure."""
    repaired = repair_translation("找蓝白色器物，上口像喇叭一样外撇", "a blue and white vessel with a flaring mouth")
    assert "figure" not in repaired
    assert "human" not in repaired


def test_ontology_uses_parser_slots_and_treats_synonyms_as_covered():
    original = "a pale bottle with a long neck"
    assert repair_translation("找浅色瓶", original, {"object_type": ["vase"]}) == original


def test_ontology_appends_when_slot_concept_is_absent():
    repaired = repair_translation("找青花瓷盘", "a round plate with painted decoration",
                                  {"object_type": ["plate"], "material": ["porcelain"]})
    assert "porcelain" in repaired


def test_append_count_is_capped():
    source = "找素净光亮带盖缝、有竖棱、镂空的葫芦形梨形瓶"
    repaired = repair_translation(source, "an object")
    appended = repaired.split(", ", 1)[1]
    assert len(appended.split(", ")) <= MAX_APPENDED


def test_non_chinese_source_is_untouched():
    original = "a blue and white vase"
    assert repair_translation(original, original) == original


def test_empty_translation_is_safe():
    assert repair_translation("找瓶", "") == ""


def test_nllb_detection_covers_paths_and_names():
    assert _is_nllb("facebook/nllb-200-distilled-600M")
    assert _is_nllb("artifacts/models/nllb-200-distilled-600M")
    assert not _is_nllb("Helsinki-NLP/opus-mt-zh-en")
