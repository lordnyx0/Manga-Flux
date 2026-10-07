"""CPU-only tests: Qwen aponta, heurística escreve (sem servidor/GPU).

Run: python -m pytest tests/test_vlm_palette.py -q
"""
import pytest
from PIL import Image

from core.identity import vlm_palette as vp
from core.identity.character_registry import CharacterRegistry


def test_parse_accepts_valid():
    body = ('<think>t</think>\n```json\n{"hair": {"color": "dark purple", '
            '"usable": true}, "skin": {"color": null, "usable": false}, '
            '"eyes": {"color": "green", "usable": true}, '
            '"clothes": {"color": "white", "usable": true}, '
            '"crop_ok": true, "note": ""}\n```')
    out = vp.parse_vlm_palette_response(body)
    assert out["hair"] == {"color": "dark purple", "usable": True}
    assert out["skin"]["usable"] is False
    assert out["crop_ok"] is True


def test_parse_rejects_garbage():
    for bad in ["no json", '{"hair": 42}', "```\n{}\n```\n"]:
        try:
            vp.parse_vlm_palette_response(bad)
        except ValueError:
            continue
        raise AssertionError(f"should reject: {bad}")


def test_word_hex_agree():
    assert vp.word_hex_agree("dark purple", "#4B3550") is True
    assert vp.word_hex_agree("light blue", "#92BBEE") is True
    assert vp.word_hex_agree("dark purple", "#EECEBC") is False
    assert vp.word_hex_agree(None, "#FFFFFF") is False
    assert vp.word_hex_agree("dark purple", None) is False


def _synth(path, hair_rgb=(75, 53, 80)):
    img = Image.new("RGB", (100, 200), (240, 240, 240))
    px = img.load()
    for y in range(0, 30):
        for x in range(100):
            px[x, y] = hair_rgb
    img.save(path)
    return path


def test_register_from_vlm_confirms_agreement(tmp_path):
    p = _synth(tmp_path / "c.png")
    reg = CharacterRegistry(tmp_path / "ledger.json", chapter="t")
    vlm = {"crop_ok": True, "note": "",
           "hair": {"color": "dark purple", "usable": True},
           "skin": {"color": None, "usable": False},
           "eyes": {"color": None, "usable": False},
           "clothes": {"color": None, "usable": False}}
    e = reg.register_from_vlm("P1", p, (0, 0, 100, 200), 1, vlm,
                              input_size=(100, 200))
    assert e["region_verdicts"]["hair"] == "confirmed"
    assert e["palette"]["hair"] is not None
    assert e["needs_review"] is False


def test_register_from_vlm_holds_conflict(tmp_path):
    p = _synth(tmp_path / "c.png", hair_rgb=(200, 120, 60))  # laranja válido
    reg = CharacterRegistry(tmp_path / "ledger.json", chapter="t")
    vlm = {"crop_ok": True, "note": "",
           "hair": {"color": "dark purple", "usable": True},
           "skin": {"color": None, "usable": False},
           "eyes": {"color": None, "usable": False},
           "clothes": {"color": None, "usable": False}}
    e = reg.register_from_vlm("P1", p, (0, 0, 100, 200), 1, vlm,
                              input_size=(100, 200))
    assert e["region_verdicts"]["hair"] == "conflict"
    assert e["palette"]["hair"] is None
    assert e["needs_review"] is True
    assert e["status"] == "provisional"


def test_register_from_vlm_face_only_crop(tmp_path):
    p = _synth(tmp_path / "c.png")
    reg = CharacterRegistry(tmp_path / "ledger.json", chapter="t")
    vlm = {"crop_ok": False, "note": "face only",
           "hair": {"color": "dark purple", "usable": True},
           "skin": {"color": "peach", "usable": True},
           "eyes": {"color": None, "usable": False},
           "clothes": {"color": "white", "usable": True}}
    e = reg.register_from_vlm("P1", p, (0, 0, 100, 200), 1, vlm,
                              input_size=(100, 200))
    assert e["region_verdicts"]["clothes"] == "held"


def test_vlm_first_falls_back_offline(tmp_path):
    p = _synth(tmp_path / "c.png")
    reg = CharacterRegistry(tmp_path / "ledger.json", chapter="t")
    e = vp.register_character_vlm_first(reg, "P9", p, (0, 0, 100, 200), 1,
                                        vlm_service=None, input_size=(100, 200))
    assert e.get("vlm_skipped") is True
    assert e["palette"]["hair"] is not None


def _hx(s):
    return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))


def test_ruler_ignores_shadow(tmp_path):
    from core.identity.character_registry import _median_hex, _robust_hex
    from core.identity.palette_manager import PaletteExtractor
    ext = PaletteExtractor()
    base = Image.new("RGB", (100, 100), (75, 53, 80))
    px = base.load()
    for y in range(35, 100):  # 65% de sombra fria (maioria: mediana cega cai)
        for x in range(100):
            px[x, y] = (25, 22, 30)
    old = _median_hex(base)
    new = _robust_hex(base)
    assert new is not None
    d_old = ext.calculate_delta_e(_hx(old), (75, 53, 80))
    d_new = ext.calculate_delta_e(_hx(new), (75, 53, 80))
    assert d_new < d_old
    assert d_new < 25.0


def test_ruler_refuses_gray(tmp_path):
    from core.identity.character_registry import _robust_hex
    gray = Image.new("RGB", (100, 100), (150, 150, 150))
    assert _robust_hex(gray) is None
