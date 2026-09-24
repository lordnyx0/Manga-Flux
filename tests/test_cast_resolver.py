"""CPU-only tests for cast_resolver (no VLM, no GPU).

Run: python -m pytest tests/test_cast_resolver.py -q
"""
from PIL import Image

from core.identity import cast_resolver as cr


def test_draw_numbered_marks_downscales():
    img = Image.new("RGB", (2500, 3556), "white")
    marked, _ = cr.draw_numbered_marks(img, [(10, 20, 100, 200)])
    assert max(marked.size) == cr.VLM_PAGE_MAX_SIDE


def test_parse_accepts_valid():
    good = '{"cast": [{"id": "P1"}], "assignments": [{"page": 1, "mark": 1, "person_id": "P1"}]}'
    out = cr.parse_cast_response(good)
    assert out["assignments"][0]["person_id"] == "P1"


def test_parse_rejects_malformed():
    for bad in ['{}', '{"cast": []}', "not json"]:
        try:
            cr.parse_cast_response(bad)
        except ValueError:
            continue
        raise AssertionError(f"should reject: {bad}")


def test_parse_strips_thinking():
    body = '<think>long trail...</think>\n{"cast": [], "assignments": []}'
    out = cr.parse_cast_response(body, expected_marks={})
    assert out["assignments"] == []


def test_coverage_exact():
    good = (
        '{"cast": [{"id": "P1"}], "assignments": ['
        '{"page": 1, "mark": 1, "person_id": "P1"}, '
        '{"page": 2, "mark": 1, "person_id": "P1"}]}'
    )
    cr.parse_cast_response(good, expected_marks={1: 1, 2: 1})
    cases = [
        ('{"cast": [], "assignments": []}', {1: 1}, "incomplete"),
        ('{"cast": [], "assignments": [{"page": 1, "mark": 2, "person_id": "P1"}]}',
         {1: 1}, "out-of-range"),
        ('{"cast": [], "assignments": [{"page": 1, "mark": 1, "person_id": "P1"}, '
         '{"page": 1, "mark": 1, "person_id": "P2"}]}', {1: 1}, "duplicate"),
        ('{"cast": [], "assignments": [{"page": 9, "mark": 1, "person_id": "P1"}]}',
         {1: 1}, "unknown-page"),
    ]
    for payload, expected, why in cases:
        try:
            cr.parse_cast_response(payload, expected_marks=expected)
        except ValueError:
            continue
        raise AssertionError(f"should reject ({why}): {payload}")
    # zero-mark page with zero assignments passes
    cr.parse_cast_response('{"cast": [], "assignments": []}', expected_marks={1: 0})
    # assignments to micro marks are tolerated (post-filter forces EXTRA)
    micro_ok = (
        '{"cast": [], "assignments": [{"page": 1, "mark": 1, "person_id": "P1"}, '
        '{"page": 1, "mark": 2, "person_id": "P2"}]}'
    )
    out = cr.parse_cast_response(
        micro_ok, expected_marks={1: 2}, micro_marks={1: {2}}
    )
    assert len(out["assignments"]) == 2


def test_user_prompt_labels_anchors_and_pages():
    p = cr.build_cast_user_prompt(
        ["a", "b"], anchors=[{"id": "P3"}],
        n_anchor_images=2, sent_numbers=[2, 3],
        mark_counts={2: 2, 3: 1},
    )
    assert "[Anchor ID]" in p and "Page 2" in p and "marks 1-2" in p
