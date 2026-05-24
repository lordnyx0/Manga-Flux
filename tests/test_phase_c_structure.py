from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core.correction.phase_c_structure import (
    run_phase_c_structure_check,
    save_phase_c_artifacts,
    serialize_phase_c_report,
)


def _mk_base(path: Path) -> None:
    img = Image.new("L", (256, 256), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 236, 236), outline=0, width=3)
    draw.line((20, 128, 236, 128), fill=0, width=2)
    draw.line((128, 20, 128, 236), fill=0, width=2)
    img.save(path)


def _mk_color_base(path: Path) -> None:
    img = Image.new("RGB", (256, 256), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 236, 236), outline=(0, 0, 0), width=3)
    draw.line((20, 128, 236, 128), fill=(0, 0, 0), width=2)
    draw.line((128, 20, 128, 236), fill=(0, 0, 0), width=2)
    img.save(path)


def test_phase_c_detects_no_alert_on_identical(tmp_path: Path) -> None:
    orig = tmp_path / "orig.png"
    color = tmp_path / "color.png"
    _mk_base(orig)
    _mk_base(color)

    report = run_phase_c_structure_check(orig, color)

    assert report["panel_count"] >= 1
    assert report["has_structural_alert"] is False
    assert report["page_affected_ratio_pct"] == 0.0
    assert report["line_metrics"]["line_iou"] == 1.0
    assert report["line_metrics"]["line_dice"] == 1.0


def test_phase_c_detects_structural_alert_on_large_change(tmp_path: Path) -> None:
    orig = tmp_path / "orig.png"
    color = tmp_path / "color.png"
    _mk_base(orig)
    _mk_base(color)

    arr = np.asarray(Image.open(color).convert("L"), dtype=np.uint8).copy()
    arr[40:200, 40:200] = 255
    Image.fromarray(arr, mode="L").save(color)

    report = run_phase_c_structure_check(orig, color, micro_threshold_pct=1.0, critical_threshold_pct=5.0)

    assert report["has_structural_alert"] is True
    assert report["page_affected_ratio_pct"] > 0.0
    assert report["line_metrics"]["line_iou"] < 1.0
    assert any(p["verdict"] in {"micro_inpaint", "critical_inpaint"} for p in report["panels"])


def test_phase_c_serialization_removes_masks(tmp_path: Path) -> None:
    orig = tmp_path / "orig.png"
    color = tmp_path / "color.png"
    _mk_base(orig)
    _mk_base(color)

    report = run_phase_c_structure_check(orig, color)
    slim = serialize_phase_c_report(report)

    assert "inpaint_mask" in report
    assert "added_lines_mask" in report
    assert "lost_lines_mask" in report
    assert "ssim_mask" in report

    assert "inpaint_mask" not in slim
    assert "added_lines_mask" not in slim
    assert "lost_lines_mask" not in slim
    assert "ssim_mask" not in slim


def test_phase_c_artifact_writer_outputs_json_mask_and_overlay(tmp_path: Path) -> None:
    orig = tmp_path / "orig.png"
    color = tmp_path / "color.png"
    colorized = tmp_path / "colorized.png"
    _mk_base(orig)
    _mk_base(color)
    _mk_color_base(colorized)

    report = run_phase_c_structure_check(orig, color)
    artifacts = save_phase_c_artifacts(
        report,
        output_dir=tmp_path,
        page_num=7,
        overlay_base_image=colorized,
    )

    json_path = Path(artifacts["json"])
    mask_path = Path(artifacts["inpaint_mask"])
    overlay_path = Path(artifacts["overlay"])

    assert json_path.exists()
    assert mask_path.exists()
    assert overlay_path.exists()
    assert json_path.name == "page_007_phase_c_structure.json"
    assert mask_path.name == "page_007_phase_c_inpaint_mask.png"
    assert overlay_path.name == "page_007_phase_c_overlay.png"
