"""CPU-only tests for the Qwen-Image-2.1 path (no ComfyUI, no torch, no VLM).

Run: python -m pytest tests/test_qwen_payload.py -q
"""
from pathlib import Path

from core.generation.engines.qwen_engine import QwenEngine
from core.generation.orchestrator import Pass2Orchestrator


def _engine():
    return QwenEngine()


def test_workflow_has_no_flux_nodes():
    wf = _engine()._build_comfyui_workflow_json(
        prompt="Colorize <image1> using <image2>.",
        bw_image_name="bw.png",
        style_image_name="style.png",
        ref_crop_names=["crop0.png"],
        seed=42,
        options={},
    )
    kinds = {node["class_type"] for node in wf.values()}
    assert "ReferenceLatent" not in kinds
    assert "EmptyFlux2LatentImage" not in kinds
    assert "Flux2Scheduler" not in kinds
    assert "LoraLoaderModelOnly" not in kinds
    assert "TextEncodeQwenImage21" in kinds
    assert "QwenImage21Cache" in kinds
    assert wf["4"]["inputs"]["type"] == "qwen_image"
    assert wf["3"]["inputs"]["unet_name"] == "qwen-image-2.1-Q4_K_M.gguf"
    assert wf["8"]["inputs"]["cfg"] == 1.0
    assert wf["8"]["inputs"]["denoise"] == 1.0
    assert wf["8"]["inputs"]["latent_image"] == ["10", 0]
    assert "19" in wf  # polling do engine espera o SaveImage="19"


def test_workflow_image_slots():
    wf = _engine()._build_comfyui_workflow_json(
        prompt="x <image1> <image2> <image3>",
        bw_image_name="bw.png",
        style_image_name="style.png",
        ref_crop_names=["c0.png", "c1.png"],
        seed=7,
        options={},
    )
    enc = wf["6"]["inputs"]
    assert enc["images.image_1"] == ["2", 0]
    assert enc["images.image_2"] == ["21", 0]
    assert enc["images.image_3"][0] not in {"2", "21"}
    assert enc["images.image_4"][0] not in {"2", "21"}


def test_workflow_without_style_has_no_image2():
    wf = _engine()._build_comfyui_workflow_json(
        prompt="x <image1>",
        bw_image_name="bw.png",
        style_image_name=None,
        ref_crop_names=[],
        seed=1,
        options={},
    )
    assert "images.image_2" not in wf["6"]["inputs"]
    assert "20" not in wf


def test_qwen_prompt_is_deterministic_and_tagged():
    orch = Pass2Orchestrator.__new__(Pass2Orchestrator)
    prompt = Pass2Orchestrator.build_qwen_edit_prompt(
        orch, ["character with blue hair (match sim=0.50)"], "", 1
    )
    assert "<image1>" in prompt and "<image2>" in prompt
    assert "<image3>" in prompt
    assert "colorMangaKlein" not in prompt


def test_qwen_payload_fallback_without_faiss(tmp_path):
    orch = Pass2Orchestrator.__new__(Pass2Orchestrator)
    orch.metadata = {"page_image": str(tmp_path / "p.png"), "detections": []}
    orch.faiss_service = None
    orch.reference_characters_count = 0
    orch.style_ref_path = str(tmp_path / "missing.png")

    from core.generation.orchestrator import MaskBinder, PromptBuilder, StyleBinder
    orch.prompt_builder = PromptBuilder(orch.metadata)
    orch.style_binder = StyleBinder(orch.style_ref_path)
    orch.vlm_character_registry = None

    out = orch.prepare_qwen_edit_payload()
    assert out["base_image_path"].endswith("p.png")
    assert out["ref_crops"] == []
    assert "<image1>" in out["qwen_prompt"] and "<image2>" in out["qwen_prompt"]
    assert "colorMangaKlein" not in out["qwen_prompt"]


def test_ref_crops_dir_override(tmp_path):
    d = tmp_path / "refs"
    d.mkdir()
    (d / "b.png").write_bytes(b"x")
    (d / "a.png").write_bytes(b"x")
    crops = _engine()._resolve_ref_crops({"ref_crops": ["ignored.png"]}, {"ref_crops_dir": str(d)})
    assert [Path(c).name for c in crops] == ["a.png", "b.png"]
