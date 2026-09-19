from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from arxiv_assistant.hotspot.support.schema import HotspotItem
from arxiv_assistant.hotspot.support.source_fetch import fetch_text, is_fresh

# Labs that release by publishing weights, not by writing a post.
#
# Measured on a year of this archive: 93 frontier models were being discussed and
# 34 of them had no announcement in it at all. DeepSeek is the clearest case --
# the registry's deepseek source points at api-docs.deepseek.com, which carries
# API guides, and the GitHub organisation exposes no usable feed (an org's
# .atom returns membership events, and releases.atom needs a repository name you
# only know after the release). The weights themselves are the announcement, and
# the hub's model index is where they appear first: DeepSeek-V4.1-Flash was on
# the hub on 2026-09-10 while nothing in the archive mentioned it.
REGISTRY = Path(__file__).resolve().parents[3] / "configs" / "hotspot" / "model_hubs.json"
API = "https://huggingface.co/api/models"

FALLBACK_ORGS = [
    {"org": "deepseek-ai", "label": "DeepSeek"},
    {"org": "Qwen", "label": "Qwen"},
    {"org": "moonshotai", "label": "Moonshot AI"},
    {"org": "zai-org", "label": "Zhipu AI"},
]


def _load_registry(registry_path: str | None = None) -> list[dict]:
    path = Path(registry_path) if registry_path else REGISTRY
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return FALLBACK_ORGS
    return [o for o in data if o.get("enabled", True)] or FALLBACK_ORGS


#: Family names that identify whose model something is. Used to spot a repackage:
#: nvidia/DeepSeek-V4.1-Flash-NVFP4 is NVIDIA quantising DeepSeek's release, and
#: inclusionAI/gpt-oss-120b-singprobe is Ant probing OpenAI's. Neither is a
#: release by the account that uploaded it.
OWNERS = {
    "deepseek": "deepseek-ai", "qwen": "qwen", "kimi": "moonshotai",
    "glm": "zai-org", "minimax": "minimaxai", "minicpm": "openbmb",
    "step": "stepfun-ai", "llama": "meta-llama", "mistral": "mistralai",
    "gemma": "google", "nemotron": "nvidia", "phi": "microsoft",
    "olmo": "allenai", "granite": "ibm-granite", "aya": "coherelabs",
    "hunyuan": "tencent", "ernie": "baidu", "ling": "inclusionai",
    "mimo": "xiaomimimo", "seed": "bytedance-seed", "gpt-oss": "openai",
}

FORMAT_MARKS = (
    "-gguf", "-awq", "-gptq", "-int4", "-int8", "-fp8", "-fp4", "-bf16", "-mlx",
    "-nvfp4", "-mxfp4", "-4bit", "-8bit", "-onnx", "-openvino", "-w4a16",
    "-w8a8", "-w4a8", "-quantized", "-singprobe", "-eagle", "-draft",
)


def _is_derivative(model_id: str, org: str) -> bool:
    """True when the upload repackages a model rather than announcing one.

    Two shapes. An organisation publishes its OWN release several times -- GGUF,
    AWQ, a BF16 copy -- and each is a hub entry with its own timestamp, so
    counting them reports one release as four. And an organisation publishes
    somebody ELSE'S model, quantised or instrumented; the name still carries the
    original owner's family, which is what gives it away.
    """
    tail = model_id.rsplit("/", 1)[-1].lower()
    if any(mark in tail for mark in FORMAT_MARKS):
        return True
    here = org.lower()
    for family, owner in OWNERS.items():
        if family in tail and owner != here:
            return True
    return False


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    registry_path: str | None = None,
    per_org: int = 12,
) -> list[HotspotItem]:
    items: list[HotspotItem] = []
    for entry in _load_registry(registry_path):
        org = entry.get("org")
        if not org:
            continue
        url = "%s?author=%s&sort=createdAt&direction=-1&limit=%d" % (API, org, per_org)
        try:
            models = json.loads(fetch_text(url))
        except (ValueError, OSError):
            continue
        if not isinstance(models, list):
            continue
        for model in models:
            model_id = model.get("modelId") or model.get("id") or ""
            created = model.get("createdAt")
            if not model_id or not created:
                continue
            if _is_derivative(model_id, org):
                continue
            if not is_fresh(created, target_date, freshness_hours):
                continue
            name = model_id.rsplit("/", 1)[-1]
            downloads = model.get("downloads") or 0
            likes = model.get("likes") or 0
            items.append(HotspotItem(
                source_id="hf_models_%s" % org.lower().replace("-", "_"),
                source_name="%s on Hugging Face" % entry.get("label", org),
                source_role="official_news",
                source_type="model_release",
                title="%s released %s" % (entry.get("label", org), name),
                summary=(
                    "Weights for %s appeared on the hub. Pipeline tags: %s. "
                    "%d downloads, %d likes at collection time."
                ) % (model_id, ", ".join(model.get("tags", [])[:6]) or "none", downloads, likes),
                url="https://huggingface.co/%s" % model_id,
                canonical_url="https://huggingface.co/%s" % model_id,
                published_at=created,
                tags=list(model.get("tags", [])[:12]),
                authors=[],
                metadata={
                    "is_official": True,
                    "publisher": entry.get("label", org),
                    "model_id": model_id,
                    "downloads": downloads,
                    "likes": likes,
                },
            ))
    return items
