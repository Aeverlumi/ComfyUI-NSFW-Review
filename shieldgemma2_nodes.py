"""Optional ShieldGemma 2 image-policy review for stills and decoded video frames."""

import json
import math
from pathlib import Path

from image_models import _pil_frames, _sample_indices, _torch_device
from model_store import models_root


MODEL_ID = "google/shieldgemma-2-4b-it"
POLICY_PRESETS = json.loads(
    (Path(__file__).parent / "web" / "policy_presets.json").read_text(encoding="utf-8")
)
POLICY_MODES = (*POLICY_PRESETS, "custom")


def _selected_policy(policy_mode, custom_policy):
    if policy_mode == "custom":
        if not custom_policy.strip():
            raise ValueError("A custom policy description is required")
        return {"label": "Custom policy", "kind": "custom", "processor_key": "review_policy",
                "source": None, "text": custom_policy.strip()}
    if policy_mode not in POLICY_PRESETS:
        raise ValueError("Unknown ShieldGemma 2 policy mode")
    preset = POLICY_PRESETS[policy_mode]
    return {**preset, "processor_key": preset.get("processor_key", "review_policy")}


def _model_folder(model_path):
    folder = Path(model_path).expanduser().resolve() if model_path.strip() else models_root() / "shieldgemma2"
    if not (folder / "config.json").is_file() or not list(folder.glob("*.safetensors")):
        raise FileNotFoundError(
            f"ShieldGemma 2 weights are missing from {folder}. Accept the model license at "
            f"https://huggingface.co/{MODEL_ID} and download the complete repository there."
        )
    return folder


def _score_frames(frames, folder, policy_mode, custom_policy, batch_size, preference):
    try:
        import torch
        from transformers import AutoProcessor, ShieldGemma2ForImageClassification
    except ImportError as exc:
        raise RuntimeError("ShieldGemma 2 needs transformers>=4.56 and PyTorch") from exc

    device = _torch_device(preference)
    processor = AutoProcessor.from_pretrained(str(folder), local_files_only=True)
    model = ShieldGemma2ForImageClassification.from_pretrained(
        str(folder), local_files_only=True, use_safetensors=True, dtype=torch.bfloat16
    ).to(device).eval()
    scores = []
    try:
        selected = _selected_policy(policy_mode, custom_policy)
        if selected["kind"] == "official":
            policy_key = selected["processor_key"]
            if policy_key not in processor.policy_definitions:
                raise ValueError(f"The model processor does not provide the official {policy_key} policy")
            policy_kwargs = {"policies": [policy_key]}
        else:
            policy_kwargs = {"policies": ["review_policy"],
                             "custom_policies": {"review_policy": selected["text"]}}
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                inputs = processor(
                    images=frames[start:start + batch_size], return_tensors="pt", **policy_kwargs
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                probabilities = model(**inputs).probabilities.float().cpu().tolist()
                if len(probabilities) != len(frames[start:start + batch_size]):
                    raise RuntimeError("ShieldGemma 2 returned an unexpected number of policy scores")
                # Transformers documents the output order as [Yes, No].
                scores.extend({"yes": float(row[0]), "no": float(row[1])} for row in probabilities)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return scores


def review_frames(images, fps, max_frames, batch_size, device, top_k,
                  policy_mode, custom_policy, review_threshold, model_path="", scorer=None):
    if not math.isfinite(fps) or fps <= 0 or max_frames < 0 or batch_size < 1 or top_k < 1:
        raise ValueError("fps, batch_size and top_k must be positive; max_frames must be nonnegative")
    if not math.isfinite(review_threshold) or not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold must be between 0 and 1")
    selected = _selected_policy(policy_mode, custom_policy)

    indices = _sample_indices(len(images), max_frames)
    frames = _pil_frames(images, indices)
    folder = _model_folder(model_path) if scorer is None else None
    raw = (scorer or _score_frames)(frames, folder, policy_mode, custom_policy, batch_size, device)
    if len(raw) != len(indices):
        raise RuntimeError("ShieldGemma 2 returned a different number of results than input frames")

    records = []
    for index, probabilities in zip(indices, raw):
        yes, no = float(probabilities["yes"]), float(probabilities["no"])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (yes, no)) or abs(yes + no - 1) > 0.01:
            raise ValueError(f"Invalid ShieldGemma 2 probabilities for frame {index}")
        records.append({
            "frame_index": index,
            "timestamp_seconds": round(index / fps, 3),
            "probabilities": {"yes": round(yes, 6), "no": round(no, 6)},
            "predicted_label": "yes" if yes > no else "no",
            "review_flag": yes >= review_threshold,
        })

    top_frames = sorted(records, key=lambda frame: frame["probabilities"]["yes"], reverse=True)[:top_k]
    report = {
        "schema_version": "shieldgemma2-policy-signals-v1",
        "model": MODEL_ID,
        "policy_mode": policy_mode,
        "policy_key": selected["processor_key"],
        "policy_label": selected["label"],
        "policy_kind": selected["kind"],
        "policy_source": selected["source"],
        "policy_text": selected["text"],
        "custom_policy": custom_policy if policy_mode == "custom" else None,
        "input_frames": len(images),
        "sampled_frames": len(records),
        "fps_assumption": fps,
        "review_threshold": review_threshold,
        "review_required": any(frame["review_flag"] for frame in records),
        "label_counts": {label: sum(frame["predicted_label"] == label for frame in records)
                         for label in ("yes", "no")},
        "max_yes_probability": top_frames[0]["probabilities"]["yes"],
        "top_frames": top_frames,
        "frames": records,
        "limitations": "Yes means this image violates the specified policy. Scores are not calibrated Meta approval probabilities. Only sampled frames were checked; audio, ad text and landing page were not checked.",
    }
    lines = [
        "ShieldGemma 2 图片政策检测",
        f"政策：{selected['label']}（{selected['kind']}）",
        selected["text"],
        f"输入 {len(images)} 帧，检测 {len(records)} 帧；Yes {report['label_counts']['yes']} 帧，No {report['label_counts']['no']} 帧",
        f"复核阈值：{review_threshold:.1%}；需复核：{'是' if report['review_required'] else '否'}",
        f"最高 Yes 分数：{report['max_yes_probability']:.1%}",
        "逐帧结果：",
    ]
    lines.extend(
        f"- {frame['timestamp_seconds']:.2f}s | 第 {frame['frame_index']} 帧 | "
        f"{frame['predicted_label'].upper()} | Yes {frame['probabilities']['yes']:.1%}, "
        f"No {frame['probabilities']['no']:.1%} | {'复核' if frame['review_flag'] else '未触发阈值'}"
        for frame in records
    )
    lines.append("分数只表示该政策下的模型判断，不是视频 NSFW 总评分或 Meta 过审概率；未抽中的画面、音频和广告上下文仍需复核。")
    return json.dumps(report, ensure_ascii=False), "\n".join(lines), report["max_yes_probability"]


class ShieldGemma2PolicySignals:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 14.0, "min": 0.001, "max": 240.0}),
            "max_frames": ("INT", {"default": 48, "min": 0, "max": 100000}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 32}),
            "device": (["auto", "cuda", "cpu"],),
            "policy_mode": (list(POLICY_MODES),),
            "custom_policy": ("STRING", {"default": "", "multiline": True}),
            "review_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "top_k": ("INT", {"default": 8, "min": 1, "max": 100}),
            "model_path": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("signals_json", "readable_report", "max_yes_probability", "policy_text")
    FUNCTION = "run"
    CATEGORY = "video/review/image signals"

    def run(self, images, fps, max_frames, batch_size, device, policy_mode,
            custom_policy, review_threshold, top_k, model_path):
        result = review_frames(images, fps, max_frames, batch_size, device, top_k,
                               policy_mode, custom_policy, review_threshold, model_path)
        return (*result, _selected_policy(policy_mode, custom_policy)["text"])


NODE_CLASS_MAPPINGS = {"ShieldGemma2PolicySignals": ShieldGemma2PolicySignals}
NODE_DISPLAY_NAME_MAPPINGS = {"ShieldGemma2PolicySignals": "ShieldGemma 2 政策逐帧检测"}
