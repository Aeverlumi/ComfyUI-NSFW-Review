"""Optional image classifier nodes. Imports of model frameworks happen only on use."""

import importlib.util
import json
import math

from model_store import verify_model


def _sample_indices(count, limit):
    if count < 1:
        raise ValueError("The IMAGE batch is empty")
    if limit == 0 or limit >= count:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return [round(i * (count - 1) / (limit - 1)) for i in range(limit)]


def _pil_frames(images, indices):
    import numpy as np
    from PIL import Image

    frames = []
    for index in indices:
        frame = images[index]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        pixels = np.asarray(frame)
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError(f"Frame {index} must be RGB or RGBA in NHWC format")
        pixels = np.clip(pixels[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(pixels))
    return frames


def _torch_device(preference):
    import torch
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return "cuda" if preference == "auto" and torch.cuda.is_available() else (
        "cpu" if preference == "auto" else preference
    )


def _opennsfw2(frames, folder, batch_size, preference):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("OpenNSFW2 needs requirements-opennsfw2.txt") from exc
    gpus = tf.config.list_physical_devices("GPU")
    if preference == "cuda" and not gpus:
        raise RuntimeError("CUDA was requested but TensorFlow cannot see a GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # TensorFlow was already initialized by another ComfyUI node.
    try:
        import opennsfw2
    except ImportError as exc:
        raise RuntimeError("OpenNSFW2 needs requirements-opennsfw2.txt") from exc
    selected_device = "/CPU:0" if preference == "cpu" or not gpus else "/GPU:0"
    with tf.device(selected_device):
        scores = opennsfw2.predict_images(
            frames, batch_size=batch_size, weights_path=str(folder / "open_nsfw_weights.h5")
        )
    return [{"sfw": 1.0 - float(score), "nsfw": float(score)} for score in scores]


def _falconsai(frames, folder, batch_size, preference):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError("Falconsai needs requirements-torch.txt") from exc
    device = _torch_device(preference)
    processor = AutoImageProcessor.from_pretrained(str(folder), local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(
        str(folder), local_files_only=True, use_safetensors=True
    ).to(device).eval()
    results = []
    try:
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                inputs = processor(images=frames[start:start + batch_size], return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu().tolist()
                labels = model.config.id2label
                results.extend({str(labels[i]).lower(): float(value) for i, value in enumerate(row)}
                               for row in probabilities)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def _blue_eye(frames, folder, batch_size, preference):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Blue-Eye needs requirements-torch.txt") from exc
    module_path = folder / "modeling.py"
    spec = importlib.util.spec_from_file_location("meta_nsfw_blue_eye_modeling", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = _torch_device(preference)
    model = module.BlueEyeClassifier.from_pretrained(str(folder), device=device)
    results = []
    try:
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                pixels = module.preprocess(frames[start:start + batch_size], model.config).to(device)
                probabilities = model.predict_proba(pixels).cpu().tolist()
                results.extend(dict(zip(model.labels, map(float, row))) for row in probabilities)
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


BACKENDS = {
    "opennsfw2": _opennsfw2,
    "falconsai": _falconsai,
    "blue_eye": _blue_eye,
}


MODEL_CLASSES = {
    "opennsfw2": ("sfw", "nsfw"),
    "falconsai": ("normal", "nsfw"),
    "blue_eye": ("safe", "suggestive", "explicit"),
}


def _predicted_label(probabilities):
    return max(probabilities, key=probabilities.get)


def _label_counts(frames, model):
    counts = {label: 0 for label in MODEL_CLASSES[model]}
    for frame in frames:
        label = frame.get("predicted_label") or _predicted_label(frame["probabilities"])
        counts[label] += 1
    return counts


def classify_batch(model, images, fps, max_frames, batch_size, device, top_k, root=None):
    if fps <= 0 or batch_size < 1 or max_frames < 0 or top_k < 1:
        raise ValueError("fps and batch_size/top_k must be positive; max_frames must be nonnegative")
    folder = verify_model(model, root)
    count = len(images)
    indices = _sample_indices(count, max_frames)
    frames = _pil_frames(images, indices)
    raw = BACKENDS[model](frames, folder, batch_size, device)
    if len(raw) != len(indices):
        raise RuntimeError(f"{model} returned {len(raw)} results for {len(indices)} frames")
    records = []
    for index, probabilities in zip(indices, raw):
        if not probabilities or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1
                                    for value in probabilities.values()):
            raise ValueError(f"{model} returned invalid probabilities for frame {index}")
        if model == "blue_eye":
            signal = probabilities.get("suggestive", 0.0) + probabilities.get("explicit", 0.0)
        else:
            signal = probabilities.get("nsfw", 0.0)
        records.append({
            "frame_index": index,
            "timestamp_seconds": round(index / fps, 3),
            "probabilities": probabilities,
            "predicted_label": _predicted_label(probabilities),
            "signal": round(float(signal), 6),
        })
    highest = sorted(records, key=lambda item: item["signal"], reverse=True)[:top_k]
    manifest_path = folder / "manifest.json"
    hashes = json.loads(manifest_path.read_text(encoding="utf-8")).get("sha256", {}) if manifest_path.is_file() else {}
    weights_name = "open_nsfw_weights.h5" if model == "opennsfw2" else "model.safetensors"
    report = {
        "schema_version": "meta-image-signals-v1",
        "model": model,
        "model_sha256": hashes.get(weights_name),
        "signal_definition": "suggestive_plus_explicit" if model == "blue_eye" else "nsfw",
        "input_frames": count,
        "sampled_frames": len(records),
        "label_counts": _label_counts(records, model),
        "fps_assumption": fps,
        "max_signal": highest[0]["signal"],
        "top_frames": highest,
        "frames": records,
        "limitations": "Image-model scores are uncalibrated for Meta ad approval. Timestamps assume the input batch is evenly spaced at fps_assumption.",
    }
    return json.dumps(report, ensure_ascii=False), report["max_signal"]


class _BaseImageSignal:
    MODEL = None
    CATEGORY = "video/review/image signals"
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("signals_json", "max_auxiliary_signal")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 14.0, "min": 0.001, "max": 240.0}),
            "max_frames": ("INT", {"default": 48, "min": 0, "max": 100000}),
            "batch_size": ("INT", {"default": 4, "min": 1, "max": 128}),
            "device": (["auto", "cuda", "cpu"],),
            "top_k": ("INT", {"default": 8, "min": 1, "max": 100}),
        }}

    def run(self, images, fps, max_frames, batch_size, device, top_k):
        return classify_batch(self.MODEL, images, fps, max_frames, batch_size, device, top_k)


class OpenNSFW2ImageSignals(_BaseImageSignal):
    MODEL = "opennsfw2"


class FalconsaiImageSignals(_BaseImageSignal):
    MODEL = "falconsai"


class BlueEyeImageSignals(_BaseImageSignal):
    MODEL = "blue_eye"


class CompareImageSignals:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {
            "opennsfw2_json": ("STRING", {"forceInput": True}),
            "falconsai_json": ("STRING", {"forceInput": True}),
            "blue_eye_json": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_json",)
    FUNCTION = "compare"
    CATEGORY = "video/review/image signals"

    def compare(self, opennsfw2_json=None, falconsai_json=None, blue_eye_json=None):
        supplied = {
            "opennsfw2": opennsfw2_json,
            "falconsai": falconsai_json,
            "blue_eye": blue_eye_json,
        }
        reports = {}
        for model, raw in supplied.items():
            if raw is None:
                continue
            report = json.loads(raw)
            if report.get("schema_version") != "meta-image-signals-v1" or report.get("model") != model:
                raise ValueError(f"Expected {model} meta-image-signals-v1 JSON")
            reports[model] = report
        if not reports:
            raise ValueError("Connect at least one image signal node")
        signatures = {
            (report["input_frames"], report["fps_assumption"],
             tuple(frame["frame_index"] for frame in report["frames"]))
            for report in reports.values()
        }
        result = {
            "schema_version": "meta-image-comparison-v1",
            "sampling_aligned": len(signatures) == 1,
            "models": {
                model: {
                    "model_sha256": report.get("model_sha256"),
                    "signal_definition": report["signal_definition"],
                    "sampled_frames": report["sampled_frames"],
                    "label_counts": report.get("label_counts") or _label_counts(report["frames"], model),
                    "max_signal": report["max_signal"],
                    "top_frames": report["top_frames"],
                }
                for model, report in reports.items()
            },
            "review_frame_indices": sorted({
                frame["frame_index"]
                for report in reports.values() for frame in report["top_frames"]
            }),
            "limitations": "Model scores are not comparable across models and do not predict Meta approval.",
        }
        return (json.dumps(result, ensure_ascii=False),)


MODEL_LABELS = {
    "opennsfw2": "OpenNSFW2",
    "falconsai": "Falconsai NSFW",
    "blue_eye": "Blue-Eye",
}
PROBABILITY_LABELS = {
    "sfw": "安全", "normal": "正常", "nsfw": "NSFW",
    "safe": "安全", "suggestive": "暗示性", "explicit": "露骨",
}


def _percent(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1.000001:
        raise ValueError("Model signal must be a finite value between 0 and 1")
    return f"{number * 100:.1f}%"


def _time(seconds):
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Frame timestamp must be nonnegative")
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def _frame_line(frame):
    probabilities = frame["probabilities"]
    predicted = frame.get("predicted_label") or _predicted_label(probabilities)
    classes = "，".join(
        f"{name}（{PROBABILITY_LABELS.get(name, name)}）{_percent(value)}"
        for name, value in probabilities.items()
    )
    return (f"{_time(frame['timestamp_seconds'])} | 第 {frame['frame_index']} 帧 | "
            f"预测类别 {predicted} | {classes}")


def _counts_line(counts, model):
    return "，".join(f"{label} {counts.get(label, 0)} 帧" for label in MODEL_CLASSES[model])


def format_image_signals_report(raw):
    data = json.loads(raw)
    schema = data.get("schema_version")
    if schema == "meta-image-signals-v1":
        model = data["model"]
        if model not in MODEL_LABELS:
            raise ValueError(f"Unknown image signal model: {model}")
        top = data["top_frames"]
        if not top:
            raise ValueError("Image signal report has no sampled frames")
        counts = data.get("label_counts") or _label_counts(data["frames"], model)
        lines = [
            f"{MODEL_LABELS[model]} 视频帧检测报告",
            "=" * 32,
            f"检测范围：输入 {data['input_frames']} 帧，实际抽查 {data['sampled_frames']} 帧；"
            f"输入帧率按 {data['fps_assumption']} fps 估算。",
            "原生预测类别统计（逐帧取最高概率类别）：" + _counts_line(counts, model),
            "", "供复核的画面（由工作流辅助分数选取，不是作者的视频评级）",
        ]
        lines.extend(f"{index}. {_frame_line(frame)}" for index, frame in enumerate(top, 1))
        lines.extend([
            "", "结果说明",
            "类别和百分比来自模型逐帧输出；工作流只负责抽帧、选取复核画面和排版。",
            "原始 JSON 中的 signal/max_signal 是工作流的辅助排序值，不是作者默认的视频结论。",
            "这里的百分比不是检测准确率，也不是 Meta 过审概率；不能跨模型平均或投票。",
            "PG/PG13/R/X/XXX 是人工分级，不由图片模型自动换算。",
            "本报告只覆盖传入节点的画面；若视频加载器限制了帧数，未加载部分没有接受检测。",
            "时间按输入帧率估算；若视频经过跳帧或截取，请回原视频核对画面。",
        ])
    elif schema == "meta-image-comparison-v1":
        models = data["models"]
        if not models:
            raise ValueError("Image comparison has no models")
        lines = ["图片模型原生类别对照报告", "=" * 32,
                 "采样对齐：" + ("是" if data["sampling_aligned"] else "否；时间点不能直接对应"),
                 "", "各模型逐帧预测类别统计"]
        for name, result in models.items():
            if name not in MODEL_LABELS or not result["top_frames"]:
                raise ValueError(f"Invalid comparison entry: {name}")
            counts = result.get("label_counts")
            if counts:
                lines.append(f"- {MODEL_LABELS[name]}：{_counts_line(counts, name)}")
            else:
                lines.append(f"- {MODEL_LABELS[name]}：旧版对照 JSON 未保存完整类别统计")
        lines.extend(["", "供复核的画面（由工作流辅助分数选取）"])
        for name, result in models.items():
            lines.append(f"{MODEL_LABELS[name]}：")
            lines.extend(f"  {index}. {_frame_line(frame)}"
                         for index, frame in enumerate(result["top_frames"], 1))
        lines.extend([
            "", "结果说明",
            "每行的预测类别是该模型概率最高的原生类别；选帧排序由工作流完成。",
            "模型类别定义不同；意见不一致时，请对照原视频逐帧复核。",
            "本对照只覆盖传入节点的画面；若视频加载器限制了帧数，未加载部分没有接受检测。",
            "百分比不是检测准确率，也不能预测 Meta 是否过审或自动换算人工 PG/PG13/R/X/XXX。",
        ])
    else:
        raise ValueError("Expected meta-image-signals-v1 or meta-image-comparison-v1 JSON")
    return "\n".join(lines) + "\n"


class FormatImageSignalsReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"signals_json": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("readable_report",)
    FUNCTION = "format"
    CATEGORY = "video/review/image signals"

    def format(self, signals_json):
        return (format_image_signals_report(signals_json),)


NODE_CLASS_MAPPINGS = {
    "OpenNSFW2ImageSignals": OpenNSFW2ImageSignals,
    "FalconsaiImageSignals": FalconsaiImageSignals,
    "BlueEyeImageSignals": BlueEyeImageSignals,
    "CompareImageSignals": CompareImageSignals,
    "FormatImageSignalsReport": FormatImageSignalsReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenNSFW2ImageSignals": "OpenNSFW2 逐帧信号",
    "FalconsaiImageSignals": "Falconsai 逐帧信号",
    "BlueEyeImageSignals": "Blue-Eye 逐帧信号",
    "CompareImageSignals": "图片模型信号对照",
    "FormatImageSignalsReport": "检测结果易读报告",
}
