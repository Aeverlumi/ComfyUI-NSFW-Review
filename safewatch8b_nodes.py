"""Video-level SafeWatch-8B review from a decoded ComfyUI IMAGE batch."""

import ast
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

from image_models import _pil_frames, _sample_indices, _torch_device
from model_store import models_root


MODEL_ID = "Virtue-AI-HUB/SafeWatch-8B"
CATEGORIES = (
    "C1(Sexual Content)", "C2(Harassment & Bullying)",
    "C3(Threats, Violence & Harm)", "C4(False & Deceptive Information)",
    "C5(Illegal/Regulated Activities)", "C6(Hateful Content & Extremism)",
)

# Exact video_moderation_prompt_vg_fine_grained_final from the SafeWatch authors (MIT).
# https://github.com/BillChan226/SafeWatch/blob/main/utility/prompt_template_new.py
ORIGINAL_POLICY = """You are a helpful video guardrail model to detect if a video contains unsafe, harmful content that needs to be moderated.

**As a guardrail model, your tasks are:**

**1. Summarize the content of the video, beginning with "DESCRIPTION:".**
**2. Analyze the video for the presence of the following harmful content categories:**

<BEGIN HARMFUL CATEGORIES>
C1: Sexual Content; C2: Harassment & Bullying; C3: Threats, Violence & Harm; C4: False & Deceptive Information; C5: Illegal/Regulated Activities; C6: Hateful Content & Extremism;
<END HARMFUL CATEGORIES>

**Refer to the detailed explanations of each category below:**

<BEGIN HARMFUL CATEGORIES EXPLANATIONS>
<box>
C1: Sexual Content:
Core Value: Protect users, especially minors, from sexual exploitation and unwanted exposure to sexually explicit content.
[BLOCKED] Content involving sexual exploitation or abuse of minors, including child sexual abuse imagery, should be flagged.
[BLOCKED] Content containing pornography or sexually explicit material primarily intended for arousal should be flagged.
[BLOCKED] Content promoting sexual services and sexual implications should be flagged.
[ALLOWED] Non-sexual contexts (e.g. breastfeeding, art), dancing, gymnastics, live performances, and sports are allowed.
</box>
<box>
C2: Harassment & Bullying:
Core Value: Foster a safe and respectful environment where users are protected from harassment, intimidation, and privacy violations.
[BLOCKED] No bullying, harassment including sexual, animal, child abuse, cyberbullying, and campus bullying.
[BLOCKED] No sharing others' private info or images of them in private spaces without consent.
[BLOCKED] No potential abuse and cruelty to animals, child, and students including speech and actions.
</box>
<box>
C3: Threats, Violence & Harm:
Core Value: Prevent the promotion and glorification of violence, self-harm, and dangerous activities to ensure user safety.
[BLOCKED] Content involving violence, dangerous behavior such as fighting, shooting, vandalism, assault that result in injury or destruction of property.
[BLOCKED] Content intending to cause harm or violence to the audience or other people should be flagged.
[BLOCKED] Content depicting graphic violence, or dangerous activities that could incite harm or violence should be flagged.
</box>
<box>
C4: False & Deceptive Information:
Core Value: Maintain a trustworthy platform by combating the spread of false information, deception, and fraudulent activities.
[BLOCKED] Content spreading false information that causes harm or is malicious, such as denying the existence of tragic events, unsubstantiated medical claims.
[BLOCKED] Content spreading misinformation, out-of-date facts, alternation of videos, AI-generated content, or other false information should be flagged.
[BLOCKED] Content that involve individuals acting to deceive or manipulate others, such as impersonation, false identity, or false claims, should be flagged.
</box>
<box>
C5: Illegal/Regulated Activities:
Core Value: Comply with legal requirements and prevent the promotion of illegal activities to protect users and society at large.
[BLOCKED] Content promoting illegal activity like selling drugs, weapons, or exploitation should be flagged.
[BLOCKED] Content containing unauthorized promotion of regulated goods like gambling, alcohol, or tobacco should be flagged.
[BLOCKED] Content involving regulated activities such as arson, explosion, robbery, shoplifting should be flagged.
[BLOCKED] Content involving war scenes, military actions, terrorism actions, heresy activities should be flagged.
</box>
<box>
C6: Hateful Content & Extremism:
Core Value: Stand against hatred, discrimination, and extremism to create an inclusive community that respects diversity.
[BLOCKED] Content involving extremely disturbing content, such as torture, gore, mutilation, or self-harm, should be flagged.
[BLOCKED] Content inciting anti-social, depression, self-harm, or suicide should be flagged.
</box>
<END HARMFUL CATEGORIES EXPLANATIONS>

**Provide the moderation result in the following JSON format:**

GUARDRAIL = {{
"C1(Sexual Content)": boolean,
"C2(Harassment & Bullying)": boolean,
"C3(Threats, Violence & Harm)": boolean,
"C4(False & Deceptive Information)": boolean,
"C5(Illegal/Regulated Activities)": boolean,
"C6(Hateful Content & Extremism)": boolean
}}

**Set the value to true ONLY for the most relevant categories found in the video. If no harmful content is detected, set all values to false.**
**The first line must be the summarization the content of the video, beginning with "DESCRIPTION:"**
**The second line must be the JSON format of "GUARDRAIL", beginning with "GUARDRAIL:"**
**For each category flagged as true, provide a brief explanation of why it was flagged and list the specific policy violations within that category in the third line, beginning with "EXPLANATION:"**

"""

SEXUAL_POLICY = """Review the entire video for sexual content and sexual implication. Distinguish ordinary clothing, fashion, dancing and non-sexual contact from deliberately sexualized posing, framing, interaction, nudity or sexual activity. Judge temporal context, not one isolated pose. Do not infer concealed body parts or actions. Start the answer with GUARDRAIL: followed by a complete JSON object with the single boolean key "C1(Sexual Content)". Then describe concrete visible evidence and any uncertainty. This is a content signal, not a Meta ad approval decision."""


def _model_folder(model_path):
    folder = Path(model_path).expanduser().resolve() if model_path.strip() else models_root() / "safewatch8b"
    if not (folder / "config.json").is_file() or len(list(folder.glob("model-*-of-*.safetensors"))) != 4:
        raise FileNotFoundError(f"SafeWatch-8B is incomplete at {folder}; download {MODEL_ID} first")
    return folder


def _policy(mode, custom_policy):
    if mode == "original_categories":
        return ORIGINAL_POLICY
    if mode == "sexual_focus":
        return SEXUAL_POLICY
    if mode == "custom" and custom_policy.strip():
        return custom_policy.strip()
    raise ValueError("Select a policy mode or provide custom policy text")


def _parse_guardrail(response):
    marker = response.find("GUARDRAIL")
    search = response[marker:] if marker >= 0 else response
    start = search.find("{")
    if start < 0:
        return None
    for end in range(start + 1, len(search)):
        if search[end] != "}":
            continue
        candidate = search[start:end + 1]
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(result, dict) and result and all(
                isinstance(k, str) and isinstance(v, bool) for k, v in result.items()
            ):
                return result
    return None


def _load_tokenizer(folder):
    from transformers import AutoTokenizer

    source = folder / "tokenizer.model"
    if hashlib.sha256(source.read_bytes()).hexdigest() != (
        "f868398fc4e05ee1e8aeba95ddf18ddcc45b8bce55d5093bead5bbf80429b48b"
    ):
        return AutoTokenizer.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True)

    # SentencePiece 0.2.2 rejects the original NUL byte token. Keep all IDs stable.
    from sentencepiece import sentencepiece_model_pb2

    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(source.read_bytes())
    affected = [piece for piece in proto.pieces if chr(0) in piece.piece]
    if len(affected) != 1 or affected[0].piece != chr(0):
        raise RuntimeError("Unexpected SafeWatch tokenizer format")
    affected[0].piece = "<safewatch_null_piece>"
    with tempfile.TemporaryDirectory(prefix="safewatch-tokenizer-") as temp_dir:
        replacement = Path(temp_dir) / "tokenizer.model"
        replacement.write_bytes(proto.SerializeToString())
        return AutoTokenizer.from_pretrained(
            str(folder), trust_remote_code=True, local_files_only=True,
            vocab_file=str(replacement),
        )


def _infer_batches_direct(frame_batches, folder, prompt, device,
                          max_new_tokens=256, cpu_threads=8):
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel

    started = time.perf_counter()
    selected_device = _torch_device(device)
    if selected_device == "cpu":
        torch.set_num_threads(cpu_threads)
    else:
        free_before, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16 if selected_device == "cuda" else torch.float32
    transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tokenizer = _load_tokenizer(folder)
    model = AutoModel.from_pretrained(
        str(folder), trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).eval().to(selected_device)
    try:
        responses = []
        for frames in frame_batches:
            pixels = torch.stack([transform(frame) for frame in frames]).to(
                selected_device, dtype=dtype)
            prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(frames)))
            with torch.inference_mode():
                responses.append(model.chat(
                    tokenizer, pixels, prefix + prompt,
                    {"num_beams": 1, "max_new_tokens": max_new_tokens, "do_sample": False},
                    num_patches_list=[1] * len(frames),
                ))
            del pixels
        metrics = {"device_used": selected_device,
                   "worker_elapsed_seconds": round(time.perf_counter() - started, 2)}
        if selected_device == "cuda":
            metrics.update({
                "cuda_free_before_gib": round(free_before / 2**30, 2),
                "cuda_total_gib": round(total / 2**30, 2),
                "worker_peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "worker_peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            })
        return {"responses": responses, "metrics": metrics}
    finally:
        del model
        if selected_device == "cuda":
            torch.cuda.empty_cache()


def _infer(frame_batches, folder, prompt, device, max_new_tokens=256, cpu_threads=8):
    worker_python = Path(os.environ.get(
        "SAFEWATCH_PYTHON", str(Path(__file__).parent / ".venv-safewatch" / "bin" / "python")
    ))
    if not worker_python.is_file():
        raise FileNotFoundError(
            f"SafeWatch isolated Python is missing: {worker_python}. See README setup."
        )
    with tempfile.TemporaryDirectory(prefix="safewatch-review-") as temp_dir:
        work = Path(temp_dir)
        batches = []
        for batch_index, frames in enumerate(frame_batches):
            paths = []
            for frame_index, frame in enumerate(frames):
                path = work / f"frame_{batch_index:02d}_{frame_index:02d}.png"
                frame.save(path)
                paths.append(str(path))
            batches.append(paths)
        output = work / "response.json"
        request = work / "request.json"
        request.write_text(json.dumps({
            "frame_batches": batches, "folder": str(folder), "prompt": prompt,
            "device": device, "output": str(output),
            "max_new_tokens": max_new_tokens, "cpu_threads": cpu_threads,
        }), encoding="utf-8")
        completed = subprocess.run(
            [str(worker_python), str(Path(__file__).parent / "safewatch8b_worker.py"), str(request)],
            capture_output=True, text=True, timeout=900, check=False,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        if completed.returncode != 0:
            raise RuntimeError(f"SafeWatch worker failed: {completed.stderr[-3000:]}")
        return json.loads(output.read_text(encoding="utf-8"))


def _window_starts(count):
    if count <= 8:
        return [0]
    starts = list(range(0, count - 7, 6))
    final = count - 8
    if starts[-1] != final:
        starts.append(final)
    return starts


def _review_indices(images, max_frames):
    count = len(images)
    if count <= max_frames:
        return list(range(count))
    if max_frames < 16:
        return _sample_indices(count, max_frames)

    import numpy as np

    array = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    step_y = max(1, array.shape[1] // 32)
    step_x = max(1, array.shape[2] // 32)
    thumbnails = array[:, ::step_y, ::step_x, :3].astype(np.float32)
    motion = np.abs(np.diff(thumbnails, axis=0)).mean(axis=(1, 2, 3))
    selected = set(_sample_indices(count, max_frames // 2))
    for index in np.argsort(-motion, kind="stable"):
        selected.add(int(index) + 1)
        if len(selected) == max_frames:
            break
    return sorted(selected)


def review_video(images, fps, max_frames, device, policy_mode, custom_policy,
                 model_path="", max_new_tokens=256, cpu_threads=8, infer=None):
    if not math.isfinite(fps) or fps <= 0 or max_frames < 1 or max_frames > 48:
        raise ValueError("fps must be positive and max_frames must be 1..48")
    if not 16 <= max_new_tokens <= 512 or not 1 <= cpu_threads <= 64:
        raise ValueError("max_new_tokens must be 16..512 and cpu_threads must be 1..64")
    prompt = _policy(policy_mode, custom_policy)
    started = time.perf_counter()
    indices = _review_indices(images, max_frames)
    frames = _pil_frames(images, indices)
    starts = _window_starts(len(frames))
    frame_batches = [frames[start:start + 8] for start in starts]
    folder = _model_folder(model_path) if infer is None else None
    result = (infer or (lambda *args: _infer(*args, max_new_tokens, cpu_threads)))(
        frame_batches, folder, prompt, device)
    responses = result["responses"]
    if len(responses) != len(starts) or not all(isinstance(value, str) for value in responses):
        raise ValueError("SafeWatch returned an invalid number of segment responses")
    segments = []
    for start, response in zip(starts, responses):
        window_indices = indices[start:start + 8]
        segments.append({
            "frame_indices": window_indices,
            "timestamps_seconds": [round(i / fps, 3) for i in window_indices],
            "guardrail": _parse_guardrail(response),
            "raw_response": response,
        })
    labels = {}
    for category in CATEGORIES:
        values = [segment["guardrail"].get(category) if segment["guardrail"] else None
                  for segment in segments]
        labels[category] = True if True in values else (None if None in values else False)
    sexual = labels["C1(Sexual Content)"]
    raw_response = "\n\n".join(
        f"Segment {index + 1} ({segment['timestamps_seconds'][0]}-"
        f"{segment['timestamps_seconds'][-1]}s):\n{segment['raw_response']}"
        for index, segment in enumerate(segments))
    report = {
        "schema_version": "safewatch8b-video-review-v2",
        "model": MODEL_ID,
        "policy_mode": policy_mode,
        "policy_text": prompt,
        "input_frames": len(images),
        "sampled_frames": len(indices),
        "sampling_strategy": "all_frames" if len(images) <= max_frames else (
            "hybrid_coverage_motion" if max_frames >= 16 else "uniform"),
        "sampled_timestamps_seconds": [round(i / fps, 3) for i in indices],
        "segment_count": len(segments),
        "segments": segments,
        "guardrail": labels,
        "sexual_content_flag": sexual,
        "raw_response": raw_response,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "performance": result.get("metrics", {}),
        "limitations": "Overlapping eight-frame model judgments on sampled silent frames; category summary flags any positive segment. No calibrated probability, audio analysis, or Meta approval prediction.",
    }
    verdict = "无法解析，需人工复核" if sexual is None else ("有性内容信号，需复核" if sexual else "未检出性内容信号")
    summary = ", ".join(f"{category.split('(')[0]}={labels[category]}" for category in CATEGORIES)
    metrics = report["performance"]
    memory = (f"，子进程 CUDA 峰值保留 {metrics['worker_peak_cuda_reserved_gib']} GiB"
              if "worker_peak_cuda_reserved_gib" in metrics else "")
    readable = (f"SafeWatch-8B 视频审核\n模式：{policy_mode}\n取样：{len(indices)}/{len(images)} 帧，"
                f"{len(segments)} 段\n性内容：{verdict}\n六类汇总：{summary}\n"
                f"耗时：{report['elapsed_seconds']} 秒{memory}\n"
                f"模型原文：\n{raw_response}\n限制：{report['limitations']}")
    return json.dumps(report, ensure_ascii=False), readable, raw_response


class SafeWatch8BVideoReview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 4.0, "min": 0.001, "max": 240.0}),
            "max_frames": ("INT", {"default": 48, "min": 1, "max": 48}),
            "device": (["auto", "cuda", "cpu"],),
            "policy_mode": (["original_categories", "sexual_focus", "custom"],),
            "custom_policy": ("STRING", {"default": "", "multiline": True}),
            "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 512}),
            "cpu_threads": ("INT", {"default": 8, "min": 1, "max": 64}),
            "model_path": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("review_json", "readable_report", "raw_response")
    FUNCTION = "run"
    CATEGORY = "video/review"

    def run(self, images, fps, max_frames, device, policy_mode, custom_policy,
            max_new_tokens, cpu_threads, model_path):
        return review_video(images, fps, max_frames, device, policy_mode, custom_policy,
                            model_path, max_new_tokens, cpu_threads)


NODE_CLASS_MAPPINGS = {"SafeWatch8BVideoReview": SafeWatch8BVideoReview}
NODE_DISPLAY_NAME_MAPPINGS = {"SafeWatch8BVideoReview": "SafeWatch-8B 视频审核"}
