"""Versioned, reproducible sexual-content risk index for Qwen and Blue-Eye."""

import json
import math


RATING_ANCHORS = {"PG": 0, "PG13": 45, "R": 70, "X": 90, "XXX": 100}
VISUAL_WEIGHT = 0.55
BLUE_EYE_WEIGHT = 0.35
PROMPT_WEIGHT = 0.10
SUGGESTIVE_WEIGHT = 0.35
EXPLICIT_WEIGHT = 1.0


def _object(raw, name):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _probabilities(frame):
    values = frame.get("probabilities")
    if not isinstance(values, dict) or set(values) != {"safe", "suggestive", "explicit"}:
        raise ValueError("Blue-Eye frame must contain safe, suggestive, and explicit probabilities")
    probabilities = {}
    for label, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid Blue-Eye {label} probability")
        probabilities[label] = float(value)
    if abs(sum(probabilities.values()) - 1) > 0.01:
        raise ValueError("Blue-Eye probabilities must sum to one")
    return probabilities


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _blue_eye_summary(raw, media_type):
    report = _object(raw, "blue_eye_json")
    if report.get("schema_version") != "meta-image-signals-v1" or report.get("model") != "blue_eye":
        raise ValueError("Expected a Blue-Eye meta-image-signals-v1 report")
    frames = report.get("frames")
    if not isinstance(frames, list) or not frames or report.get("sampled_frames") != len(frames):
        raise ValueError("Blue-Eye report must contain its sampled frames")
    if media_type == "image" and len(frames) != 1:
        raise ValueError("Image review requires exactly one Blue-Eye frame")

    frame_scores = []
    counts = {"safe": 0, "suggestive": 0, "explicit": 0}
    records = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid Blue-Eye frame")
        probabilities = _probabilities(frame)
        label = max(probabilities, key=probabilities.get)
        counts[label] += 1
        score = 100 * (SUGGESTIVE_WEIGHT * probabilities["suggestive"] + probabilities["explicit"])
        frame_scores.append(score)
        records.append({
            "frame_index": frame.get("frame_index"),
            "timestamp_seconds": frame.get("timestamp_seconds"),
            "label": label,
            "probabilities": probabilities,
            "risk_component": round(score, 4),
        })

    peak = max(frame_scores)
    p90 = _percentile(frame_scores, 0.9)
    aggregate = frame_scores[0] if media_type == "image" else 0.7 * peak + 0.3 * p90
    summary = {
        "model": "blue_eye",
        "model_sha256": report.get("model_sha256"),
        "input_frames": report.get("input_frames"),
        "sampled_frames": len(records),
        "label_counts": counts,
        "aggregation": "single_image" if media_type == "image" else "0.70*max_frame+0.30*p90_frame",
        "max_frame_risk_component": round(peak, 4),
        "p90_frame_risk_component": round(p90, 4),
        "risk_component": round(aggregate, 4),
        "top_frames": sorted(records, key=lambda item: item["risk_component"], reverse=True)[:8],
    }
    if media_type == "image":
        summary["probabilities"] = records[0]["probabilities"]
        summary["predicted_label"] = records[0]["label"]
    return summary


def build_risk_report(visual_json, prompt_json, blue_eye_json, media_type, age_json=None):
    if media_type not in ("image", "video"):
        raise ValueError("media_type must be image or video")
    visual = _object(visual_json, "visual_json")
    prompt = _object(prompt_json, "prompt_json")
    blue_eye = _blue_eye_summary(blue_eye_json, media_type)
    age_safety = None
    if age_json is not None:
        age_safety = _object(age_json, "age_json")
        if age_safety.get("schema_version") != "age-safety-signal-v1":
            raise ValueError("Expected an age-safety-signal-v1 report")
        if age_safety.get("age_verification") != "unverified":
            raise ValueError("Visual age signals cannot verify age")
    visual_rating = visual.get("rating", "UNCERTAIN")
    prompt_rating = prompt.get("rating", "UNCERTAIN")
    if visual_rating not in (*RATING_ANCHORS, "UNCERTAIN") or prompt_rating not in (*RATING_ANCHORS, "UNCERTAIN"):
        raise ValueError("Qwen ratings must use PG, PG13, R, X, XXX, or UNCERTAIN")

    complete = visual_rating != "UNCERTAIN" and prompt_rating != "UNCERTAIN"
    visual_anchor = RATING_ANCHORS.get(visual_rating)
    prompt_anchor = RATING_ANCHORS.get(prompt_rating)
    blue_component = blue_eye["risk_component"]
    score = None
    if complete:
        score = math.floor(VISUAL_WEIGHT * visual_anchor + BLUE_EYE_WEIGHT * blue_component
                           + PROMPT_WEIGHT * prompt_anchor + 0.5)
    disagreement = complete and visual_rating != prompt_rating
    attention = []
    if visual_rating in ("R", "X", "XXX"):
        attention.append("qwen_visual_r_or_higher")
    if blue_eye["label_counts"]["explicit"]:
        attention.append("blue_eye_explicit_frame")
    if visual_rating == "PG13" or blue_eye["label_counts"]["suggestive"]:
        attention.append("suggestive_content")
    if disagreement:
        attention.append("qwen_visual_prompt_disagreement")
    if not complete:
        attention.append("uncertain_qwen_rating")
    if age_safety and age_safety.get("child_safety_review_candidate"):
        attention.append("age_safety_model_flag_requires_human_review")

    report = {
        "schema_version": "sexual-approval-risk-v1",
        "media_type": media_type,
        "risk_score": score,
        "score_range": [0, 100],
        "score_direction": "higher_is_harder_to_approve",
        "score_scope": "sexual_content_only",
        "meta_approval_probability_percent": None,
        "formula": {
            "rating_anchors": RATING_ANCHORS,
            "blue_eye_frame": "100*(0.35*p_suggestive+1.00*p_explicit)",
            "blue_eye_video": "0.70*max_frame+0.30*p90_frame",
            "total": "round_half_up(0.55*qwen_visual+0.35*blue_eye+0.10*qwen_prompt)",
        },
        "components": {
            "qwen_visual_rating": visual_rating,
            "qwen_visual_anchor": visual_anchor,
            "blue_eye_risk_component": blue_component,
            "qwen_prompt_rating": prompt_rating,
            "qwen_prompt_anchor": prompt_anchor,
            "weights": {"qwen_visual": VISUAL_WEIGHT, "blue_eye": BLUE_EYE_WEIGHT,
                        "qwen_prompt": PROMPT_WEIGHT},
        },
        "blue_eye": blue_eye,
        "qwen_visual_evidence": visual.get("evidence", []),
        "qwen_prompt_evidence": prompt.get("evidence", []),
        "qwen_rating_disagreement": bool(disagreement),
        "attention_signals": attention,
        "age_safety": age_safety,
        "release_gate": "hold_for_age_verification" if age_safety and age_safety.get("child_safety_review_candidate")
                        else "standard_review",
        "review_required": bool(attention) or visual.get("review_required") is not False
                           or prompt.get("review_required") is not False,
        "limitations": [
            "The score is a fixed rubric, not a measured Meta approval probability.",
            "Blue-Eye class probabilities are not Meta approval probabilities.",
            "Prompt rating comes from a description of the same media and is not independent evidence.",
            "Ad text, landing page, targeting, account status, and actual Meta review are not included.",
        ],
    }
    if media_type == "video":
        report["limitations"].append("Only sampled frames were reviewed; Blue-Eye timestamps assume the input fps.")
    return report


def readable_report(report):
    components = report["components"]
    blue_eye = report["blue_eye"]
    score = "待复核" if report["risk_score"] is None else f"{report['risk_score']}/100"
    lines = [
        "性内容过审风险报告 v1",
        f"风险分：{score}（越高越难过审；不是 Meta 过审概率）",
        f"Qwen 画面评级：{components['qwen_visual_rating']}；提示词评级：{components['qwen_prompt_rating']}",
        f"Blue-Eye：safe {blue_eye['label_counts']['safe']}，suggestive {blue_eye['label_counts']['suggestive']}，"
        f"explicit {blue_eye['label_counts']['explicit']}（共 {blue_eye['sampled_frames']} 帧）",
    ]
    if report["media_type"] == "image":
        probabilities = blue_eye["probabilities"]
        lines.append("单图概率：" + "，".join(f"{key} {value * 100:.2f}%" for key, value in probabilities.items()))
    else:
        lines.append("Blue-Eye 视频分量：最高帧 " + f"{blue_eye['max_frame_risk_component']:.2f}，"
                     + f"第90百分位 {blue_eye['p90_frame_risk_component']:.2f}")
    if report["age_safety"]:
        lines.append("年龄线索：" + report["age_safety"]["model_signal"]
                     + "（未核实年龄；模型提示不能证明成年或未成年）")
    lines.extend([
        f"复核：{'需要' if report['review_required'] else '未触发'}",
        "关注信号：" + ("、".join(report["attention_signals"]) or "无"),
        "", "Qwen 画面证据：", str(report["qwen_visual_evidence"]),
        "", "Qwen 提示词证据：", str(report["qwen_prompt_evidence"]),
        "", "说明：本分数仅衡量性内容风险；实际是否过审须以 Meta 审核为准。",
    ])
    return "\n".join(lines)


class SexualApprovalRiskV1:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "visual_json": ("STRING", {"forceInput": True}),
            "prompt_json": ("STRING", {"forceInput": True}),
            "blue_eye_json": ("STRING", {"forceInput": True}),
            "media_type": (["image", "video"],),
        }, "optional": {"age_json": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report_json", "readable_report")
    FUNCTION = "build"
    CATEGORY = "review/sexual content"

    def build(self, visual_json, prompt_json, blue_eye_json, media_type, age_json=None):
        report = build_risk_report(visual_json, prompt_json, blue_eye_json, media_type, age_json)
        return json.dumps(report, ensure_ascii=False), readable_report(report)


NODE_CLASS_MAPPINGS = {"SexualApprovalRiskV1": SexualApprovalRiskV1}
NODE_DISPLAY_NAME_MAPPINGS = {"SexualApprovalRiskV1": "性内容过审风险分 | Qwen + Blue-Eye"}
