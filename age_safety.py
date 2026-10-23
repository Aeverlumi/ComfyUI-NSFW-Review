"""Normalize model age cues without treating visual appearance as verified age."""

import json


SIGNALS = {"no_obvious_signal", "possible_minor", "unclear", "no_person"}


def normalize_age_signal(raw_text):
    decoder = json.JSONDecoder()
    result = None
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            result = candidate
            break
    if result is None:
        result = {}
    signal = result.get("age_signal")
    if signal not in SIGNALS:
        signal = "unclear"
    evidence = str(result.get("evidence") or "")[:300]
    if signal == "no_obvious_signal":
        evidence = "未见明确儿童线索；年龄仍未核实。"
    elif signal == "no_person":
        evidence = "模型未发现人物；请核对原图。"
    elif signal == "unclear":
        evidence = "画面或模型输出不足以判断年龄线索。"
    return {
        "schema_version": "age-safety-signal-v1",
        "model_signal": signal,
        "age_verification": "unverified",
        "child_safety_review_candidate": signal in {"possible_minor", "unclear"},
        "evidence": evidence,
        "limitations": "A visual model cannot verify anyone's age; obtain reliable age evidence before use when required.",
    }


class AgeSafetyNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"raw_text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("age_signal_json",)
    FUNCTION = "normalize"
    CATEGORY = "review/sexual content"

    def normalize(self, raw_text):
        return (json.dumps(normalize_age_signal(raw_text), ensure_ascii=False),)


NODE_CLASS_MAPPINGS = {"AgeSafetyNormalize": AgeSafetyNormalize}
NODE_DISPLAY_NAME_MAPPINGS = {"AgeSafetyNormalize": "年龄线索校验 | 非年龄认证"}
