"""Generate standalone and combined ComfyUI GUI/API video review workflows."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).parent
MODELS = {
    "opennsfw2": ("OpenNSFW2ImageSignals", "OpenNSFW2", "cpu", 4),
    "falconsai": ("FalconsaiImageSignals", "Falconsai", "auto", 4),
    "blue_eye": ("BlueEyeImageSignals", "Blue-Eye", "auto", 2),
}


def video_node(cap):
    return {
        "id": 1, "type": "VHS_LoadVideo", "pos": [40, 120], "size": [430, 750],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"name": "meta_batch", "shape": 7, "type": "VHS_BatchManager", "link": None},
            {"name": "vae", "shape": 7, "type": "VAE", "link": None},
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "frame_count", "type": "INT", "links": None},
            {"name": "audio", "type": "AUDIO", "links": None},
            {"name": "video_info", "type": "VHS_VIDEOINFO", "links": None},
        ],
        "properties": {"Node name for S&R": "VHS_LoadVideo"},
        "widgets_values": {
            "video": "example.mp4", "force_rate": 14, "custom_width": 0,
            "custom_height": 448, "frame_load_cap": cap,
            "skip_first_frames": 0, "select_every_nth": 1, "format": "None",
            "videopreview": {"hidden": False, "paused": False, "params": {
                "force_rate": 14, "custom_width": 0, "custom_height": 448,
                "frame_load_cap": cap, "skip_first_frames": 0,
                "select_every_nth": 1, "filename": "example.mp4",
                "type": "input", "format": "video/mp4",
            }},
        },
        "title": "视频输入 | 14 fps",
    }


def model_node(node_id, model, frame_limit, top_k, y):
    node_type, label, device, batch_size = MODELS[model]
    return {
        "id": node_id, "type": node_type, "pos": [550, y], "size": [365, 260],
        "flags": {}, "order": node_id, "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": None},
            *({"name": name, "type": kind, "widget": {"name": name}}
              for name, kind in (("fps", "FLOAT"), ("max_frames", "INT"),
                                 ("batch_size", "INT"), ("device", "COMBO"),
                                 ("top_k", "INT"))),
        ],
        "outputs": [
            {"name": "signals_json", "type": "STRING", "links": []},
            {"name": "max_auxiliary_signal", "type": "FLOAT", "links": None},
        ],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": [14, frame_limit, batch_size, device, top_k],
        "title": label + " | 逐帧信号",
    }


def save_node(node_id, prefix, y, label):
    return {
        "id": node_id, "type": "SaveText", "pos": [1330, y],
        "size": [380, 170], "flags": {}, "order": node_id, "mode": 0,
        "inputs": [
            {"name": "text", "type": "STRING", "link": None},
            {"name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}},
            {"name": "format", "type": "COMBO", "widget": {"name": "format"}},
        ],
        "outputs": [{"name": "text", "type": "STRING", "links": None}],
        "properties": {"Node name for S&R": "SaveText"},
        "widgets_values": ["", "video_review/" + prefix, "txt"],
        "title": "保存 " + label,
    }


def format_node(node_id, y):
    return {
        "id": node_id, "type": "FormatImageSignalsReport", "pos": [950, y],
        "size": [350, 100], "flags": {}, "order": node_id, "mode": 0,
        "inputs": [{"name": "signals_json", "type": "STRING", "link": None}],
        "outputs": [{"name": "readable_report", "type": "STRING", "links": []}],
        "properties": {"Node name for S&R": "FormatImageSignalsReport"},
        "widgets_values": [], "title": "整理中文检测报告",
    }


def compare_node(node_id, y):
    return {
        "id": node_id, "type": "CompareImageSignals", "pos": [550, y],
        "size": [365, 170], "flags": {}, "order": node_id, "mode": 0,
        "inputs": [
            {"name": name + "_json", "type": "STRING", "link": None}
            for name in MODELS
        ],
        "outputs": [{"name": "comparison_json", "type": "STRING", "links": []}],
        "properties": {"Node name for S&R": "CompareImageSignals"},
        "widgets_values": [], "title": "三模型帧信号对照",
    }


def gui_workflow(selected):
    combined = len(selected) > 1
    limit = 8 if combined else 48
    nodes = [video_node(8 if combined else 0)]
    links = []
    model_cards = []

    def add(node):
        nodes.append(node)
        return node

    def connect(source, output, target, input_slot, kind):
        link_id = len(links) + 1
        links.append([link_id, source["id"], output, target["id"], input_slot, kind])
        source["outputs"][output]["links"].append(link_id)
        target["inputs"][input_slot]["link"] = link_id

    for index, model in enumerate(selected):
        y = 80 + index * 390
        suffix = "combined" if combined else "individual"
        model_card = add(model_node(len(nodes) + 1, model, limit,
                                    4 if combined else 8, y))
        formatter = add(format_node(len(nodes) + 1, y))
        readable = add(save_node(len(nodes) + 1, f"{model}_{suffix}_report", y,
                                 model + " 中文报告"))
        raw = add(save_node(len(nodes) + 1, f"{model}_{suffix}_signals", y + 185,
                            model + " 原始 JSON"))
        model_cards.append(model_card)
        connect(nodes[0], 0, model_card, 0, "IMAGE")
        connect(model_card, 0, formatter, 0, "STRING")
        connect(formatter, 0, readable, 0, "STRING")
        connect(model_card, 0, raw, 0, "STRING")
    if combined:
        y = 80 + len(selected) * 390
        compare = add(compare_node(len(nodes) + 1, y))
        formatter = add(format_node(len(nodes) + 1, y))
        readable = add(save_node(len(nodes) + 1, "model_comparison_report", y,
                                 "三模型对照中文报告"))
        raw = add(save_node(len(nodes) + 1, "model_comparison", y + 185,
                            "三模型对照原始 JSON"))
        for index, model_card in enumerate(model_cards):
            connect(model_card, 0, compare, index, "STRING")
        connect(compare, 0, formatter, 0, "STRING")
        connect(formatter, 0, readable, 0, "STRING")
        connect(compare, 0, raw, 0, "STRING")
    return {
        "last_node_id": max(node["id"] for node in nodes),
        "last_link_id": len(links), "nodes": nodes, "links": links,
        "groups": [], "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [80, 60]}},
        "version": 0.4,
    }


def shieldgemma2_video_workflow():
    video = video_node(0)
    model = {
        "id": 2, "type": "ShieldGemma2PolicySignals", "pos": [550, 120],
        "size": [430, 550], "flags": {}, "order": 1, "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": 1},
            *({"name": name, "type": kind, "widget": {"name": name}}
              for name, kind in (("fps", "FLOAT"), ("max_frames", "INT"),
                                 ("batch_size", "INT"), ("device", "COMBO"),
                                 ("policy_mode", "COMBO"), ("custom_policy", "STRING"),
                                 ("review_threshold", "FLOAT"), ("top_k", "INT"),
                                 ("model_path", "STRING"))),
        ],
        "outputs": [
            {"name": "signals_json", "type": "STRING", "links": [3]},
            {"name": "readable_report", "type": "STRING", "links": [2]},
            {"name": "max_yes_probability", "type": "FLOAT", "links": None},
            {"name": "policy_text", "type": "STRING", "links": None},
        ],
        "properties": {"Node name for S&R": "ShieldGemma2PolicySignals"},
        "widgets_values": [14, 48, 1, "auto", "official_sexual", "", 0.5, 8, ""],
        "title": "ShieldGemma 2 | 官方 sexual 政策",
    }
    report = save_node(3, "shieldgemma2_report", 120, "ShieldGemma 2 中文报告")
    raw = save_node(4, "shieldgemma2_signals", 340, "ShieldGemma 2 原始 JSON")
    video["outputs"][0]["links"] = [1]
    report["inputs"][0]["link"] = 2
    raw["inputs"][0]["link"] = 3
    return {
        "last_node_id": 4,
        "last_link_id": 3,
        "nodes": [video, model, report, raw],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 1, 3, 0, "STRING"],
                  [3, 2, 0, 4, 0, "STRING"]],
        "groups": [], "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [80, 60]}},
        "version": 0.4,
    }


def safewatch8b_video_workflow():
    video = video_node(0)
    model = {
        "id": 2, "type": "SafeWatch8BVideoReview", "pos": [550, 120],
        "size": [430, 360], "flags": {}, "order": 1, "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": 1},
            *({"name": name, "type": kind, "widget": {"name": name}}
              for name, kind in (("fps", "FLOAT"), ("max_frames", "INT"),
                                 ("device", "COMBO"), ("policy_mode", "COMBO"),
                                 ("custom_policy", "STRING"), ("max_new_tokens", "INT"),
                                 ("cpu_threads", "INT"), ("model_path", "STRING"))),
        ],
        "outputs": [
            {"name": "review_json", "type": "STRING", "links": [3]},
            {"name": "readable_report", "type": "STRING", "links": [2]},
            {"name": "raw_response", "type": "STRING", "links": None},
        ],
        "properties": {"Node name for S&R": "SafeWatch8BVideoReview"},
        "widgets_values": [14, 48, "auto", "original_categories", "", 256, 8, ""],
        "title": "SafeWatch-8B | 视频审核",
    }
    report = save_node(3, "safewatch8b_report", 120, "SafeWatch 中文报告")
    raw = save_node(4, "safewatch8b_review", 340, "SafeWatch 原始 JSON")
    video["outputs"][0]["links"] = [1]
    report["inputs"][0]["link"] = 2
    raw["inputs"][0]["link"] = 3
    return {
        "last_node_id": 4, "last_link_id": 3,
        "nodes": [video, model, report, raw],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 1, 3, 0, "STRING"],
                  [3, 2, 0, 4, 0, "STRING"]],
        "groups": [], "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [80, 60]}},
        "version": 0.4,
    }


def api_workflow(gui):
    api = {}
    for node in gui["nodes"]:
        values = node["widgets_values"]
        if node["type"] == "VHS_LoadVideo":
            inputs = {name: values[name] for name in (
                "video", "force_rate", "custom_width", "custom_height",
                "frame_load_cap", "skip_first_frames", "select_every_nth", "format"
            )}
        elif node["type"] == "SaveText":
            link = gui["links"][node["inputs"][0]["link"] - 1]
            inputs = {"text": [str(link[1]), link[2]],
                      "filename_prefix": values[1], "format": values[2]}
        else:
            inputs = {}
            for slot, item in enumerate(node["inputs"]):
                if item.get("link") is not None:
                    link = gui["links"][item["link"] - 1]
                    inputs[item["name"]] = [str(link[1]), link[2]]
                elif "widget" in item:
                    widget_index = sum("widget" in previous for previous in node["inputs"][:slot])
                    inputs[item["name"]] = values[widget_index]
        api[str(node["id"])] = {"class_type": node["type"], "inputs": inputs,
                                "_meta": {"title": node["title"]}}
    return api


def main():
    for model in (*MODELS, "all"):
        selected = tuple(MODELS) if model == "all" else (model,)
        gui = gui_workflow(selected)
        stem = model + "_video"
        (HERE / (stem + "_gui.json")).write_text(
            json.dumps(gui, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (HERE / (stem + "_api.json")).write_text(
            json.dumps(api_workflow(gui), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gui = shieldgemma2_video_workflow()
    for suffix, value in (("gui", gui), ("api", api_workflow(gui))):
        (HERE / f"shieldgemma2_video_{suffix}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gui = safewatch8b_video_workflow()
    for suffix, value in (("gui", gui), ("api", api_workflow(gui))):
        (HERE / f"safewatch8b_video_{suffix}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
