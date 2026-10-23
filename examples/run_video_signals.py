"""Submit the example video workflow to a running ComfyUI and wait for results."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Video filename already present in ComfyUI/input")
    parser.add_argument("--model", choices=("opennsfw2", "falconsai", "blue_eye", "all"),
                        default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:18188")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--full-video", action="store_true", help="Decode the whole video and sample up to 48 frames")
    args = parser.parse_args()
    workflow = json.loads((Path(__file__).parent / (args.model + "_video_api.json")).read_text())
    workflow["1"]["inputs"]["video"] = args.video
    if args.full_video:
        workflow["1"]["inputs"]["frame_load_cap"] = 0
        for node in workflow.values():
            if node["class_type"].endswith("ImageSignals"):
                node["inputs"].update(max_frames=48, top_k=8)
    url = args.base_url.rstrip("/")
    payload = json.dumps({"prompt": workflow}).encode()
    request = urllib.request.Request(
        url + "/prompt", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            queued = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(exc.read().decode()) from exc
    prompt_id = queued["prompt_id"]
    print(f"Queued {prompt_id}", flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        history = get_json(url + "/history/" + prompt_id)
        if prompt_id in history:
            result = history[prompt_id]
            status = result.get("status", {})
            if status.get("status_str") == "error":
                raise SystemExit(json.dumps(status, ensure_ascii=False))
            if status.get("completed"):
                for node_id, output in sorted(result.get("outputs", {}).items()):
                    if "text" not in output:
                        continue
                    report_text = output["text"][0]
                    filenames = [entry["filename"] for entry in output.get("files", [])]
                    if workflow[node_id]["class_type"] == "SaveText" and "report" in workflow[node_id]["inputs"]["filename_prefix"]:
                        print("\n" + report_text, flush=True)
                    print("保存文件：" + ", ".join("video_review/" + name for name in filenames), flush=True)
                return
        time.sleep(2)
    raise SystemExit(f"Timed out after {args.timeout}s; prompt_id={prompt_id}")


if __name__ == "__main__":
    main()
