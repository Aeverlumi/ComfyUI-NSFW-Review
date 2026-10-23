"""Install official model files into ComfyUI/models/meta_nsfw_review."""

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from model_store import MODEL_FILES, KNOWN_SHA256, model_dir, models_root, sha256, verify_model, write_manifest


SOURCES = {
    "opennsfw2": "https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5",
    "falconsai": "https://huggingface.co/Falconsai/nsfw_image_detection/resolve/main/",
    "blue_eye": "https://huggingface.co/pp1618/Blue-Eye",
}


def _fetch(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-NSFW-Review/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(destination, "wb") as target:
        shutil.copyfileobj(response, target)
    if destination.stat().st_size == 0:
        raise ValueError(f"Downloaded empty file from {url}")


def install(model, root, force=False):
    target = model_dir(model, root)
    if not force:
        try:
            return verify_model(model, root)
        except FileNotFoundError:
            pass
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{model}-", dir=root) as staging_name:
        staging = Path(staging_name)
        if model == "opennsfw2":
            _fetch(SOURCES[model], staging / MODEL_FILES[model][0])
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError("Install requirements-torch.txt before downloading Hugging Face models") from exc
            try:
                snapshot_download(
                    "Falconsai/nsfw_image_detection" if model == "falconsai" else "pp1618/Blue-Eye",
                    local_dir=staging,
                    allow_patterns=list(MODEL_FILES[model]),
                )
            except Exception as exc:
                if model == "blue_eye":
                    raise RuntimeError(
                        "Blue-Eye download failed. Accept its license on Hugging Face and log in "
                        "with `hf auth login` on this machine."
                    ) from exc
                raise RuntimeError("Falconsai download failed; check Hugging Face access") from exc
        for name in MODEL_FILES[model]:
            if not (staging / name).is_file() or (staging / name).stat().st_size == 0:
                raise FileNotFoundError(f"Upstream did not provide {name} for {model}")
            known = KNOWN_SHA256.get((model, name))
            if known and sha256(staging / name) != known:
                raise ValueError(f"Official model checksum mismatch after download: {model}/{name}")
        target.mkdir(parents=True, exist_ok=True)
        for name in MODEL_FILES[model]:
            (staging / name).replace(target / name)
        write_manifest(model, target, SOURCES[model])
    return verify_model(model, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_FILES, "all"), default="all")
    parser.add_argument("--models-dir", type=Path, help="Full meta_nsfw_review model directory")
    parser.add_argument("--comfyui-root", type=Path, help="ComfyUI installation directory")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace managed model files")
    args = parser.parse_args(argv)
    if args.models_dir and args.comfyui_root:
        parser.error("Use either --models-dir or --comfyui-root")
    root = models_root(args.models_dir or (args.comfyui_root / "models" / "meta_nsfw_review" if args.comfyui_root else None))
    selected = MODEL_FILES if args.model == "all" else (args.model,)
    failed = False
    for model in selected:
        try:
            folder = verify_model(model, root) if args.verify_only else install(model, root, args.force)
            print(f"{model}: ready at {folder}")
        except Exception as exc:
            failed = True
            print(f"{model}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
