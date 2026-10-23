"""Model locations and integrity checks shared by the installer and ComfyUI nodes."""

import hashlib
import json
import os
from pathlib import Path


MODEL_FILES = {
    "opennsfw2": ("open_nsfw_weights.h5",),
    "falconsai": ("config.json", "preprocessor_config.json", "model.safetensors"),
    "blue_eye": ("config.json", "model.safetensors", "modeling.py"),
}

KNOWN_SHA256 = {
    ("opennsfw2", "open_nsfw_weights.h5"): "14ca261f48bdd88c1eecba96a761bd1579b523adae1b749b0a4ffd8b7ed8babe",
    ("falconsai", "config.json"): "cd8c0a19566f199b5c64da402f1d6a9daf80d094fe6536befb060a2cc59449c2",
    ("falconsai", "preprocessor_config.json"): "a34861a24e781942424ad790b82bc99348404a2e64ea882de88c40905851698d",
    ("falconsai", "model.safetensors"): "97b2ce64ec146884b37f98ee7944ca4891aa72f6827dc0cb10684a1cbecd5830",
    ("blue_eye", "config.json"): "b58e7e7957554959fdef4caad46c490a1c80d7e38c568856d10ce131b81cb458",
    ("blue_eye", "modeling.py"): "3fdf2349fcb4df71a380626bf5f532ebd768a1dfa34a327525c5e3ae0948283f",
    ("blue_eye", "model.safetensors"): "da7e592b912549b6f97bff7939d80aa2d867e869f0200e3901643cea9816677d",
}


def models_root(explicit=None):
    if explicit:
        return Path(explicit).expanduser().resolve()
    override = os.environ.get("META_NSFW_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    try:
        import folder_paths
    except ImportError as exc:
        raise RuntimeError(
            "Set META_NSFW_MODEL_DIR or pass --models-dir outside ComfyUI"
        ) from exc
    return Path(folder_paths.models_dir).resolve() / "meta_nsfw_review"


def model_dir(model, root=None):
    if model not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model}")
    return models_root(root) / model


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model, root=None):
    folder = model_dir(model, root)
    missing = [name for name in MODEL_FILES[model]
               if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            f"{model} is incomplete in {folder}; missing: {', '.join(missing)}. "
            "Run install_models.py for this model."
        )
    manifest_path = folder / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != model:
            raise ValueError(f"Wrong model manifest in {folder}")
    for name in MODEL_FILES[model]:
        known = KNOWN_SHA256.get((model, name))
        expected = manifest.get("sha256", {}).get(name) if manifest else None
        if known or expected:
            actual = sha256(folder / name)
            if known and actual != known:
                raise ValueError(f"Official model checksum mismatch: {folder / name}")
            if expected and actual != expected:
                raise ValueError(f"Checksum mismatch: {folder / name}")
    return folder


def write_manifest(model, folder, source):
    manifest = {
        "schema_version": 1,
        "model": model,
        "source": source,
        "sha256": {name: sha256(folder / name) for name in MODEL_FILES[model]},
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
