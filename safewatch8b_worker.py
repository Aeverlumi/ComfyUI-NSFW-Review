"""Isolated SafeWatch inference entry point for the legacy InternVL runtime."""

import json
import sys
from pathlib import Path

from PIL import Image

from safewatch8b_nodes import _infer_batches_direct


def main(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    batches = [[Image.open(path).convert("RGB") for path in batch]
               for batch in request["frame_batches"]]
    result = _infer_batches_direct(
        batches, Path(request["folder"]), request["prompt"], request["device"],
        request["max_new_tokens"], request["cpu_threads"],
    )
    Path(request["output"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
