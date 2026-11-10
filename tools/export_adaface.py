"""Export the pinned official CVLFace AdaFace IR50 RGB artifact, without fetching anything.

Invoke only after the owner approves the checkpoint's training-data/license conditions and the model-work window.
The code is MIT; the official model card separately requires following the training dataset's license.

Source/checkpoint are from minchul/cvlface_adaface_ir50_webface4m revision
60a65befbcf7e19284c4f3ac730f56867ed29594. Required local files:
  models/iresnet/model.py (bare IR50; no wrapper channel flip), pretrained_model/model.pt.
The source is loaded after its hash is checked, without trust_remote_code or changing sys.path. Its unused
benchmark-only fvcore import is omitted; all network definitions remain byte-for-byte upstream.

  tools/test-slot.sh python tools/export_adaface.py --source /tmp/adaface/model.py \
    --checkpoint /tmp/adaface/model.pt --output /tmp/adaface_ir50_webface4m_rgb.onnx

The ONNX artifact takes fixed float32 RGB NCHW [1,3,112,112], scaled by /127.5-1. It emits a raw 512-dimensional
feature; the runtime normalizes it. Export and parity verification use CPU only with one thread, never CUDA.
No gallery/state is read or changed. Keep the publisher's MIT code notice alongside the downloaded source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from liberational.vision.faces import ADAFACE_INPUT_FORMAT

REVISION = "60a65befbcf7e19284c4f3ac730f56867ed29594"
CHECKPOINT_SHA256 = "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"
SOURCE_SHA256 = "e31be55c60b538c15151887e911b7535e2f0e1114a13427ba06483e1cd2a63f9"


def checked(path: Path, expected: str) -> None:
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected checksum for {path.name}; the pinned artifact is required.")


def network_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """The official IResNet wrapper contributes only its net; never load classifier or unrelated weights."""
    if not state or any(not key.startswith("net.") for key in state):
        raise ValueError("Expected the pinned official IR50 net-prefixed state dictionary.")
    return {key.removeprefix("net."): value for key, value in state.items()}


def export(source: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    checked(source, SOURCE_SHA256)
    checked(checkpoint, CHECKPOINT_SHA256)
    import onnx
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # This import is used only by upstream's guarded __main__ FLOP benchmark, not by IR50 inference.
    # Hash-check first, then load an explicitly adjusted temporary copy with the normal Python loader.
    with tempfile.TemporaryDirectory() as source_directory:
        adjusted = Path(source_directory) / "ir50.py"
        adjusted.write_text(source.read_text().replace("from fvcore.nn import flop_count\n", ""))
        spec = importlib.util.spec_from_file_location("official_adaface_ir50", adjusted)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the verified official IR50 source.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    model = module.IR_50(input_size=(112, 112), output_dim=512).eval()
    if model.flip:
        raise ValueError("The bare RGB IR50 export must not flip channels internally.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(network_state(state), strict=True)
    sample = torch.linspace(-1, 1, 3 * 112 * 112).reshape(1, 3, 112, 112)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        artifact = Path(temporary) / "model.onnx"
        with torch.inference_mode():
            expected = np.asarray(model(sample).numpy(), dtype=np.float32)
            torch.onnx.export(
                model,
                (sample,),
                str(artifact),
                input_names=["input"],
                output_names=["embedding"],
                opset_version=17,
                dynamo=False,
                external_data=False,
            )
        graph = onnx.load(artifact)
        onnx.helper.set_model_props(
            graph,
            {
                "face_input": ADAFACE_INPUT_FORMAT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "source_sha256": SOURCE_SHA256,
                "upstream_revision": REVISION,
            },
        )
        onnx.checker.check_model(graph)
        onnx.save(graph, artifact)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
        actual = np.asarray(session.run(["embedding"], {"input": sample.numpy()})[0], dtype=np.float32)
        if expected.shape != (1, 512) or not np.isfinite(expected).all():
            raise ValueError("The pinned IR50 did not produce finite [1,512] features.")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        error = float(np.max(np.abs(actual - expected)))
        artifact.replace(output)
    return {
        "output": str(output),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_sha256": SOURCE_SHA256,
        "input": ADAFACE_INPUT_FORMAT,
        "opset": 17,
        "cpu_max_absolute_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
