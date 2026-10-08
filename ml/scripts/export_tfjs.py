"""Sprint 4C: i-convert ang .keras model sa TensorFlow.js (graph model) para sa website.

Paraan: .keras -> TF SavedModel (model.export) -> TF.js graph model. Hindi ginagamit ang direktang
Keras -> TF.js "layers model", kasi madalas pumalya iyon sa Keras 3.

Sa Windows, hindi ma-pip install nang buo ang tensorflowjs (kailangan nito ang uvloop, jax at
tensorflow_decision_forests, na walang Windows build). Kaya ganito ang install (hindi nito ginagalaw ang TensorFlow):
  pip install --no-deps tensorflowjs==4.22.0 tensorflow-hub==0.16.1 tf-keras==2.21.0
at pinapalitan ng stub ang jax at tensorflow_decision_forests (sa import lang sila ginagamit ng converter).

Mga output:
  ml/models/tfjs/<model>_<quantize>/   model.json + group1-shard*.bin + class_names.json (ito ang ilalagay sa website)
  ml/models/tfjs/<model>_verify/       test images at hula ng Keras, para sa verify_tfjs.cjs

Usage:
  python ml/scripts/export_tfjs.py --model ml/models/efficientnetb0_v4_finetuned.keras
  python ml/scripts/export_tfjs.py --model ml/models/efficientnetb0_v4_finetuned.keras --quantize none
"""
import argparse
import json
import shutil
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

import common
import predict

FUSED_OPS = {"_FusedConv2D", "FusedDepthwiseConv2dNative"}


def stub_unused_imports():
    """In-import ng tensorflowjs ang jax (para sa JAX models) at tensorflow_decision_forests (para sa TF-DF ops).
    Hindi kailangan ang mga iyon sa pag-convert ng SavedModel, kaya stub kapag wala."""
    try:
        import tensorflow_decision_forests  # noqa: F401
    except ImportError:
        sys.modules["tensorflow_decision_forests"] = types.ModuleType("tensorflow_decision_forests")
    try:
        import jax  # noqa: F401
    except ImportError:
        jax, exp = types.ModuleType("jax"), types.ModuleType("jax.experimental")
        exp.jax2tf = types.ModuleType("jax.experimental.jax2tf")
        jax.experimental = exp
        sys.modules.update({"jax": jax, "jax.experimental": exp, "jax.experimental.jax2tf": exp.jax2tf})


def flatten_fused_bias(model_json: Path) -> int:
    """Ang BatchNorm ng Keras 3 ay nagiging bias na [1, 1, 1, C] ng _FusedConv2D. Rank-1 lang ang tinatanggap ng
    WASM backend ("FusedConv2D only supports rank-1 bias"), kaya gawing [C]. Shape lang ang nagbabago, hindi ang data."""
    mj = json.loads(model_json.read_text(encoding="utf-8"))
    nodes = mj["modelTopology"]["node"]
    uses = defaultdict(list)
    for n in nodes:
        for k, inp in enumerate(n.get("input", [])):
            uses[inp.split(":")[0].lstrip("^")].append((n["op"], k))
    fixed = set()
    for group in mj["weightsManifest"]:
        for w in group["weights"]:
            u = uses.get(w["name"], [])
            if len(w["shape"]) == 4 and w["shape"][:3] == [1, 1, 1] and u and all(
                op in FUSED_OPS and k == 2 for op, k in u
            ):
                w["shape"] = w["shape"][3:]
                fixed.add(w["name"])
    for n in nodes:
        if n["name"] in fixed:
            n["attr"]["value"]["tensor"]["tensorShape"]["dim"] = n["attr"]["value"]["tensor"]["tensorShape"]["dim"][3:]
    model_json.write_text(json.dumps(mj), encoding="utf-8")
    return len(fixed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--quantize", choices=["none", "float16", "uint8"], default="float16",
                    help="float16: kalahati ang laki, halos walang epekto sa hula (i-check gamit ang verify_tfjs.cjs)")
    ap.add_argument("--verify-samples", type=int, default=300, help="ilang test images ang isasama sa verify data (0 = wala)")
    args = ap.parse_args()

    model = common.load_trained_model(args.model)
    class_names = common.load_class_names()
    common.check_class_count(model, class_names)
    tfjs_dir = common.MODELS_DIR / "tfjs"
    out = tfjs_dir / f"{args.model.stem}_{args.quantize}"
    for old in out.glob("group*-shard*.bin"):  # lumang export: baka iba ang bilang ng shards
        old.unlink()

    stub_unused_imports()
    from tensorflowjs.converters import tf_saved_model_conversion_v2 as conversion

    with tempfile.TemporaryDirectory() as saved_model_dir:
        model.export(saved_model_dir, format="tf_saved_model", verbose=False)
        conversion.convert_tf_saved_model(
            saved_model_dir, str(out), signature_def="serving_default", saved_model_tags="serve",
            quantization_dtype_map=None if args.quantize == "none" else {args.quantize: True},
        )
    print(f"Bias na ginawang rank-1 (para sa WASM backend): {flatten_fused_bias(out / 'model.json')}")
    shutil.copy(common.CLASS_NAMES_PATH, out / "class_names.json")
    size_mb = sum(p.stat().st_size for p in out.iterdir()) / (1024 * 1024)
    print(f"\nTF.js model: {out} ({size_mb:.1f} MB)")

    if args.verify_samples > 0:
        paths = predict.gather_images(common.DATASET_DIR / "test")
        paths = paths[:: max(1, len(paths) // args.verify_samples)][: args.verify_samples]
        size = common.model_img_size(model)
        x = np.stack([predict.load_image(p, size) for p in paths])  # parehong preprocessing ng predict.py
        probs = model.predict(x, verbose=0)
        verify = tfjs_dir / f"{args.model.stem}_verify"
        verify.mkdir(parents=True, exist_ok=True)
        (verify / "inputs.u8").write_bytes(x.astype(np.uint8).tobytes())
        (verify / "reference.json").write_text(json.dumps({
            "shape": list(x.shape), "class_names": class_names,
            "files": [p.name for p in paths], "true": [p.parent.name for p in paths],
            "keras_probs": probs.round(6).tolist(),
        }), encoding="utf-8")
        print(f"Verify data ({len(paths)} test images): {verify}")
        print(f"Susunod: NODE_PATH=<folder>/node_modules node ml/scripts/verify_tfjs.cjs {out} {verify}")


if __name__ == "__main__":
    main()
