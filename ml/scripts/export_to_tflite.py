"""I-convert ang PyTorch ONNX model (galing sa export_to_onnx.py) sa TensorFlow Lite.

Paraan: onnx2tf (ONNX -> TF SavedModel -> .tflite). Ginagawa nitong NHWC ang input, kaya [1, 224, 224, 3] ang
tinatanggap ng TFLite model imbes na [1, 3, 224, 224] ng ONNX.

Input ng TFLite model: RGB float32, 224x224, NORMALIZED tulad ng val_transforms ng train_v2.py:
  (pixel / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
Hindi raw 0-255 (iba sa mga Keras model). Output: 6 logits (walang softmax), sunod sa class_names.json.

Kailangan sa venv: pip install onnx2tf onnxruntime sng4onnx psutil

Mga output (ml/models/tflite/):
  <onnx name>_float32.tflite
  <onnx name>_float16.tflite   kalahati ang laki, halos walang epekto sa hula (tingnan ang verify sa dulo)
  class_names.json

Usage:
  python ml/scripts/export_to_tflite.py
  python ml/scripts/export_to_tflite.py --onnx ml/scripts/eggplant_model.onnx --verify-samples 0
"""
import argparse
import contextlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx

import common
import predict

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Sample images na ginagamit ng onnx2tf para ikumpara ang ONNX at TF habang nagko-convert. Hinahanap muna nito ang
# file na ito sa cwd bago mag-download (at pumapalya ang download), kaya sariling test images ang ibinibigay.
SAMPLE_NPY = "calibration_image_sample_data_20x128x128x3_float32.npy"


def preprocess(path: Path) -> np.ndarray:
    """NHWC [224, 224, 3], parehong normalization ng val_transforms sa train_v2.py."""
    return ((predict.load_image(path) / 255.0 - MEAN) / STD).astype(np.float32)


def test_images(n: int) -> list[Path]:
    paths = predict.gather_images(common.DATASET_DIR / "test")
    return paths[:: max(1, len(paths) // n)][:n]


def convert(onnx_path: Path, out: Path) -> list[Path]:
    """Pinapalitan ng onnx2tf ang mismong ONNX file na binigay (op names), at hiwalay sa .onnx.data ang weights
    ng model. Kaya isang buong kopya sa temp folder ang kino-convert, hindi ang orihinal."""
    import onnx2tf

    onnx_path, out = onnx_path.resolve(), out.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work_onnx = Path(tmp) / onnx_path.name  # parehong pangalan, para <stem>_float32.tflite ang output
        onnx.save(onnx.load(onnx_path), work_onnx)  # isinasama ang external data sa iisang file
        np.save(Path(tmp) / SAMPLE_NPY, np.stack([preprocess(p) for p in test_images(20)]))
        with contextlib.chdir(tmp):
            onnx2tf.convert(
                input_onnx_file_path=str(work_onnx), output_folder_path=str(Path(tmp) / "out"),
                not_use_onnxsim=True,  # wala ang onnxsim sa venv; naka-fold na ang BatchNorm ng PyTorch exporter
                copy_onnx_input_output_names_to_tflite=True, verbosity="error",
            )
        out.mkdir(parents=True, exist_ok=True)
        return [Path(shutil.copy(f, out / f.name)) for f in sorted((Path(tmp) / "out").glob("*.tflite"))]


def run_tflite(path: Path, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    logits = []
    for img in x:  # batch 1 ang TFLite model
        interp.set_tensor(inp["index"], img[None])
        interp.invoke()
        logits.append(interp.get_tensor(outp["index"])[0])
    return np.stack(logits)


def verify(onnx_path: Path, tflite_paths: list[Path], class_names, samples: int):
    """Ikumpara ang TFLite sa ONNX Runtime gamit ang test images: logits diff, parehong hula, at accuracy."""
    import onnxruntime as ort

    paths = test_images(samples)
    x = np.stack([preprocess(p) for p in paths])  # NHWC
    true = np.array([class_names.index(p.parent.name) for p in paths])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: img[None].transpose(0, 3, 1, 2)})[0] for img in x])
    print(f"\nVerify sa {len(paths)} test images")
    print(f"  {'ONNX (reference)':<36} accuracy {np.mean(ref.argmax(1) == true) * 100:5.1f}%")
    for t in tflite_paths:
        logits = run_tflite(t, x)
        print(f"  {t.name:<36} accuracy {np.mean(logits.argmax(1) == true) * 100:5.1f}% | "
              f"parehong hula sa ONNX {np.mean(logits.argmax(1) == ref.argmax(1)) * 100:5.1f}% | "
              f"max |logit diff| {np.abs(logits - ref).max():.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, default=Path(__file__).resolve().parent / "eggplant_model.onnx")
    ap.add_argument("--out", type=Path, default=common.MODELS_DIR / "tflite")
    ap.add_argument("--verify-samples", type=int, default=300, help="ilang test images ang gagamitin sa verify (0 = wala)")
    args = ap.parse_args()

    class_names = common.load_class_names()
    n_out = onnx.load(args.onnx, load_external_data=False).graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    if n_out != len(class_names):
        raise SystemExit(f"{n_out} classes ang ONNX model pero {len(class_names)} ang nasa "
                         f"{common.CLASS_NAMES_PATH.name} ({class_names}).")

    tflite_paths = convert(args.onnx, args.out)
    if not tflite_paths:
        raise SystemExit("Walang .tflite na nagawa ang onnx2tf; tingnan ang error sa itaas.")
    shutil.copy(common.CLASS_NAMES_PATH, args.out / "class_names.json")
    for t in tflite_paths:
        print(f"TFLite model: {t} ({t.stat().st_size / (1024 * 1024):.1f} MB)")

    if args.verify_samples > 0:
        verify(args.onnx, tflite_paths, class_names, args.verify_samples)


if __name__ == "__main__":
    main()
