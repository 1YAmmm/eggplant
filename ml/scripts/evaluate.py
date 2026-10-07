"""Sprint 2 + 3: i-evaluate ang model sa TEST set.

Mga output (sa ml/results/, prefix = pangalan ng model file):
  <name>_report.txt              accuracy, precision/recall/F1 bawat class, macro F1
  <name>_confusion_matrix.png    confusion matrix (counts at normalized)
  <name>_metrics.json            para sa comparison table
  <name>_misclassified.csv       listahan ng maling hula (para linisin ang dataset)

Usage:
  python ml/scripts/evaluate.py --model-path ml/models/mobilenetv2_baseline.keras
  python ml/scripts/evaluate.py --compare mobilenetv2_finetuned efficientnetb0_finetuned
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import common
from common import keras, tf

RECALL_TARGET = 0.80
ACCURACY_TARGET = 0.85


def measure_speed(model, ds, n_images=50):
    """Average ms bawat image (batch size 1, CPU). May warmup bago sukatin."""
    images = []
    for x, _ in ds:
        images.extend(x.numpy())
        if len(images) >= n_images + 5:
            break
    images = images[: n_images + 5]
    for img in images[:5]:
        model.predict(img[None], verbose=0)
    t0 = time.perf_counter()
    for img in images[5:]:
        model.predict(img[None], verbose=0)
    return (time.perf_counter() - t0) / max(len(images) - 5, 1) * 1000


def plot_confusion(cm, class_names, out_path, title):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Bilang"), (axes[1], cm_norm, ".2f", "Normalized (recall)")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Hinulaan")
        ax.set_ylabel("Totoong class")
        ax.set_title(f"{title}: {sub}")
        threshold = data.max() / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > threshold else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(model_path: Path, speed_images: int):
    name = model_path.stem
    model = common.load_trained_model(model_path)
    class_names = common.load_class_names()

    test_ds = common.load_split("test")
    assert test_ds.class_names == class_names, "Iba ang class order ng test set kaysa sa class_names.json"

    probs = model.predict(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    ms = measure_speed(model, test_ds, speed_images)
    size_mb = model_path.stat().st_size / (1024 * 1024)

    recalls = {c: report_dict[c]["recall"] for c in class_names}
    weak = [c for c, r in recalls.items() if r < RECALL_TARGET]

    # pinakamaraming pagkakamali
    confusions = [
        (class_names[i], class_names[j], int(cm[i, j]))
        for i in range(len(class_names)) for j in range(len(class_names)) if i != j and cm[i, j] > 0
    ]
    confusions.sort(key=lambda t: -t[2])

    lines = [
        f"Model: {model_path}",
        f"Test images: {len(y_true)}",
        f"Test accuracy: {acc * 100:.2f}%  (target >= {ACCURACY_TARGET * 100:.0f}%: {'OK' if acc >= ACCURACY_TARGET else 'HINDI PA'})",
        f"Macro F1: {macro_f1:.3f}",
        f"Laki ng model: {size_mb:.1f} MB",
        f"Bilis: {ms:.1f} ms/image (CPU, batch 1)",
        "",
        report_text,
        "Recall bawat class (target >= 80%):",
        *[f"  {c:<20}{r * 100:6.1f}%  {'OK' if r >= RECALL_TARGET else 'MABABA'}" for c, r in recalls.items()],
        "",
        "Mga nagkakapalitan (totoo -> hinula, bilang):",
        *([f"  {a} -> {b}: {n}" for a, b, n in confusions[:8]] or ["  wala"]),
    ]
    text = "\n".join(lines)
    print(text)

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (common.RESULTS_DIR / f"{name}_report.txt").write_text(text + "\n", encoding="utf-8")
    plot_confusion(cm, class_names, common.RESULTS_DIR / f"{name}_confusion_matrix.png", name)
    (common.RESULTS_DIR / f"{name}_metrics.json").write_text(
        json.dumps(
            {"model": name, "test_accuracy": acc, "macro_f1": macro_f1, "size_mb": size_mb,
             "ms_per_image": ms, "recall": recalls, "weak_classes": weak},
            indent=2,
        ),
        encoding="utf-8",
    )

    paths = [p for p in test_ds.file_paths]
    with open(common.RESULTS_DIR / f"{name}_misclassified.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "predicted", "confidence"])
        for p, t, pr, pb in zip(paths, y_true, y_pred, probs):
            if t != pr:
                w.writerow([p, class_names[t], class_names[pr], f"{pb[pr]:.3f}"])
    print(f"\nNaka-save ang mga resulta sa {common.RESULTS_DIR}")


def compare(names):
    rows = []
    for n in names:
        p = common.RESULTS_DIR / f"{n}_metrics.json"
        if not p.exists():
            raise SystemExit(f"Walang {p}. I-run muna ang evaluate.py --model-path ml/models/{n}.keras")
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    lines = [
        "| Model | Test accuracy | Macro F1 | Laki ng model (MB) | Bilis (ms/image) | Pinakamababang recall |",
        "|---|---|---|---|---|---|",
    ]
    for m in rows:
        worst_class = min(m["recall"], key=m["recall"].get)
        lines.append(
            f"| {m['model']} | {m['test_accuracy'] * 100:.2f}% | {m['macro_f1']:.3f} | {m['size_mb']:.1f} | "
            f"{m['ms_per_image']:.1f} | {m['recall'][worst_class] * 100:.1f}% ({worst_class}) |"
        )
    text = "\n".join(lines)
    print(text)
    (common.RESULTS_DIR / "comparison.md").write_text(text + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=Path)
    ap.add_argument("--compare", nargs="+", metavar="NAME", help="mga pangalan ng model file (walang .keras) na ikukumpara")
    ap.add_argument("--speed-images", type=int, default=50)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    elif args.model_path:
        evaluate(args.model_path, args.speed_images)
    else:
        ap.error("Kailangan ng --model-path o --compare")


if __name__ == "__main__":
    main()
