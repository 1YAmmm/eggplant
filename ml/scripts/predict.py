"""Sprint 4: prediction at real-world testing.

Mga mode:
  1. Isang image:
       python ml/scripts/predict.py --model ml/models/mobilenetv2_finetuned.keras --image foto.jpg
  2. Folder ng images (walang labels), lahat ay hinuhulaan:
       python ml/scripts/predict.py --model ... --folder ml/data/other_images
  3. Real-world set na may labels (subfolder = class name, tulad ng dataset/test):
       python ml/scripts/predict.py --model ... --folder ml/data/realworld --labeled
     Naglalabas ng accuracy, confidence ng tama vs mali, at threshold sweep (Sprint 4B).
     Naka-save ang detalye sa ml/results/<folder>_predictions.csv

Kapag mas mababa ang top confidence kaysa --threshold (default 0.60),
sinasabing "Hindi sigurado, paki-retake ang photo".
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import common
from common import tf

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: Path) -> np.ndarray:
    """RGB, 224x224, raw 0-255 (nasa loob ng model ang preprocessing).

    Parehong decode at resize ng image_dataset_from_directory (ginamit sa training at evaluate.py),
    kasi nag-iiba ang hula kapag PIL ang nag-resize.
    """
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    return tf.image.resize(img, (common.IMG_SIZE, common.IMG_SIZE)).numpy().astype(np.float32)


def predict_paths(model, paths, class_names, batch_size=32):
    results = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        batch = np.stack([load_image(p) for p in chunk])
        probs = model.predict(batch, verbose=0)
        for p, pr in zip(chunk, probs):
            order = np.argsort(pr)[::-1]
            results.append({"path": p, "probs": pr, "top": class_names[order[0]], "conf": float(pr[order[0]]),
                            "second": class_names[order[1]], "second_conf": float(pr[order[1]])})
    return results


def gather_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def describe(r, threshold):
    flag = "" if r["conf"] >= threshold else "  -> Hindi sigurado, paki-retake ang photo"
    return f"{r['top']} ({r['conf'] * 100:.1f}%), susunod: {r['second']} ({r['second_conf'] * 100:.1f}%){flag}"


def report_labeled(results, class_names, folder: Path, threshold):
    """Accuracy + confidence analysis para sa real-world set na may labels."""
    for r in results:
        r["true"] = r["path"].parent.name
    unknown = {r["true"] for r in results} - set(class_names)
    if unknown:
        raise SystemExit(f"Hindi class name ang mga subfolder: {sorted(unknown)}. Dapat {class_names}")

    correct = [r for r in results if r["top"] == r["true"]]
    wrong = [r for r in results if r["top"] != r["true"]]
    print(f"\nTotal: {len(results)} | Tama: {len(correct)} | Mali: {len(wrong)} | Accuracy: {len(correct) / len(results) * 100:.1f}%")

    print("\nAccuracy bawat class:")
    for c in class_names:
        rs = [r for r in results if r["true"] == c]
        if rs:
            ok = sum(r["top"] == c for r in rs)
            print(f"  {c:<20}{ok:>4}/{len(rs):<4}{ok / len(rs) * 100:6.1f}%")

    if correct:
        print(f"\nAverage confidence ng TAMA: {np.mean([r['conf'] for r in correct]) * 100:.1f}%")
    if wrong:
        print(f"Average confidence ng MALI: {np.mean([r['conf'] for r in wrong]) * 100:.1f}%")

    print("\nThreshold sweep (ang hula na mas mababa sa threshold ay 'Hindi sigurado'):")
    print(f"  {'threshold':>9} {'nasagot':>8} {'accuracy ng nasagot':>20} {'maling nakalusot':>17} {'tamang naitapon':>16}")
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        answered = [r for r in results if r["conf"] >= t]
        ok = sum(r["top"] == r["true"] for r in answered)
        acc = ok / len(answered) * 100 if answered else float("nan")
        wrong_passed = len(answered) - ok
        right_dropped = sum(r["top"] == r["true"] and r["conf"] < t for r in results)
        mark = "  <- napili" if abs(t - threshold) < 1e-9 else ""
        print(f"  {t:>9.2f} {len(answered):>4}/{len(results):<3} {acc:>19.1f}% {wrong_passed:>17} {right_dropped:>16}{mark}")

    if wrong:
        print("\nMga maling hula (pinakamataas ang confidence muna):")
        for r in sorted(wrong, key=lambda r: -r["conf"])[:15]:
            print(f"  {r['path'].name}: totoo={r['true']} hula={r['top']} ({r['conf'] * 100:.0f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--labeled", action="store_true", help="subfolder = class name; kukwentahin ang accuracy")
    ap.add_argument("--threshold", type=float, default=0.60)
    args = ap.parse_args()
    if not args.image and not args.folder:
        ap.error("Kailangan ng --image o --folder")

    model = common.load_trained_model(args.model)
    class_names = common.load_class_names()

    paths = [args.image] if args.image else gather_images(args.folder)
    if not paths:
        raise SystemExit("Walang image na nahanap.")
    results = predict_paths(model, paths, class_names)

    if args.image:
        print(describe(results[0], args.threshold))
        return

    if args.labeled:
        report_labeled(results, class_names, args.folder, args.threshold)
    else:
        for r in results:
            print(f"{r['path'].name}: {describe(r, args.threshold)}")
        unsure = sum(r["conf"] < args.threshold for r in results)
        print(f"\n{len(results)} images | 'Hindi sigurado': {unsure}")

    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS_DIR / f"{args.folder.name}_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", *( ["true"] if args.labeled else [] ), "predicted", "confidence", *class_names])
        for r in results:
            w.writerow([r["path"], *([r["true"]] if args.labeled else []), r["top"], f"{r['conf']:.4f}",
                        *[f"{p:.4f}" for p in r["probs"]]])
    print(f"\nNaka-save: {out}")


if __name__ == "__main__":
    main()
