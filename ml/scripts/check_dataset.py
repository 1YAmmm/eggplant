"""Sprint 1D: sanity checks for ml/data/dataset.

Checks:
  1. pareho ang class folders sa train / valid / test (at tugma sa 5 expected classes)
  2. tama ang bilang bawat class bawat split (default 200 / 50 / 50)
  3. mabubuksan ang lahat ng image file
  4. walang parehong file hash sa pagitan ng mga split (anti-leakage)

Exit code 0 kung pasado lahat, 1 kung may problema.
Naka-save ang output sa ml/results/check_dataset.txt.

Usage:
  python ml/scripts/check_dataset.py
  python ml/scripts/check_dataset.py --expected 200 50 50
"""
import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ML_DIR / "data" / "dataset"
DEFAULT_REPORT = ML_DIR / "results" / "check_dataset.txt"

CLASSES = ["aphids", "flea_beetle", "fruit_shoot_borer", "healthy", "thrips"]
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Report:
    """Print sa console at i-save din sa file."""

    def __init__(self):
        self.lines = []
        self.problems = 0

    def log(self, msg=""):
        print(msg)
        self.lines.append(msg)

    def fail(self, msg):
        self.problems += 1
        self.log(f"  [FAIL] {msg}")

    def ok(self, msg):
        self.log(f"  [ OK ] {msg}")


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def can_open(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() invalidates the handle, kaya buksan ulit
            im.load()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--expected", type=int, nargs=3, metavar=("TRAIN", "VALID", "TEST"), default=[200, 50, 50])
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()
    expected = dict(zip(SPLITS, args.expected))

    rep = Report()
    rep.log(f"Dataset: {args.dataset}")
    rep.log(f"Expected per class: {expected}")
    rep.log()

    # 1. class folders
    rep.log("[1] Class folders")
    folders = {}
    for split in SPLITS:
        split_dir = args.dataset / split
        if not split_dir.is_dir():
            rep.fail(f"wala ang folder: {split_dir}")
            folders[split] = set()
            continue
        folders[split] = {p.name for p in split_dir.iterdir() if p.is_dir()}
    for split in SPLITS:
        missing = set(CLASSES) - folders[split]
        extra = folders[split] - set(CLASSES)
        if missing:
            rep.fail(f"{split}: kulang na class folder: {sorted(missing)}")
        if extra:
            rep.fail(f"{split}: hindi inaasahang folder: {sorted(extra)}")
        if not missing and not extra:
            rep.ok(f"{split}: {len(CLASSES)} class folders, tugma sa expected")
    rep.log()

    # 2. counts + 3. openable + gather hashes
    rep.log("[2] Bilang ng images / [3] Mabubuksan ba")
    counts = defaultdict(dict)
    hashes = defaultdict(list)  # hash -> [(split, class, filename)]
    corrupt = []
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = args.dataset / split / cls
            if not cls_dir.is_dir():
                counts[split][cls] = 0
                continue
            images = list_images(cls_dir)
            counts[split][cls] = len(images)
            for img in images:
                if not can_open(img):
                    corrupt.append(img)
                    continue
                hashes[file_hash(img)].append((split, cls, img.name))

    header = f"  {'class':<20}" + "".join(f"{s:>8}" for s in SPLITS)
    rep.log(header)
    for cls in CLASSES:
        rep.log(f"  {cls:<20}" + "".join(f"{counts[s].get(cls, 0):>8}" for s in SPLITS))
    rep.log(f"  {'TOTAL':<20}" + "".join(f"{sum(counts[s].values()):>8}" for s in SPLITS))

    problems_before = rep.problems
    for split in SPLITS:
        for cls in CLASSES:
            got = counts[split].get(cls, 0)
            if got != expected[split]:
                rep.fail(f"{split}/{cls}: {got} images (expected {expected[split]})")
    if rep.problems == problems_before:
        rep.ok("pantay at tama ang bilang bawat class sa bawat split")
    if corrupt:
        for c in corrupt:
            rep.fail(f"hindi mabuksan: {c}")
    else:
        rep.ok("mabubuksan ang lahat ng image")
    rep.log()

    # 4. leakage / duplicates
    rep.log("[4] Duplicates at leakage (file hash)")
    leaks, dupes = [], []
    for h, entries in hashes.items():
        if len(entries) < 2:
            continue
        if len({e[0] for e in entries}) > 1:
            leaks.append(entries)
        else:
            dupes.append(entries)
    if leaks:
        for entries in leaks:
            rep.fail("parehong file sa magkaibang split: " + " | ".join(f"{s}/{c}/{n}" for s, c, n in entries))
    else:
        rep.ok("walang parehong file hash sa pagitan ng train / valid / test")
    if dupes:
        for entries in dupes:
            rep.fail("duplicate sa loob ng iisang split: " + " | ".join(f"{s}/{c}/{n}" for s, c, n in entries))
    else:
        rep.ok("walang duplicate sa loob ng bawat split")
    rep.log()

    rep.log("RESULT: PASS" if rep.problems == 0 else f"RESULT: FAIL ({rep.problems} problema)")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    return 0 if rep.problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
