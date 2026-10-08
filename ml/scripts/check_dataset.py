"""Sprint 1D: sanity checks for ml/data/dataset.

Checks:
  1. pareho ang class folders sa train / valid / test (at tugma sa 6 expected classes)
  2. tama ang bilang bawat class bawat split (default 200 / 50 / 50)
  3. mabubuksan ang lahat ng image file
  4. walang parehong file hash sa pagitan ng mga split (anti-leakage)
  5. walang near-duplicate (kahit naka-flip/rotate) sa pagitan ng mga split o class; babala para sa
     magkakalapit na kuha na magkaiba ang split

Exit code 0 kung pasado lahat, 1 kung may problema.
Naka-save ang output sa ml/results/check_dataset.txt.

Usage:
  python ml/scripts/check_dataset.py
  python ml/scripts/check_dataset.py --expected 200 50 50
"""
import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import dupes

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ML_DIR / "data" / "dataset"
DEFAULT_REPORT = ML_DIR / "results" / "check_dataset.txt"

CLASSES = ["aphids", "flea_beetle", "fruit_shoot_borer", "healthy", "healthy_leaf", "thrips"]
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
    ap.add_argument("--allow-uneven", action="store_true",
                    help="hindi kailangang pantay ang bilang; kailangan lang na may laman ang bawat class sa bawat split")
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
    near = []  # (split, class, path, dihedral hashes)
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
                near.append((split, cls, img, dupes.hashes_from_path(img)))

    header = f"  {'class':<20}" + "".join(f"{s:>8}" for s in SPLITS)
    rep.log(header)
    for cls in CLASSES:
        rep.log(f"  {cls:<20}" + "".join(f"{counts[s].get(cls, 0):>8}" for s in SPLITS))
    rep.log(f"  {'TOTAL':<20}" + "".join(f"{sum(counts[s].values()):>8}" for s in SPLITS))

    problems_before = rep.problems
    for split in SPLITS:
        for cls in CLASSES:
            got = counts[split].get(cls, 0)
            if args.allow_uneven:
                if got == 0:
                    rep.fail(f"{split}/{cls}: walang image")
            elif got != expected[split]:
                rep.fail(f"{split}/{cls}: {got} images (expected {expected[split]})")
    if rep.problems == problems_before:
        rep.ok("may laman ang bawat class sa bawat split (hindi kailangang pantay: --allow-uneven)" if args.allow_uneven
               else "pantay at tama ang bilang bawat class sa bawat split")
    if corrupt:
        for c in corrupt:
            rep.fail(f"hindi mabuksan: {c}")
    else:
        rep.ok("mabubuksan ang lahat ng image")
    rep.log()

    # 4. leakage / duplicates
    rep.log("[4] Duplicates at leakage (file hash)")
    leaks, in_split_dupes = [], []
    for h, entries in hashes.items():
        if len(entries) < 2:
            continue
        if len({e[0] for e in entries}) > 1:
            leaks.append(entries)
        else:
            in_split_dupes.append(entries)
    if leaks:
        for entries in leaks:
            rep.fail("parehong file sa magkaibang split: " + " | ".join(f"{s}/{c}/{n}" for s, c, n in entries))
    else:
        rep.ok("walang parehong file hash sa pagitan ng train / valid / test")
    if in_split_dupes:
        for entries in in_split_dupes:
            rep.fail("duplicate sa loob ng iisang split: " + " | ".join(f"{s}/{c}/{n}" for s, c, n in entries))
    else:
        rep.ok("walang duplicate sa loob ng bawat split")
    rep.log()

    # 5. near-duplicates (kasama ang naka-flip / naka-rotate), at magkakalapit na kuha
    rep.log("[5] Near-duplicates (dHash, tolerant sa flip at rotation) at magkakalapit na kuha")
    pairs = dupes.near_duplicate_pairs(np.stack([n[3] for n in near])) if near else []
    cross_split = [p for p in pairs if near[p[0]][0] != near[p[1]][0]]
    cross_class = [p for p in pairs if near[p[0]][1] != near[p[1]][1]]
    same_split = [p for p in pairs if near[p[0]][0] == near[p[1]][0] and near[p[0]][1] == near[p[1]][1]]
    for i, j, d in cross_split[:10]:
        rep.fail(f"near-duplicate sa magkaibang split (d={d}): {near[i][0]}/{near[i][1]}/{near[i][2].name} ~ "
                 f"{near[j][0]}/{near[j][1]}/{near[j][2].name}")
    if not cross_split:
        rep.ok("walang near-duplicate sa pagitan ng train / valid / test")
    for i, j, d in cross_class[:10]:
        if (i, j, d) not in cross_split:
            rep.fail(f"near-duplicate sa magkaibang class (d={d}): {near[i][1]}/{near[i][2].name} ~ {near[j][1]}/{near[j][2].name}")
    if not cross_class:
        rep.ok("walang near-duplicate sa pagitan ng magkaibang class")
    rep.log(f"  [INFO] near-duplicate sa loob ng iisang split at class: {len(same_split)} pares (hindi leakage)")

    # babala lang: camera photos (YYYYMMDD_HHMMSS) na magkalapit ang oras pero magkaibang split
    ts_rx = re.compile(r"(20\d{6})_?(\d{6})")
    stamped = []
    for split, cls, path, _ in near:
        m = ts_rx.search(path.stem)
        if m:
            stamped.append((split, cls, datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")))
    close = Counter()
    by_class = defaultdict(list)
    for s, c, t in stamped:
        by_class[c].append((s, t))
    for c, lst in by_class.items():
        for s, t in lst:
            if s != "train" and any(s2 == "train" and abs((t - t2).total_seconds()) <= 30 for s2, t2 in lst):
                close[(s, c)] += 1
    n_close = sum(close.values())
    n_eval = sum(1 for s, c, t in stamped if s != "train")
    rep.log(f"  [WARN] valid/test photos na may train photo (parehong class) sa loob ng 30 segundo: {n_close} / {n_eval}"
            " (babala lang; hindi ito exact o near duplicate)")
    for (s, c), n in sorted(close.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rep.log(f"         {c:<18}{s:<6}{n}")
    rep.log()

    rep.log("RESULT: PASS" if rep.problems == 0 else f"RESULT: FAIL ({rep.problems} problema)")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    return 0 if rep.problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
