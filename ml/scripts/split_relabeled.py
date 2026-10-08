"""Hatiin muli ang aphids at flea_beetle pagkatapos ma-relabel ng user (panuntunan ng agriculturist:
malalaking butas = flea_beetle, maliliit = aphids).

Ang FOLDER ng bawat larawan sa --source ang label (hindi ang pangalan ng file, hindi tulad ng rebuild_dataset.py).

Ano ang ginagawa:
  1. Pool ng bawat class sa --classes = lahat ng nasa <source>/train/<class>.
     Ang lumang <source>/valid at <source>/test ng mga class na iyon ay HINDI ginagamit (hindi pa narepaso
     ayon sa bagong panuntunan), maliban kung --include-old-valid-test.
  2. Exact at near-duplicate check (kasama ang flip/rotate); kung magkaiba ang label ng magkaduplicate, pareho
     silang tinatanggal; at tinatanggal ang pool image na near-duplicate ng larawan sa ibang class.
  3. BLOCK split (magkakasunod na kuha ay nasa iisang split): --valid at --test bawat class, ang iba ay train.
  4. Ang ibang class (hindi sa --classes) ay kinokopya nang walang pagbabago sa split.
  5. Sumusulat ng resized copy (max 1024px) sa --out at manifest. HINDI binabago ang --source.

Default ay DRY RUN. Usage:
  python ml/scripts/split_relabeled.py
  python ml/scripts/split_relabeled.py --apply
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import dupes
import rebuild_dataset as rb

ML_DIR = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def collect(source: Path, splits, classes):
    items = []
    for split in splits:
        for c in classes:
            d = source / split / c
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    it = rb.Item(p, "user", 0, c, orig_rel=p.relative_to(source))
                    it.label = c
                    it.orig_split = split
                    if p.suffix.lower() not in rb.IMAGE_EXTS:
                        it.reject = "not_an_image"
                    items.append(it)
    return items


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_path", "class", "old_split", "new_split", "status", "reason", "duplicate_of", "sha256"])
        w.writerows(rows)
    print(f"Manifest: {path}")


def save_resized(src_path: Path, dest_dir: Path, stem: str, max_side: int):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem}.jpg"
    k = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_dup{k}.jpg"
        k += 1
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        im.save(dest, "JPEG", quality=95)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path(r"C:\Users\laman\OneDrive\Pictures\eggplant_pest_dataset"))
    ap.add_argument("--out", type=Path, default=ML_DIR / "data" / "dataset_v2")
    ap.add_argument("--manifest", type=Path, default=ML_DIR / "results" / "dataset_v2_manifest.csv")
    ap.add_argument("--classes", nargs="+", default=["aphids", "flea_beetle"], help="mga class na hahatiin muli")
    ap.add_argument("--valid", type=int, default=50, help="valid images bawat class na hahatiin muli")
    ap.add_argument("--test", type=int, default=50, help="test images bawat class na hahatiin muli")
    ap.add_argument("--include-old-valid-test", action="store_true",
                    help="isama sa pool ang lumang valid/test ng mga class na iyon (kung narepaso na)")
    ap.add_argument("--extra", action="append", default=[], metavar="CLASS=FOLDER",
                    help="dagdag na images para sa isang class (hal. healthy=C:\\...\\Healthy Leaf); hahatiin din (block split)")
    ap.add_argument("--extra-valid", type=int, default=30, help="valid images bawat --extra class")
    ap.add_argument("--extra-test", type=int, default=30, help="test images bawat --extra class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-size", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    resplit = args.classes
    others = [c for c in rb.CLASSES if c not in resplit]
    pool = collect(args.source, ("train", "valid", "test") if args.include_old_valid_test else ("train",), resplit)
    old = [] if args.include_old_valid_test else collect(args.source, ("valid", "test"), resplit)
    other = collect(args.source, SPLITS, others)
    print(f"Source: {args.source}")
    print(f"  pool: {dict(Counter(it.label for it in pool))} | hindi gagamitin (lumang valid/test): "
          f"{dict(Counter((it.orig_split, it.label) for it in old))} | ibang class: {len(other)} files")

    extras = []
    for spec in args.extra:
        cls, _, folder = spec.partition("=")
        if cls not in rb.CLASSES or not Path(folder).is_dir():
            sys.exit(f"--extra {spec!r}: kailangan ang CLASS=FOLDER na umiiral; class ay isa sa {rb.CLASSES}")
        for p in sorted(Path(folder).iterdir()):
            if p.is_file() and p.suffix.lower() in rb.IMAGE_EXTS:
                it = rb.Item(p, "extra", 0, cls, orig_rel=Path(p.name))
                it.label = cls
                it.orig_split = "extra"
                extras.append(it)
    if extras:
        print(f"  extra: {dict(Counter(it.label for it in extras))} mula sa {args.extra}")

    print("Hashing at near-duplicate check...")
    rb.hash_items([it for it in pool + other + extras if not it.reject])
    comp = rb.dedupe(pool)
    print(f"  pinakamalaking duplicate group sa pool: {comp[:5]}")
    for cls in sorted({it.label for it in extras}):
        mine = [it for it in extras if it.label == cls]
        rb.dedupe(mine)  # duplicate sa loob ng extra
        # near-duplicate ng larawan sa ibang class (hindi sa sariling class) = conflict; tanggalin ang extra
        ref = [it for it in pool + other if not it.reject and it.label != cls]
        ref_h = np.array([it.hashes[0] for it in ref], dtype=np.uint64)
        for it in mine:
            if not it.reject and len(ref):
                d = dupes.nearest_distance(it.hashes, ref_h)
                if (d <= dupes.DEFAULT_MAX_DIST).any():
                    it.reject = "near_duplicate_of_other_class"
                    it.dup_of = ref[int(d.argmin())].out_name
        # near-duplicate ng umiiral na larawan ng parehong class: leakage kung magkaibang split; tanggalin ang extra
        same = [it for it in other if not it.reject and it.label == cls]
        if same:
            same_h = np.array([it.hashes[0] for it in same], dtype=np.uint64)
            for it in mine:
                if not it.reject:
                    d = dupes.nearest_distance(it.hashes, same_h)
                    if (d <= dupes.DEFAULT_MAX_DIST).any():
                        it.reject = "near_duplicate_of_existing_same_class"
                        it.dup_of = same[int(d.argmin())].out_name
    base_other = np.array([it.hashes[0] for it in other if not it.reject], dtype=np.uint64)
    names_other = [it for it in other if not it.reject]
    for it in pool:
        if it.reject:
            continue
        d = dupes.nearest_distance(it.hashes, base_other)
        if (d <= dupes.DEFAULT_MAX_DIST).any():
            it.reject = "near_duplicate_of_other_class"
            it.dup_of = names_other[int(d.argmin())].out_name

    print("\n=== Tinanggal sa pool at extra ===")
    for (c, why), n in sorted(Counter((it.label, it.reject) for it in pool + extras if it.reject).items(), key=str):
        print(f"  {c:<12} {why:<40} {n}")
    for cls in sorted({it.label for it in extras}):
        cand = [it for it in extras if it.label == cls and not it.reject]
        if len(cand) <= args.extra_valid + args.extra_test:
            sys.exit(f"extra {cls}: {len(cand)} lang ang magagamit; kailangan ng higit sa {args.extra_valid + args.extra_test}.")
        rb.SPLIT_SIZES = {"test": args.extra_test, "valid": args.extra_valid, "train": 10**9}
        rb.block_split(cand, cls, args.seed, args.block_size)
        print(f"\n  extra {cls}:", {s: sum(it.split == s for it in cand) for s in SPLITS})
    for c in resplit:
        cand = [it for it in pool if it.label == c and not it.reject]
        need = args.valid + args.test
        if len(cand) <= need:
            sys.exit(f"{c}: {len(cand)} lang ang magagamit; kailangan ng higit sa {need}.")
        rb.SPLIT_SIZES = {"test": args.test, "valid": args.valid, "train": 10**9}  # lahat ng natira ay train
        rb.block_split(cand, c, args.seed, args.block_size)

    print("\n=== Bagong split at source mix ===")
    for c in resplit:
        print(f"  {c}")
        for split in SPLITS:
            sel = [it for it in pool if it.label == c and it.split == split]
            mix = Counter(rb.source_of(it) for it in sel)
            print(f"    {split:<6}{len(sel):>4}  {dict(mix.most_common(7))}")

    rows = []
    for it in pool:
        rows.append([str(it.orig_rel), it.label, it.orig_split, it.split or "", "kept" if it.split else "removed",
                     it.reject or "", it.dup_of or "", it.sha or ""])
    for it in extras:
        rows.append([f"EXTRA/{it.orig_rel}", it.label, "extra", it.split or "", "kept" if it.split else "removed",
                     it.reject or "", it.dup_of or "", it.sha or ""])
    for it in old:
        rows.append([str(it.orig_rel), it.label, it.orig_split, "", "not_used", "old_valid_test_not_reviewed", "", ""])
    for it in other:
        rows.append([str(it.orig_rel), it.label, it.orig_split, it.orig_split, "kept" if not it.reject else "removed",
                     it.reject or "", "", it.sha or ""])
    write_manifest(args.manifest, rows)

    if not args.apply:
        print("\nDRY RUN lang ito. Walang sinulat o binago. I-run ulit na may --apply.")
        return 0
    if args.out.exists() and any(args.out.rglob("*")):
        sys.exit(f"Hindi walang laman ang {args.out}. Alisin o palitan ang --out.")
    print(f"\nGumagawa ng resized copy (max {args.max_side}px) sa {args.out} ...")
    n = 0
    for it in pool:
        if it.split:
            save_resized(it.path, args.out / it.split / it.label, Path(it.out_name).stem, args.max_side)
            n += 1
    for it in extras:
        if it.split:
            save_resized(it.path, args.out / it.split / it.label, Path(it.out_name).stem, args.max_side)
            n += 1
    for it in other:
        if not it.reject:
            save_resized(it.path, args.out / it.orig_split / it.label, Path(it.out_name).stem, args.max_side)
            n += 1
    print(f"Tapos: {n} files. Walang binago sa {args.source}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
