"""Gawing pantay ang dataset: parehong bilang bawat class sa bawat split, at parehong laki ng lahat ng larawan.
Inihihiwalay din ang mga dahon na nasa healthy papunta sa bagong class na healthy_leaf.

Ano ang ginagawa (HINDI binabago ang --source):
  1. healthy -> healthy_leaf: ang "Healthy Leaf (n)" (Mafi et al. 2023) at "Fresh Eggplant Leaf_original*" (Nirob).
     Prutas na lang ang natitira sa healthy (sinilip ang lahat noong 2026-10-08).
  2. Kulang ang valid/test ng isang class: kumukuha ng BUONG block (magkakasunod na kuha o numero) mula sa train,
     para hindi magkahiwalay ng split ang magkakasunod na kuha.
  3. Sobra: tinatanggal muna ang mula sa source na pinakamarami sa class (hal. Aphids_original ng Nirob), para
     mabawasan ang source confound. Nasa --source pa rin ang mga tinanggal.
  4. Kulang ang train: augmented copy (flip o 90-degree rotation, zoom, brightness, contrast) ng train images ng
     parehong class, inuuna ang source na pinakakaunti. Sa train LANG; walang augmented sa valid at test.
  5. Nire-resize ang lahat sa --size x --size (default 224 = input ng model; stretch, hindi crop, tulad ng
     image_dataset_from_directory), LANCZOS, JPEG quality 95. Ganito rin ang resize sa predict.py.

Default ay DRY RUN. Usage:
  python ml/scripts/balance_dataset.py
  python ml/scripts/balance_dataset.py --apply
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance

import rebuild_dataset as rb

ML_DIR = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LEAF_RX = re.compile(r"^(Healthy Leaf|Fresh Eggplant Leaf_original)", re.I)  # mga dahon na nasa healthy
T = Image.Transpose
GEOMETRIC = {"flip_lr": T.FLIP_LEFT_RIGHT, "flip_tb": T.FLIP_TOP_BOTTOM, "rot90": T.ROTATE_90, "rot180": T.ROTATE_180,
             "rot270": T.ROTATE_270, "transpose": T.TRANSPOSE, "transverse": T.TRANSVERSE}


def source_name(path) -> str:
    """Kapareho ng evaluate.source_name (hindi in-import para hindi na i-load ang TensorFlow)."""
    stem = Path(path).stem
    if re.search(r"20\d{6}_?\d{6}", stem):
        return "camera"
    m = re.match(r"^(.*?)[\s_]*\(?\d+\)?(?:\s*\(\d+\))?$", stem)
    return (m.group(1).strip() if m and m.group(1).strip() else stem)[:24]


class Entry:
    def __init__(self, path: Path, cls: str, split: str):
        self.path = path              # file sa --source
        self.cls = cls
        self.split = split
        self.name = f"{path.stem}.jpg"
        self.notes = []
        self.aug = None               # augmentation params kapag augmented copy

    @property
    def source(self):
        return source_name(self.path)


def select(entries, cls, split):
    return [e for e in entries if e.cls == cls and e.split == split]


def blocks_of(items, block_size):
    """Magkakasunod na kuha (timestamp o numero ng file, ayon sa rb.seq_key), hinati sa blocks."""
    fams = defaultdict(list)
    for e in items:
        fam, key = rb.seq_key(e.path.name)
        fams[fam].append((key, e))
    blocks = []
    for fam in sorted(fams):
        lst = [e for _, e in sorted(fams[fam], key=lambda t: t[0])]
        blocks += [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    return blocks


def fill_from_train(entries, cls, split, need, block_size, rng):
    """Ilipat sa split ang `need` na train images, buong block. Ang sobra sa huling block ay tinatanggal (hindi
    iniiwan sa train, kasi magkakasunod na kuha sila ng nailipat)."""
    full = [b for b in blocks_of(select(entries, cls, "train"), block_size) if len(b) == block_size]
    rng.shuffle(full)
    removed = []
    for block in full:
        if need <= 0:
            break
        for e in block[:need]:
            e.split = split
            e.notes.append(f"train -> {split} (buong block)")
        for e in block[need:]:
            entries.remove(e)
            removed.append((e, f"kasama sa block na inilipat sa {split}"))
        need -= min(need, len(block))
    if need > 0:
        sys.exit(f"{cls}: kulang pa ng {need} sa {split} at wala nang buong block sa train.")
    return removed


def drop_excess(entries, cls, split, extra, rng):
    """Tanggalin ang `extra` na images, laging mula sa source na pinakamarami sa (class, split)."""
    by_src = defaultdict(list)
    for e in select(entries, cls, split):
        by_src[e.source].append(e)
    for lst in by_src.values():
        rng.shuffle(lst)
    removed = []
    for _ in range(extra):
        src = max(sorted(by_src), key=lambda s: len(by_src[s]))
        e = by_src[src].pop()
        entries.remove(e)
        removed.append((e, f"sobra sa {split}; mula sa pinakamaraming source ({src})"))
    return removed


def add_augmented(entries, cls, need, rng):
    """Magdagdag ng `need` na augmented copy ng train images. Inuuna ang source na pinakakaunti, at bawat
    orihinal ay nagagamit nang isang beses bago ulitin."""
    by_src = defaultdict(list)
    for e in select(entries, cls, "train"):
        by_src[e.source].append(e)
    for lst in by_src.values():
        rng.shuffle(lst)
    queue = [e for s in sorted(by_src, key=lambda s: (len(by_src[s]), s)) for e in by_src[s]]
    for k in range(need):
        orig = queue[k % len(queue)]
        a = Entry(orig.path, cls, "train")
        a.name = f"{orig.path.stem}_aug{k // len(queue) + 1}.jpg"
        a.aug = {
            "geo": rng.choice(sorted(GEOMETRIC)),
            "zoom": round(rng.uniform(0.85, 1.0), 3),
            "cx": round(rng.random(), 3),
            "cy": round(rng.random(), 3),
            "brightness": round(rng.uniform(0.8, 1.2), 3),
            "contrast": round(rng.uniform(0.8, 1.2), 3),
        }
        a.notes = orig.notes + [f"augmented mula sa {orig.path.name}"]
        entries.append(a)


def render(e: Entry, size: int) -> Image.Image:
    with Image.open(e.path) as im:
        im = im.convert("RGB")
        a = e.aug
        if a:  # zoom: crop na pareho ang aspect ratio, bago i-resize
            w, h = im.size
            cw, ch = round(w * a["zoom"]), round(h * a["zoom"])
            x, y = round((w - cw) * a["cx"]), round((h - ch) * a["cy"])
            im = im.crop((x, y, x + cw, y + ch))
        im = im.resize((size, size), Image.Resampling.LANCZOS)
        if a:
            im = im.transpose(GEOMETRIC[a["geo"]])
            im = ImageEnhance.Brightness(im).enhance(a["brightness"])
            im = ImageEnhance.Contrast(im).enhance(a["contrast"])
        return im


def print_table(title, entries, classes):
    c = Counter((e.cls, e.split) for e in entries)
    print(f"\n=== {title} ===")
    print(f"  {'class':<20}" + "".join(f"{s:>8}" for s in SPLITS))
    for cls in classes:
        print(f"  {cls:<20}" + "".join(f"{c[(cls, s)]:>8}" for s in SPLITS))
    print(f"  {'TOTAL':<20}" + "".join(f"{sum(c[(k, s)] for k in classes):>8}" for s in SPLITS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=ML_DIR / "data" / "dataset_v3")
    ap.add_argument("--out", type=Path, default=ML_DIR / "data" / "dataset_v4")
    ap.add_argument("--manifest", type=Path, default=ML_DIR / "results" / "dataset_v4_manifest.csv")
    ap.add_argument("--counts", type=int, nargs=3, metavar=("TRAIN", "VALID", "TEST"), default=[200, 50, 50])
    ap.add_argument("--size", type=int, default=224, help="laki (pixels, square) ng lahat ng larawan")
    ap.add_argument("--block-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    target = dict(zip(SPLITS, args.counts))
    rng = random.Random(args.seed)

    entries = []
    for split in SPLITS:
        for cls_dir in sorted(p for p in (args.source / split).iterdir() if p.is_dir()):
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    entries.append(Entry(p, cls_dir.name, split))
    print(f"Source: {args.source}")
    print_table("Bago", entries, sorted({e.cls for e in entries}))

    for e in entries:
        if e.cls == "healthy" and LEAF_RX.match(e.path.name):
            e.cls = "healthy_leaf"
            e.notes.append("healthy -> healthy_leaf")
    classes = sorted({e.cls for e in entries})

    removed = []
    for cls in classes:
        for split in ("test", "valid"):
            need = target[split] - len(select(entries, cls, split))
            if need > 0:
                removed += fill_from_train(entries, cls, split, need, args.block_size, rng)
    for cls in classes:
        for split in SPLITS:
            extra = len(select(entries, cls, split)) - target[split]
            if extra > 0:
                removed += drop_excess(entries, cls, split, extra, rng)
    for cls in classes:
        need = target["train"] - len(select(entries, cls, "train"))
        if need > 0:
            add_augmented(entries, cls, need, rng)

    print_table(f"Pagkatapos (lahat ay {args.size}x{args.size})", entries, classes)
    print("\n=== Ginawa bawat class ===")
    for cls in classes:
        mine = [e for e in entries if e.cls == cls]
        relabeled = sum("healthy -> healthy_leaf" in e.notes for e in mine if not e.aug)
        relabeled += sum("healthy -> healthy_leaf" in e.notes for e, _ in removed if e.cls == cls)
        moved = Counter(e.split for e in mine if not e.aug and any(n.startswith("train ->") for n in e.notes))
        dropped = Counter(e.source for e, _ in removed if e.cls == cls)
        augmented = Counter(e.source for e in mine if e.aug)
        parts = []
        if relabeled:
            parts.append(f"{relabeled} galing sa healthy")
        if moved:
            parts.append("mula train: " + ", ".join(f"{n} -> {s}" for s, n in sorted(moved.items())))
        if dropped:
            parts.append(f"{sum(dropped.values())} tinanggal {dict(dropped.most_common())}")
        if augmented:
            parts.append(f"{sum(augmented.values())} augmented {dict(augmented.most_common())}")
        print(f"  {cls:<20}{'; '.join(parts) or 'walang binago'}")
        mix = Counter(e.source for e in select(entries, cls, "train") if not e.aug)
        print(f"  {'':<20}train na totoong larawan: {sum(mix.values())} {dict(mix.most_common(6))}")

    bad = [(c, s, len(select(entries, c, s))) for c in classes for s in SPLITS if len(select(entries, c, s)) != target[s]]
    if bad:
        sys.exit(f"Hindi pantay pa rin: {bad}")
    names = Counter((e.split, e.cls, e.name) for e in entries)
    clash = [k for k, n in names.items() if n > 1]
    if clash:
        sys.exit(f"Magkaparehong pangalan ng output: {clash[:5]}")

    rows = []
    for e in sorted(entries, key=lambda e: (e.split, e.cls, e.name)):
        rows.append([f"{e.split}/{e.cls}/{e.name}", e.cls, e.split, e.path.relative_to(args.source), e.source,
                     "augmented" if e.aug else "kept", "; ".join(e.notes), json.dumps(e.aug) if e.aug else ""])
    for e, why in removed:
        rows.append(["", e.cls, "", e.path.relative_to(args.source), e.source, "removed", "; ".join(e.notes + [why]), ""])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_path", "class", "split", "source_path", "source", "status", "notes", "augmentation"])
        w.writerows(rows)
    print(f"\nManifest: {args.manifest}")

    if not args.apply:
        print("\nDRY RUN lang ito. Walang sinulat na larawan. I-run ulit na may --apply.")
        return 0
    if args.out.exists() and any(args.out.rglob("*")):
        sys.exit(f"Hindi walang laman ang {args.out}. Alisin o palitan ang --out.")
    print(f"\nSumusulat ng {len(entries)} larawan ({args.size}x{args.size}) sa {args.out} ...")
    for e in entries:
        dest = args.out / e.split / e.cls / e.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        render(e, args.size).save(dest, "JPEG", quality=95)
    print(f"Tapos. Walang binago sa {args.source}.")
    print(f"Susunod: python ml/scripts/check_dataset.py --dataset {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
