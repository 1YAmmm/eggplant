"""Sprint 1B-1D: i-rebuild ang dataset mula sa organized dataset ng user (+ supplement mula sa Mendeley kung kulang).

Ano ang ginagawa:
  1. Kinukuha ang lahat ng larawan sa --source (train/valid/test/<class>) bilang iisang pool.
  2. Tinatanggal (inilalagay sa _rejected, hindi binubura):
       - hindi larawan (pdf, ipynb, ...)
       - larawan ng ibang sakit (Cercospora, Leaf Wilt, Powdery mildew, ...), hindi sakop ng 5 classes
       - stock photos (istockphoto, ...)
       - duplicates at near-duplicates (kasama ang naka-flip/rotate); label conflict
  3. Itinatama ang class kapag ang filename ay galing sa ibang class (hal. Aphids_original* na nasa flea_beetle)
  4. Kapag kulang sa 300 ang isang class, dinadagdagan mula sa na-download na Mendeley datasets
  5. Hinahati sa 200/50/50 gamit ang BLOCK split: magkakasunod na kuha (timestamp o numero ng file)
     ay nasa iisang split, para hindi "makita" ng model sa train ang halos kaparehong larawan ng test.
  6. --apply: nire-rebuild ang --source in place (ang luma ay nasa <source>_rejected), at gumagawa ng
     resized copy (max 1024px) sa ml/data/dataset para sa training.

Default ay DRY RUN (walang binabago). Nagsusulat ito ng manifest sa ml/results/dataset_rebuild_manifest.csv.

Usage:
  python ml/scripts/rebuild_dataset.py
  python ml/scripts/rebuild_dataset.py --apply
"""
import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import dupes

ML_DIR = Path(__file__).resolve().parents[1]
CLASSES = ["aphids", "flea_beetle", "fruit_shoot_borer", "healthy", "thrips"]
SPLIT_SIZES = {"test": 50, "valid": 50, "train": 200}  # ito ang pagkakasunod ng pag-ubos sa stream
NEED = sum(SPLIT_SIZES.values())
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 50

DISEASE_RX = re.compile(r"cercospora|leaf wilt|powdery|wilt disease|^mosaic", re.I)
STOCK_RX = re.compile(r"(?i:^istockphoto|^brinjal-51-)|^(?=[0-9]*[A-Z])[A-Z0-9]{12}\.")  # 2nd: stock ID tulad ng 3HTJ19EY2ZMT.jpg
# ang pangalan ng file ay nagsasabi kung saang class ito galing sa orihinal na dataset
SOURCE_LABEL_RX = [
    (re.compile(r"^Aphids_original", re.I), "aphids"),
    (re.compile(r"^Flea Beetles_original|^Flea beetles and feeding", re.I), "flea_beetle"),
    (re.compile(r"^Fresh Eggplant( Leaf)?_original", re.I), "healthy"),
]

# mga supplement: (zip, folder sa loob ng zip, class, tag, tier)  tier 1 = una; tier 2 = reserba kung kulang pa
SUPPLEMENTS = [
    ("eggplant_dataset_original.zip", "Original Dataset/Aphids/", "aphids", "", 1),
    ("eggplant_dataset_original.zip", "Original Dataset/Flea Beetles/", "flea_beetle", "", 1),
    ("eggplant_dataset_original.zip", "Original Dataset/Fresh Eggplant Leaf/", "healthy", "", 1),
    ("eggplant_dataset_original.zip", "Original Dataset/Fresh Eggplant/", "healthy", "", 1),
    ("tomato_insect_pests.zip", "Tomato Dataset/Original_image_of_tomato_pest/Thrips/", "thrips", "tomato_", 1),
    ("tomato_insect_pests.zip", "Tomato Dataset/Original_image_of_tomato_pest/Peach_Aphid/", "aphids", "tomato_", 2),
]

TS_RX = re.compile(r"(20\d{6})_?(\d{6})")
NUM_RX = re.compile(r"^(?P<fam>.*?)[\s_]*\(?(?P<num>\d+)\)?(?:\s*\(\d+\))?$")


def seq_key(name: str):
    """(family, key): pinagsasama ang magkakasunod na kuha. Camera timestamp o numero ng file."""
    stem = Path(name).stem
    m = TS_RX.search(stem)
    if m:
        return "camera", m.group(1) + m.group(2)
    m = NUM_RX.match(stem)
    if m and m.group("fam").strip():
        return m.group("fam").strip(), f"{int(m.group('num')):08d}"
    return "misc", stem


class Item:
    def __init__(self, path: Path, origin: str, tier: int, folder_label=None, orig_rel=None, out_name=None):
        self.path = path
        self.origin = origin            # "user" o "supp"
        self.tier = tier                # 0 = sa user, 1/2 = supplement
        self.folder_label = folder_label
        self.orig_rel = orig_rel        # relative path sa source (para sa manifest/restore)
        self.out_name = out_name or path.name
        self.label = None
        self.reject = None              # reason
        self.dup_of = None
        self.split = None
        self.sha = None
        self.hashes = None
        self.size = None

    @property
    def family(self):
        return seq_key(self.out_name)


def resolve_label(item: Item):
    name = item.path.name
    if item.path.suffix.lower() not in IMAGE_EXTS:
        return None, "not_an_image"
    if DISEASE_RX.search(name):
        return None, "disease_not_in_scope"
    if STOCK_RX.search(name):
        return None, "stock_photo"
    for rx, label in SOURCE_LABEL_RX:
        if rx.search(name):
            return label, None
    return item.folder_label, None


def collect_user(source: Path):
    items = []
    for p in sorted(source.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(source)
        parts = rel.parts
        folder_label = parts[1] if len(parts) >= 3 and parts[1] in CLASSES else None
        items.append(Item(p, "user", 0, folder_label, orig_rel=rel))
    return items


def extract_supplements(raw_dir: Path, staging: Path):
    items = []
    for zname, prefix, label, tag, tier in SUPPLEMENTS:
        zpath = raw_dir / zname
        if not zpath.exists():
            print(f"  [skip] wala ang {zpath}")
            continue
        dest = staging / zname.removesuffix(".zip") / prefix.strip("/").replace("/", "__")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.startswith(prefix):
                    continue
                name = info.filename[len(prefix):]
                if "/" in name or Path(name).suffix.lower() not in IMAGE_EXTS:
                    continue
                target = dest / (tag + name)
                if not target.exists():
                    target.write_bytes(z.read(info))
                it = Item(target, "supp", tier, label, out_name=tag + name)
                it.label = label
                items.append(it)
    return items


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def hash_items(items):
    ok = []
    for n, it in enumerate(items, 1):
        if n % 500 == 0:
            print(f"  hashing {n}/{len(items)}")
        try:
            it.sha = hashlib.sha256(it.path.read_bytes()).hexdigest()
            with Image.open(it.path) as im:
                it.size = im.size
            it.hashes = dupes.hashes_from_path(it.path)
            if min(it.size) < MIN_SIDE:
                it.reject = "too_small"
        except Exception:
            it.reject = "unreadable"
        ok.append(it)
    return ok


def dedupe(items):
    """Exact + near duplicates (flip/rotate). Magkaibang label sa iisang component = label_conflict."""
    live = [it for it in items if not it.reject]
    dsu = DSU(len(live))
    by_sha = {}
    for i, it in enumerate(live):
        if it.sha in by_sha:
            dsu.union(i, by_sha[it.sha])
        else:
            by_sha[it.sha] = i
    pairs = dupes.near_duplicate_pairs(np.stack([it.hashes for it in live]))
    for i, j, _ in pairs:
        dsu.union(i, j)
    comps = defaultdict(list)
    for i in range(len(live)):
        comps[dsu.find(i)].append(live[i])
    sizes = sorted((len(c) for c in comps.values()), reverse=True)
    for comp in comps.values():
        if len(comp) == 1:
            continue
        labels = {it.label for it in comp}
        if len(labels) > 1:
            for it in comp:
                it.reject = "label_conflict"
            continue
        comp.sort(key=lambda it: (it.tier, -(it.size[0] * it.size[1]), it.out_name))
        for it in comp[1:]:
            it.reject = "duplicate"
            it.dup_of = comp[0].out_name
    return sizes


def source_of(it):
    """Ang 'source' para sa pagbalanse ng mix: file series, o petsa para sa camera photos."""
    fam, key = it.family
    return f"camera-{key[:8]}" if fam == "camera" else fam


def block_split(pool, label, seed, block_size, trials=400):
    """Hatiin ang pool sa magkakasunod na blocks, i-shuffle ang blocks, at ubusin sa pagkakasunod: test, valid, train.

    Sinusubukan ang `trials` na random shuffles at pinipili ang may pinakapantay na source mix sa tatlong
    split (para hindi, hal., puro isang source ang valid pero halo ang test).
    """
    groups = defaultdict(list)
    for it in pool:
        groups[it.family[0]].append((it.family[1], it))
    blocks = []
    for fam in sorted(groups):
        lst = [it for _, it in sorted(groups[fam], key=lambda t: t[0])]
        blocks += [lst[i : i + block_size] for i in range(0, len(lst), block_size)]

    sources = sorted({source_of(it) for it in pool})
    overall = {s: sum(source_of(it) == s for it in pool) / len(pool) for s in sources}
    best = None
    for t in range(trials):
        rng = random.Random(f"{seed}-{label}-{t}")
        order = blocks[:]
        rng.shuffle(order)
        stream = [it for b in order for it in b]
        pos, parts = 0, {}
        for split, n in SPLIT_SIZES.items():
            parts[split] = stream[pos : pos + n]
            pos += n
        score = 0.0
        for split, part in parts.items():
            for s in sources:
                share = sum(source_of(it) == s for it in part) / len(part)
                score += (share - overall[s]) ** 2
        if best is None or score < best[0]:
            best = (score, parts, stream[pos:])
    _, parts, surplus = best
    for split, part in parts.items():
        for it in part:
            it.split = split
    for it in surplus:
        it.reject = "unused_surplus"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path(r"C:\Users\laman\OneDrive\Pictures\eggplant_pest_dataset"))
    ap.add_argument("--raw-dir", type=Path, default=Path(r"C:\eggplant-data\raw"))
    ap.add_argument("--staging", type=Path, default=Path(r"C:\eggplant-data\extracted"))
    ap.add_argument("--out", type=Path, default=ML_DIR / "data" / "dataset")
    ap.add_argument("--manifest", type=Path, default=ML_DIR / "results" / "dataset_rebuild_manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-size", type=int, default=10)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    print(f"Source: {args.source}")
    user = collect_user(args.source)
    print(f"  {len(user)} files sa source")
    print("Supplements:")
    supp = extract_supplements(args.raw_dir, args.staging)
    print(f"  {len(supp)} candidate images")

    for it in user:
        it.label, it.reject = resolve_label(it)
    relabeled = [it for it in user if it.label and it.folder_label and it.label != it.folder_label]

    items = user + supp
    print("Hashing at near-duplicate check...")
    hash_items([it for it in items if not it.reject])
    comp_sizes = dedupe(items)
    print(f"  pinakamalaking duplicate group: {comp_sizes[:5]}")

    # --- pool bawat class, dagdag ng supplements kung kulang
    pools = {}
    for c in CLASSES:
        mine = [it for it in items if it.label == c and not it.reject]
        base = [it for it in mine if it.tier == 0]
        pool = list(base)
        for tier in (1, 2):
            short = NEED - len(pool)
            cands = [it for it in mine if it.tier == tier]
            rng = random.Random(f"{args.seed}-supp-{c}-{tier}")
            rng.shuffle(cands)
            pool += cands[: max(short, 0)]
            for it in cands[max(short, 0):]:
                it.reject = "supplement_not_needed"
        pools[c] = (len(base), pool)

    # --- report
    reasons = Counter((it.origin, it.reject) for it in items if it.reject)
    print("\n=== Tinanggal / hindi ginamit ===")
    for (origin, why), n in sorted(reasons.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"  {origin:<5} {why:<24} {n}")
    print(f"\nNa-relabel (filename vs folder): {len(relabeled)}")
    for k, v in Counter((it.folder_label, it.label) for it in relabeled).items():
        print(f"  {k[0]} -> {k[1]}: {v}")

    print(f"\n=== Pool bawat class (kailangan {NEED}) ===")
    shortage = False
    for c in CLASSES:
        n_user, pool = pools[c]
        n_supp = len(pool) - n_user
        flag = "" if len(pool) >= NEED else f"   <-- KULANG ng {NEED - len(pool)}"
        shortage |= len(pool) < NEED
        print(f"  {c:<18} user {n_user:>4} + supplement {n_supp:>4} = {len(pool):>4}{flag}")

    if shortage:
        print("\nKulang ang images sa isa o higit pang class. Hindi ko itutuloy; kailangan ng dagdag na images.")
        write_manifest(args.manifest, items)
        return 2

    for c in CLASSES:
        block_split(pools[c][1], c, args.seed, args.block_size)

    print("\n=== Bagong split at source mix ===")
    for c in CLASSES:
        print(f"  {c}")
        for split in SPLIT_SIZES:
            sel = [it for it in pools[c][1] if it.split == split]
            mix = Counter(source_of(it) for it in sel)
            print(f"    {split:<6}{len(sel):>4}  {dict(mix.most_common(7))}")
    write_manifest(args.manifest, items)

    if not args.apply:
        print("\nDRY RUN lang ito. Walang binago. I-run ulit na may --apply para itutuloy.")
        return 0
    apply_changes(args, items)
    return 0


def write_manifest(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origin", "original_path", "label", "final_split", "status", "reason", "duplicate_of", "sha256"])
        for it in items:
            status = "kept" if it.split else "removed"
            orig = str(it.orig_rel) if it.orig_rel else str(it.path)
            w.writerow([it.origin, orig, it.label or "", it.split or "", status, it.reject or "", it.dup_of or "", it.sha or ""])
    print(f"Manifest: {path}")


def apply_changes(args, items):
    src = args.source
    rejected_root = src.with_name(src.name + "_rejected")
    if rejected_root.exists():
        sys.exit(f"Existing na ang {rejected_root}. Ilipat o palitan ang pangalan muna.")
    kept = [it for it in items if it.split]
    print(f"\nApply: {len(kept)} kept, {len([i for i in items if not i.split])} tinanggal")

    # 1. ilipat ang buong lumang source sa _rejected (isang rename lang)
    src.rename(rejected_root)
    for it in items:
        if it.origin == "user":
            it.path = rejected_root / it.orig_rel  # nasa bagong lokasyon na
    # 2. gawin ang bagong source: ilipat ang kept user files, kopyahin ang supplements
    used = set()
    final_paths = []
    for it in kept:
        dest_dir = src / it.split / it.label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / it.out_name
        k = 1
        while dest in used or dest.exists():
            dest = dest_dir / f"{Path(it.out_name).stem}_dup{k}{Path(it.out_name).suffix}"
            k += 1
        used.add(dest)
        if it.origin == "user":
            shutil.move(str(it.path), dest)
        else:
            shutil.copy2(it.path, dest)
        final_paths.append((it, dest))
    # 3. ayusin ang natirang files sa _rejected ayon sa dahilan
    for it in items:
        if it.origin == "user" and not it.split and it.path.exists():
            why = it.reject or "not_selected"
            target = rejected_root / "_by_reason" / why / it.orig_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(it.path), target)
    print(f"  Bagong dataset: {src}")
    print(f"  Mga tinanggal: {rejected_root}\\_by_reason\\<dahilan>\\...  (ang ibang laman ng lumang folder ay nandoon pa)")

    # 4. resized copy para sa training
    out = args.out
    if out.exists() and any(out.rglob("*.jpg")):
        sys.exit(f"Hindi walang laman ang {out}; hindi ko ito ire-resize doon. Alisin muna.")
    print(f"Gumagawa ng resized copy (max {args.max_side}px) sa {out} ...")
    for it, dest in final_paths:
        o = out / it.split / it.label / (Path(dest.name).stem + ".jpg")
        o.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(dest) as im:
            im = im.convert("RGB")
            im.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            im.save(o, "JPEG", quality=95)
    print("Tapos. I-run ang: python ml/scripts/check_dataset.py")


if __name__ == "__main__":
    sys.exit(main())
