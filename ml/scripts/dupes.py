"""Near-duplicate detection na hindi naaapektuhan ng flip at 90-degree rotation.

Bakit kailangan: may mga dataset (hal. Tomato Insect Pests) na may magkaparehong larawan na naka-flip o
naka-rotate. Hindi sila mahuhuli ng exact file hash, pero kapag napunta ang isa sa train at ang isa sa
test, leakage pa rin iyon at peke ang accuracy.

Paraan: dHash (64 bits) ng larawan, at ng 8 dihedral variants nito (4 rotations x flip).
Magka-near-duplicate ang A at B kapag may variant ng A na malapit (Hamming distance) sa hash ng B.
"""
import io

import numpy as np
from PIL import Image

T = Image.Transpose
DIHEDRAL = [None, T.FLIP_LEFT_RIGHT, T.FLIP_TOP_BOTTOM, T.ROTATE_90, T.ROTATE_180, T.ROTATE_270, T.TRANSPOSE, T.TRANSVERSE]
DEFAULT_MAX_DIST = 5  # sa 64 bits


def _dhash(gray_9x8: Image.Image) -> np.uint64:
    a = np.asarray(gray_9x8, dtype=np.int16)  # shape (8, 9)
    bits = (a[:, 1:] > a[:, :-1]).flatten()  # 64 bits
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def hashes_from_image(im: Image.Image) -> np.ndarray:
    """8 hashes (uint64): isa para sa bawat dihedral variant. Ang index 0 ay ang orihinal."""
    im.draft("L", (160, 160))  # mas mabilis mag-decode ng malalaking JPEG
    square = im.convert("L").resize((72, 72), Image.Resampling.BILINEAR)
    out = np.empty(8, dtype=np.uint64)
    for k, t in enumerate(DIHEDRAL):
        v = square if t is None else square.transpose(t)
        out[k] = _dhash(v.resize((9, 8), Image.Resampling.BOX))
    return out


def hashes_from_path(path) -> np.ndarray:
    with Image.open(path) as im:
        return hashes_from_image(im)


def hashes_from_bytes(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return hashes_from_image(im)


def nearest_distance(variants: np.ndarray, base_hashes: np.ndarray) -> np.ndarray:
    """Para sa bawat base hash, ang pinakamaliit na Hamming distance mula sa alinmang variant ng larawan."""
    if len(base_hashes) == 0:
        return np.empty(0, dtype=np.int64)
    x = variants[:, None] ^ base_hashes[None, :]
    return np.bitwise_count(x).min(axis=0).astype(np.int64)


def near_duplicate_pairs(all_hashes: np.ndarray, max_dist: int = DEFAULT_MAX_DIST):
    """all_hashes: array (N, 8). Ibalik ang [(i, j, distance)] na i < j at magka-near-duplicate."""
    n = len(all_hashes)
    base = all_hashes[:, 0]
    pairs = []
    for i in range(n - 1):
        d = nearest_distance(all_hashes[i], base[i + 1 :])
        for off in np.nonzero(d <= max_dist)[0]:
            pairs.append((i, i + 1 + int(off), int(d[off])))
    return pairs
