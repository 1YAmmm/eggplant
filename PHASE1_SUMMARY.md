# Phase 1 Summary: ano ang nagawa

Petsa: 2026-10-07. Saklaw: Sprint 0 (kumpleto) at Sprint 1 (halos kumpleto) ng `plan.md`. Hindi pa nasisimulan ang pagte-train (Sprint 2 pataas).

## Sa isang tingin

| Sprint | Status |
|---|---|
| Sprint 0: Setup | Tapos |
| Sprint 1: Dataset | Halos tapos. Pasado ang `check_dataset.py` (200/50/50, walang leakage). Kulang pa: mga source/license na ikaw lang ang makakapunan, at ang pag-confirm ng agriculturist sa mga label |
| Sprint 2: Baseline training | Naisulat na ang mga script. **Hindi pa na-run o na-test.** Syntax check lang ang ginawa ko |
| Sprint 3 at 4 | Hindi pa nasisimulan |

## Sprint 0: Setup

- Python 3.11.9 (via winget), venv sa `venv/`, TensorFlow 2.21.0, at naka-pin ang versions sa `ml/requirements.txt`.
- **CPU lang** ang training. May RTX 2060 ka, pero hindi na sinusuportahan ng TensorFlow ang GPU sa native Windows, at walang WSL2 na naka-install. Para sa 1,500 images, okay lang ito.
- Gumawa ng `ml/` folder structure at `git init` na may `.gitignore` (`venv/`, `ml/data/`, `ml/models/`). Wala pang commit.
- Ang `eggplant/` folder mismo ang project root, hindi `eggplant-pest-app/`.
- Para lumabas ang `python` sa PATH, mag-open ng **bagong terminal**. Gamitin ang `venv\Scripts\python.exe` sa ngayon.

## Sprint 1: ang dataset

### Ang nahanap sa orihinal na dataset mo

Sinuri ko ang `C:\Users\laman\OneDrive\Pictures\eggplant_pest_dataset` (1,610 larawan) bago ko ito ginalaw:

| Problema | Bilang |
|---|---|
| Eksaktong kopya ng train sa valid o test | aphids: 44 sa 50 (valid) at 44 sa 50 (test); flea_beetle test: 7; thrips test: 8; 12 ang test na kapareho ng valid |
| Mga unique na larawan lang bawat class | aphids 214, flea_beetle 278, healthy 315, thrips 300, fruit_shoot_borer 373 (kailangan 300) |
| Larawan ng ibang sakit (Cercospora, Leaf Wilt, Powdery mildew, Mosaic) na nasa pest classes | 67 |
| Stock photos sa `healthy` | 37 |
| `Aphids_original*` na nasa flea_beetle (kasama ang ilang kapareho ng nasa aphids) | 34 |
| Magkakalapit na kuha (loob ng 30 segundo) na nasa magkaibang split | mga 70% ng valid/test camera photos ang may train photo na magkalapit ang oras |
| Hindi larawan | `.pdf` (inalis mo na bago ako natapos), `.ipynb` |

### Ang ginawa ko

Sinunod ko ang mga sagot mo: **I-rebuild ang split**, **alisin ang ibang sakit**, at **alisin ang stock photos**. Nasa `ml/scripts/rebuild_dataset.py` ang buong proseso.

1. Inalis ang exact at near-duplicate (kasama ang naka-flip o naka-rotate), 123 sa dataset mo. May `ml/scripts/dupes.py` para dito.
2. Inalis ang 67 sakit, 37 stock photo, at 1 hindi larawan.
3. Itinama ang class ng 38 larawan batay sa pangalan ng file: 34 `Aphids_original*` mula flea_beetle papuntang aphids, 2 papuntang flea_beetle, at 2 `Fresh Eggplant Leaf_original*` papuntang healthy.
4. Dinagdagan ang kulang mula sa Mendeley (CC BY 4.0, Original folders lang): aphids +86, flea_beetle +80, healthy +24.
5. Hinati sa **200/50/50** sa pamamagitan ng block split: ang 10 magkakasunod na kuha (ayon sa timestamp o numero ng file) ay nasa iisang split. Sinubukan ang 400 na pagkakasunod at pinili ang may pinakapantay na source mix. Seed 42.

### Resulta (`ml/results/check_dataset.txt`)

```
class                  train   valid    test
aphids                   200      50      50
flea_beetle              200      50      50
fruit_shoot_borer        200      50      50
healthy                  200      50      50
thrips                   200      50      50
TOTAL                   1000     250     250
RESULT: PASS
```

- Walang parehong file sa pagitan ng train, valid at test, at walang near-duplicate.
- **Babala (hindi nawala):** 63 sa 299 na valid/test camera photos ay may train photo na kinuha sa loob ng 30 segundo. Bumaba ito mula sa halos 70%, pero hindi zero.

### Nasaan ang mga file

| Ano | Saan |
|---|---|
| Bagong dataset (full resolution) | `C:\Users\laman\OneDrive\Pictures\eggplant_pest_dataset` (1.1 GB) |
| Dataset para sa training (resized, max 1024px) | `ml/data/dataset` (215 MB) |
| Lahat ng inalis | `C:\Users\laman\OneDrive\Pictures\eggplant_pest_dataset_rejected\_by_reason\<dahilan>\` (507 MB). Walang binura |
| Manifest (bawat file: saan galing, saan napunta, bakit inalis) | `ml/results/dataset_rebuild_manifest.csv` |
| Mga source at citation | `ml/DATA_SOURCES.md` |

**Paalala:** ang `.ipynb` mo (`eggplant_pest_training_local.ipynb`) ay nasa `eggplant_pest_dataset_rejected\_by_reason\not_an_image\valid\healthy\`. Ibalik mo kung kailangan.

Para ibalik ang orihinal na ayos: ang `_rejected` folder ay ang lumang folder na pinalitan ng pangalan, at ang manifest ang nagsasabi kung saan napunta ang bawat file.

## Mga babala tungkol sa dataset

Ang mga ito ay hindi mapapatunayan ng `check_dataset.py`, at maaaring maging dahilan kung bakit mataas ang test accuracy pero mahina sa totoong photo:

1. **Aphids at flea_beetle ay halos magkamukha.** 218 sa 300 aphids at 243 sa 300 flea_beetle ay galing sa iisang Mendeley dataset, at ang mga larawan ay nasirang dahon (may butas at spots), hindi insekto. Magkamukha ang dalawang folder sa mata ko. Asahan na magkakamali ang model dito. Ipa-confirm sa agriculturist ang mga label.
2. **Iisang farm session lang ang thrips, fruit_shoot_borer at healthy.** Mula ang mga ito sa camera photos na kinuha sa apat na petsa lang (2022-11-09, 2023-02-17, 2023-02-20, 2023-03-10). Optimistic ang test accuracy. Kailangan ng bagong photos mula sa ibang farm at ibang araw para sa Sprint 4.
3. **Hindi ko alam ang source at license ng 121 na larawan** (`EGGPLANT_ANGAT_ORIG`, `NEW_URBAN_ORIG`, atbp.), ng 874 na camera photo (fruit_shoot_borer, healthy, thrips), at ng 2 file na mukhang galing sa web. Nakalista ang mga ito sa "KAILANGAN MONG PUNAN" ng `ml/DATA_SOURCES.md`. Hindi pa kumpleto ang Definition of Done ng Sprint 1 hangga't hindi ito napupunan.
4. **Mga label na itinama ko ayon sa pangalan ng file** (38 larawan) ay hula mula sa pangalan. Tingnan kung tama.
5. Hindi ko sinuri kung may malabong larawan.

## Mga script (`ml/scripts/`)

| Script | Para saan | Status |
|---|---|---|
| `rebuild_dataset.py` | Nagbuo ng dataset (default ay dry run, `--apply` para isagawa) | Na-run |
| `check_dataset.py` | Bilang, class folders, file na hindi mabuksan, exact at near-duplicate | Na-run, PASS |
| `dupes.py` | Near-duplicate detection (dHash, tolerant sa flip at rotation) | Na-run (ginagamit ng dalawang nasa itaas) |
| `common.py` | Shared na constants, data loading at model builder | Syntax check lang |
| `train.py` | Baseline (frozen) at fine-tuning, MobileNetV2 o EfficientNetB0 | Syntax check lang |
| `evaluate.py` | Test accuracy, classification report, confusion matrix, comparison table | Syntax check lang |
| `predict.py` | Isang larawan o folder, confidence threshold, real-world accuracy | Syntax check lang |

## Susunod na hakbang

1. Punan ang `ml/DATA_SOURCES.md` (mga source at license) at ipa-confirm ang mga label sa agriculturist.
2. Mag-train (sa bagong terminal, sa project folder):
   ```
   venv\Scripts\python.exe ml\scripts\train.py --model mobilenetv2
   venv\Scripts\python.exe ml\scripts\evaluate.py --model-path ml\models\mobilenetv2_baseline.keras
   ```
   Posibleng may mga bug pa ang mga script dahil hindi pa sila na-run. Ipaayos mo sa akin kung may lumabas na error. Sa CPU, inaasahan ang ilang minuto kada epoch.
3. Pagkatapos, `--model efficientnetb0`, at `evaluate.py --compare ...` para sa comparison table (Sprint 3).
4. Maghanda ng bagong photos mula sa ibang farm at ibang araw para sa Sprint 4 (real-world test).

## Paglilinis na puwede mong gawin

- `C:\eggplant-data\` (mga 2.5 GB): mga na-download na zip (kasama ang mga hindi natapos na download) at ang mga na-extract. Puwede nang burahin kapag tapos ka na sa dataset. Kung ire-run mo ang `rebuild_dataset.py` ulit, kailangan mo ito ulit.
- `eggplant_pest_dataset_rejected` (507 MB): burahin kapag sigurado ka na.
- Nasa loob ng OneDrive ang `venv/` (maraming file at ilang GB), kaya nagsi-sync ito. Isaalang-alang na i-exclude ito sa OneDrive.
