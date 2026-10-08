# Eggplant Pest Detection Web App: Project Plan

Isang website (Next.js) kung saan mag-a-upload o kukuha ng picture ang user, ire-recognize ng model ang eggplant pest, at magbibigay ng tamang pesticide (IPM muna) para sa pest na iyon.

**Phase 1 (detalyado): Model training, locally**
**Phase 2 (outline lang muna): Next.js website + pesticide recommendation**

---

## Mga Class (6)

| # | Class (folder name) | Pest / kondisyon |
|---|---|---|
| 1 | `aphids` | Aphids |
| 2 | `flea_beetle` | Flea beetle |
| 3 | `fruit_shoot_borer` | Eggplant fruit and shoot borer (EFSB) |
| 4 | `healthy` | Walang pest (prutas) |
| 5 | `healthy_leaf` | Walang pest (dahon). Hiniwalay sa `healthy` noong 2026-10-08 |
| 6 | `thrips` | Thrips |

**Target dataset: 200 train / 50 valid / 50 test bawat class, lahat ng larawan ay 224x224**
Total: 1,200 train + 300 valid + 300 test = **1,800 images**

---

## Tech Stack

| Bahagi | Gagamitin |
|---|---|
| Training | Python 3.11, TensorFlow/Keras |
| Model | MobileNetV2 at EfficientNetB0 (transfer learning), pagkukumparahin |
| Dev tools | VS Code o Jupyter, Git |
| Website | Next.js (React) |
| Model sa website | Pipiliin sa Sprint 4 (tingnan ang Decision Point sa ibaba) |
| Pest + pesticide data | JSON o database, validated ng agriculturist |

---

## Project Folder Structure

```
eggplant-pest-app/
├── plan.md
├── ml/                          <- Phase 1 (training)
│   ├── data/
│   │   ├── raw/                 <- mga na-download na datasets (huwag galawin)
│   │   └── dataset/             <- final dataset
│   │       ├── train/{aphids, flea_beetle, fruit_shoot_borer, healthy, healthy_leaf, thrips}
│   │       ├── valid/{...parehong 6}
│   │       └── test/{...parehong 6}
│   ├── scripts/
│   │   ├── balance_dataset.py
│   │   ├── check_dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── models/                  <- naka-save na trained models
│   ├── results/                 <- graphs, confusion matrix, reports
│   ├── DATA_SOURCES.md          <- listahan ng sources at citations
│   └── requirements.txt
└── web/                         <- Phase 2 (Next.js)
```

---

## Sprint Overview

| Sprint | Layunin | Tinatayang tagal |
|---|---|---|
| **Sprint 0** | Setup ng environment at project | 1 araw |
| **Sprint 1** | Dataset: download, ayusin, i-split | 3-5 araw |
| **Sprint 2** | Baseline training (unang model) | 2-3 araw |
| **Sprint 3** | Fine-tuning at pag-compare ng models | 3-4 araw |
| **Sprint 4** | Real-world testing at finalize ng model | 3-5 araw |
| **Sprint 5** | Next.js setup + model integration | (Phase 2) |
| **Sprint 6** | Pest at pesticide database | (Phase 2) |
| **Sprint 7** | UI/UX, confidence check, wika | (Phase 2) |
| **Sprint 8** | Testing at deployment | (Phase 2) |

Flexible ang tagal. Beginner ka, kaya okay lang kung humaba. Huwag lumipat sa susunod na sprint hangga't hindi tapos ang **Definition of Done** ng kasalukuyan.

---

# PHASE 1: MODEL TRAINING

## Sprint 0: Setup

**Goal:** Gumagana ang Python environment at may project folder na.

- [x] Mag-install ng **Python 3.11** (i-check ang "Add Python to PATH" sa Windows): na-install via winget, Python 3.11.9
- [x] I-check: `python --version`: 3.11.9 (gamitin ang `venv\Scripts\python.exe`; mag-open ng bagong terminal para lumabas ang `python` sa PATH)
- [x] Gumawa ng project folder `eggplant-pest-app` at ang `ml/` structure sa itaas: ang `eggplant/` folder mismo ang project root
- [x] Gumawa ng virtual environment: `python -m venv venv`, tapos i-activate
- [x] Gumawa ng `ml/requirements.txt` at i-install: `tensorflow`, `matplotlib`, `scikit-learn`, `pandas`, `pillow`, `numpy` (naka-pin ang versions)
- [x] Test: `python -c "import tensorflow as tf; print(tf.__version__)"`: TensorFlow 2.21.0
- [x] I-check kung may GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`: `[]`, CPU lang (may RTX 2060 pero walang WSL2)
- [x] `git init` at gumawa ng `.gitignore` (isama ang `venv/`, `ml/data/`, `ml/models/`)

> **Tungkol sa GPU:** Sa native Windows, hindi na sinusuportahan ng bagong TensorFlow ang GPU. Kung may NVIDIA GPU ka at gusto mo itong gamitin, gumamit ng **WSL2**. Kung wala o ayaw mo, **CPU lang ay okay** para sa 1,500 images (mas mabagal lang). Puwede ring gumamit ng Google Colab kung masyadong mabagal ang laptop.

**Definition of Done:** Nakaka-print ang TensorFlow version at kumpleto ang folder structure.

---

## Sprint 1: Dataset

**Goal:** Kumpletong dataset na 200/50/50 bawat class, malinis, at walang leakage.

### 1A. Alamin muna kung sapat ang images
- [x] Gumawa ng `ml/DATA_SOURCES.md` para sa listahan ng sources at citations: nagawa na, pero may mga source na kailangan mong punan (tingnan ang "KAILANGAN MONG PUNAN")
- [x] I-download ang mga datasets (tingnan ang Sources sa baba): may sarili ka nang organized dataset, kaya Mendeley Eggplant Dataset (Original) at Tomato Insect Pests lang ang ginamit pandagdag. Hindi na kailangan ang Roboflow at BrinjalFruitX
- [x] **Bilangin ang available images bawat class.** Kailangan ng **300 bawat class** (200+50+50): pagkatapos linisin, may 214 (aphids), 220 (flea_beetle), 373 (fruit_shoot_borer), 276 (healthy), 300 (thrips) na unique at tamang larawan
- [x] Kung kulang ang isang class (malamang ang `flea_beetle` at `thrips`), isulat kung ilan ang kulang: kulang ng 86 (aphids), 80 (flea_beetle), 24 (healthy). Pinunan mula sa Mendeley

### 1B. Pag-ayos
- [x] Gumawa ng 5 folders sa `ml/data/raw_sorted/`, isa bawat class
- [x] Kopyahin ang tamang images sa bawat folder (hindi ang buong dataset, ang mga class lang na kailangan): ginawa ng `ml/scripts/rebuild_dataset.py` (hindi na dumaan sa `raw_sorted/`)
- [x] **Gamitin ang original images lang, hindi ang augmented versions.** Kung isasama ang augmented, magiging parehong-pareho ang ilang images sa train at test (leakage), at titingkad nang peke ang accuracy: Original folders lang ang ginamit sa Mendeley; ang mga flip/rotate na kopya ay hinuli ng near-duplicate check
- [ ] Sa Roboflow, **huwag mag-turn on ng Augmentation** sa export: hindi applicable, walang Roboflow dataset na ginamit
- [ ] Palitan ang filenames gamit ang source prefix (`bisu_001.jpg`, `brinjal_001.jpg`) para walang magkapareho: hindi ginawa. Pinanatili ang orihinal na pangalan; ang pinagmulan ng bawat file ay nasa `ml/results/dataset_rebuild_manifest.csv`
- [x] I-delete ang malabo, mali ang label, at maliit na crops (mas maliit sa mga 50x50 pixels): inalis ang 67 sakit (hindi pest), 37 stock photo at mga label conflict; walang maliit na crop (pinakamaliit na gilid ay 285px). Hindi ko sinuri kung malabo
- [ ] Mag-decide: cropped (Isolate Objects) o full images? Ang rekomendasyon ay **halo**, para pareho sa totoong photos ng farmer: full images lang ang laman ng dataset
- [x] Tanggalin ang **duplicates** (magkaparehong picture sa magkaibang datasets). Pwedeng gumamit ng hash check: exact hash at near-duplicate (kasama ang flip/rotate)

### 1C. Kung kulang pa ang images
- [ ] Kumuha ng sarili mong photos sa mga eggplant farm (Central Luzon), 100-300 bawat kulang na class
- [ ] Ipa-confirm sa agriculturist ang labels
- [ ] Iba-ibang anggulo, layo, liwanag, at phone

### 1D. I-split
- [x] I-split bawat class: **200 train / 50 valid / 50 test** (random, may fixed seed): block split ng magkakasunod na kuha, seed 42
- [x] **Mag-split bago mag-augment.** May augmented copy na sa disk (v4), pero ginawa ito pagkatapos ng split at sa train LANG; walang augmented sa valid at test
- [x] Siguraduhing **pantay ang bilang bawat class** at **pareho ang folder names** sa train, valid, at test: 200/50/50 sa lahat ng 6 classes (v4, tingnan ang 1E)
- [x] Isulat ang `ml/scripts/check_dataset.py` na nagbibilang ng images bawat folder at nagche-check ng:
  - [x] pare-pareho ang class folders sa tatlong splits
  - [x] walang file na hindi mabuksan
  - [x] walang parehong file hash sa train at test (anti-leakage); dinagdagan din ng near-duplicate check at babala para sa magkakalapit na kuha
- [x] I-run ang check at i-save ang output: PASS, nasa `ml/results/check_dataset.txt`

### 1E. Pagbalanse at healthy_leaf (v4, 2026-10-08)
Pagkatapos ng relabel (v3), hindi na pantay ang dataset (hal. aphids 261, flea_beetle 137 sa train) at halo-halo ang laki ng larawan (800x800, 512x512, 456x608). Dahon din pala ang 205 sa `healthy` (ang iba ay prutas). Ginawa ng `ml/scripts/balance_dataset.py` (mula sa `ml/data/dataset_v3`, hindi ito binago):
- [x] Inilipat sa bagong class na `healthy_leaf` ang 205 na dahon ("Healthy Leaf" ng Mafi et al. 2023 at "Fresh Eggplant Leaf" ng Nirob). Prutas na lang ang `healthy` (sinilip lahat)
- [x] aphids: tinanggal ang 61 sa train, puro Nirob photos (buo ang field photos)
- [x] flea_beetle: +63 augmented copy sa train (walang ibang totoong larawan); inuna ang field photos at Nirob para hindi puro "Insect Pest"
- [x] healthy_leaf: 40 inilipat mula train papuntang valid/test (buong block ng magkakasunod na numero), +95 augmented sa train. Halos kalahati ng train nito ay augmented, kasi 205 lang ang totoong healthy leaf
- [x] healthy: +5 augmented sa train
- [x] Lahat ay 224x224 (PIL LANCZOS, stretch; ganito rin ang resize sa `predict.py` at dapat sa website)
- [x] Pasado ang `check_dataset.py` (200/50/50, walang leakage o near-duplicate sa pagitan ng splits): `ml/results/check_dataset_v4.txt`. Listahan ng bawat larawan: `ml/results/dataset_v4_manifest.csv`

**Definition of Done:** Pasado ang `check_dataset.py` (200/50/50 bawat class, walang duplicate sa pagitan ng splits) at kumpleto ang `DATA_SOURCES.md`.

### Dataset Sources

| Class | Mga source |
|---|---|
| `fruit_shoot_borer` | BISU (PH): https://universe.roboflow.com/bohol-island-state-university-vgjlb/eggplant-disease-detection (class: Fruit borer) |
| | BrinjalFruitX: https://data.mendeley.com/datasets/ngc58fsxgd (Shoot and Fruit Borer) |
| | eggplant_damage_detect: https://universe.roboflow.com/asphyxiasea/eggplant_damage_detect (Fruit borer) |
| `thrips` | BISU (PH): link sa taas (class: Melon Thrips) |
| | Tomato Insect Pests: https://data.mendeley.com/datasets/ysk546n9np/2 (Thrips) |
| `aphids` | Eggplant Leaf Diseases (Roboflow): https://universe.roboflow.com/projects-wivhq/final-eggplant-leaf-diseases-dataset-fxule (Aphids) |
| | Cotton Aphid Images: https://data.mendeley.com/datasets/5wp23rgtpy/1 |
| `flea_beetle` | Eggplant Leaf Disease Classification: https://data.mendeley.com/datasets/pvsv534ccg (Flea Beetles, 84 images lang) |
| | Eggplant Dataset (Original folder lang): https://data.mendeley.com/datasets/5drkk544k8/1 |
| | Eggplant Leaf Diseases (Roboflow): link sa aphids (Flea Beetles) |
| | Flea Beetle Dataset: https://universe.roboflow.com/insect-7gvp4/flea-beetle-dd6so |
| `healthy` | BISU (PH): link sa taas (Healthy) |
| | Eggplant Leaf Disease Detection: https://data.mendeley.com/datasets/d3ypkphghb (Healthy Leaf) |
| | BrinjalFruitX (Healthy) |

**Paalala:** I-check ang license ng bawat dataset at i-cite sa paper at sa `DATA_SOURCES.md`. Ang mga Roboflow dataset ay kailangang i-fork sa free account bago i-export. Para sa Roboflow, i-export bilang **Folder Structure** (gamit ang "Isolate Objects" kung gusto ng crops).

---

## Sprint 2: Baseline Training

**Goal:** Unang gumaganang model na may test accuracy.

- [x] Isulat ang `ml/scripts/train.py`:
  - [x] I-load ang train/valid/test gamit ang `image_dataset_from_directory` (224x224, batch 32)
  - [x] Data augmentation layer (flip, rotation, zoom, contrast)
  - [x] MobileNetV2 (ImageNet weights, `include_top=False`) na naka-**freeze**
  - [x] Bagong top layer: GlobalAveragePooling2D, Dropout, Dense(6, softmax) (5 bago ang v4)
  - [x] Compile: Adam, sparse categorical crossentropy
  - [x] Callbacks: EarlyStopping at ReduceLROnPlateau
  - [x] I-save ang best model sa `ml/models/`
  - [x] I-save ang `class_names.json`
- [x] I-train ng ~15 epochs
- [x] I-plot ang training/validation accuracy at loss, i-save sa `ml/results/`
- [x] Isulat ang `evaluate.py`: test accuracy sa test set
- [x] Isulat sa notes ang unang resulta: **Baseline MobileNetV2: 80.8% test accuracy** (macro F1 0.808; pinakamababang recall: flea_beetle 66%)

**Definition of Done:** May naka-save na model at naka-record ang baseline test accuracy.

---

## Sprint 3: Fine-tuning at Pagkukumpara

**Goal:** Mapataas ang accuracy at mapili ang pinakamahusay na model.

- [x] **Fine-tuning:** i-unfreeze ang huling ~30 layers (BatchNorm layers ay naka-freeze pa rin), learning rate `1e-5`, 10 pang epochs
- [x] Gawin ang confusion matrix (numpy sa `evaluate.py`, hindi `sklearn.metrics`: na-block ng Application Control policy ng Windows ang DLL nito) at i-save bilang larawan
- [x] Gawin ang classification report (precision, recall, F1 bawat class)
- [x] Alamin kung aling mga pest ang nagkakapalitan: **aphids at flea_beetle** (sa lahat ng apat na model; sa EfficientNetB0 finetuned: flea_beetle → aphids 15, aphids → flea_beetle 9)
- [ ] Kung may mahinang class: dagdagan o linisin ang images ng class na iyon (balik sa Sprint 1 kung kailangan). Mahina pa rin ang flea_beetle sa v4: recall 58% at 62% (dalawang seed). Napagkakamalan itong aphids, at healthy_leaf kapag 1-3 butas lang ang dahon. Sa seed 43, 74% lang din ang aphids. Mahina rin ang model sa field photos na iba ang source (hal. aphids mula NEW_URBAN: 1/6 sa parehong seed), kaya posibleng source ang natututunan at hindi laki ng butas. Kailangan ng dagdag na flea_beetle at aphids photos mula sa farm (kasama ang kakaunti ang butas), na nilagyan ng label ng agriculturist
- [x] **Ulitin ang buong training gamit ang EfficientNetB0**
- [x] Gumawa ng comparison table (test set, 250 images; nasa `ml/results/comparison.md`):

| Model | Test accuracy | Macro F1 | Laki ng model (MB) | Bilis (ms/image, CPU) | Pinakamababang recall |
|---|---|---|---|---|---|
| MobileNetV2 baseline | 80.8% | 0.808 | 9.3 | 67.9 | 66% (flea_beetle) |
| MobileNetV2 finetuned | 82.4% | 0.824 | 20.8 | 71.1 | 64% (aphids) |
| EfficientNetB0 baseline | 83.6% | 0.837 | 16.3 | 76.3 | 68% (flea_beetle) |
| EfficientNetB0 finetuned | **86.0%** | **0.860** | 27.7 | 95.7 | 70% (flea_beetle) |

Tandaan: ang laki ay kasama ang optimizer state, at ang bilis ay sa Python/TensorFlow CPU, hindi sa browser. Sa 250 images, ~±4% ang error margin ng bawat accuracy.

Sa bagong dataset (v4, 2026-10-08: 6 classes, 300 test images), EfficientNetB0, dalawang seed:

| Model | Test accuracy | Macro F1 | Laki ng model (MB) | Bilis (ms/image, CPU) | Pinakamababang recall |
|---|---|---|---|---|---|
| EfficientNetB0 baseline, seed 42 | 84.7% | 0.842 | 16.3 | 78.3 | 54% (flea_beetle) |
| EfficientNetB0 finetuned, seed 42 | **86.3%** | **0.860** | 27.7 | 79.2 | 58% (flea_beetle) |
| EfficientNetB0 baseline, seed 43 | 83.7% | 0.835 | 16.3 | 82.3 | 60% (flea_beetle) |
| EfficientNetB0 finetuned, seed 43 | 84.7% | 0.845 | 27.7 | 91.2 | 62% (flea_beetle) |

Average ng finetuned: **85.5%**, kaya nasa hangganan pa lang ng 85% target. Hindi ito direktang maikukumpara sa table sa itaas: iba ang test set (6 classes). Halos puro prutas din pala ang dating `healthy`, kaya masyadong mataas ang dating 86.0%. Nasa `ml/results/comparison.md` ang lahat ng run.

- [ ] Pumili ng model para sa app (isaalang-alang ang accuracy **at** laki/bilis). Kandidato: `efficientnetb0_v4_finetuned` (86.3%; 84.7% sa seed 43), pero hindi pa final dahil mababa ang flea_beetle recall (58-62%)

**Mga target:**
- Test accuracy **≥ 85%**
- Recall bawat class **≥ 80%**
- Hindi malaki ang agwat ng train at validation accuracy (kung 99% vs 75%, overfitting iyon)

**Definition of Done:** May napiling model, kumpleto ang comparison table, confusion matrix, at classification report.

---

## Sprint 4: Real-World Testing at Finalize

**Goal:** Makasigurong gagana ang model sa totoong photos, hindi lang sa dataset, at maihanda ito para sa website.

### 4A. Real-world test
- [ ] Kumuha ng **hiwalay na set ng photos** mula sa mga farm (hindi kasama sa training), mga 20-30 bawat class
- [ ] I-run ang model sa mga photo na iyon gamit ang `predict.py`
- [ ] Itala ang accuracy (inaasahang mas mababa kaysa test set, normal iyon)
- [ ] Kung bumaba nang malaki: idagdag ang mga ganoong photo sa training, mag-retrain, at gumawa ng **bagong** hiwalay na real-world set para sa susunod na test

### 4B. Confidence threshold
- [x] Tingnan ang confidence ng tama vs maling hula: sa test set (300, `efficientnetb0_v4_finetuned`), average na 90.8% kapag tama at 68.1% kapag mali (`ml/results/test_predictions.csv`)
- [x] Pumili ng threshold (panimula: **60%**). Kapag mas mababa, sasabihin ng app na "Hindi sigurado, paki-retake ang photo": **70%** ang napili (default na sa `predict.py`). Kumpirmahin pa sa real-world photos (4A), kasi sa test set din ito pinili

| Threshold | Nasagot | Tama sa nasagot | Maling nakalusot | flea_beetle (nasagot, tama) |
|---|---|---|---|---|
| wala | 300/300 | 86.0% | 42 | 50/50, 60% |
| 60% | 269/300 | 90.0% | 27 | 37/50, 64.9% |
| **70%** | **251/300** | **91.6%** | **21** | **32/50, 71.9%** |
| 80% | 219/300 | 95.9% | 9 | 19/50, 89.5% |

- [x] Subukan din ang mga larawan na **hindi pest o hindi talong** para makita ang ugali ng model: sa 70%, "Hindi sigurado" ang 82% ng Wilt, 62% ng Leaf Spot, 57% ng Mosaic, 45% ng Small Leaf, at 20% lang ng White Mold (40 bawat sakit, mula sa Mafi et al. 2023), at 75% ng 36 na larawang hindi talong (Windows wallpapers). Ang natitira ay nahuhulaang pest o healthy (hal. White Mold → healthy_leaf, wallpapers → healthy/thrips). **Hindi sapat ang threshold lang.** Opsyon: magdagdag ng class na "iba pa" (ibang sakit + hindi talong), at isulat sa Scope and Limitations

### 4C. Export para sa website
- [x] I-save ang final model (`.keras`) at `class_names.json`: `ml/models/efficientnetb0_v4_finetuned.keras` (kasama ang `class_names.json` sa TF.js folder)
- [x] Subukang i-convert sa **TensorFlow.js** (`tensorflowjs_converter`): `ml/scripts/export_tfjs.py` (.keras → SavedModel → TF.js graph model). Output: `ml/models/tfjs/efficientnetb0_v4_finetuned_float16/` (**8.0 MB**; float32 ay 15.6 MB). Sa Windows, `pip install --no-deps` ang tensorflowjs (nasa docstring ng script), at inaayos ng script ang bias ng BatchNorm para gumana sa WASM backend
- [x] Siguraduhing pareho ang prediction ng orihinal at converted na model sa parehong test images: `ml/scripts/verify_tfjs.cjs` (Node, WASM backend). **300/300 pareho ang hula**, accuracy 86.00% sa dalawa, pinakamalaking pagkakaiba ng probability ay 0.55 percentage points (float16) at 0.001 (float32). Bilis: ~73 ms/image
- [x] Isulat ang resulta sa notes

### Decision Point: Paano ilalagay ang model sa Next.js?

| Opsyon | Maganda | Mahirap |
|---|---|---|
| **A. TensorFlow.js sa browser** | Walang server, gumagana kahit mahina ang internet kapag na-load na | Minsan may problema sa conversion mula Keras 3 |
| **B. Python API (FastAPI) + Next.js** | Pinaka-maaasahan, ginagamit ang `.keras` as-is | Kailangan ng hiwalay na server (hal. Hugging Face Spaces, Render) |
| **C. ONNX Runtime** | Magaan, puwede sa Next.js API route | Dagdag na conversion step |

Panimulang rekomendasyon: subukan ang **A** sa 4C. Kung pumalya ang conversion, gamitin ang **B**.

- [x] Pumili ng opsyon at isulat ang dahilan sa `plan.md`: **A (TensorFlow.js sa browser)**, gamit ang float16 model. Gumana ang conversion at pareho ang hula, kaya hindi na kailangan ng server. Sa website, parehong preprocessing ang gamitin: RGB, i-resize (stretch) sa 224x224 nang high-quality, raw 0-255 na float (nasa loob ng model ang normalization), tapos threshold na 70%. Hindi pa nasusubukan sa WebGL backend ng browser; i-check sa Sprint 5

**Definition of Done:** May final model, alam ang real-world accuracy, may confidence threshold, at napili ang paraan ng integration.

---

# PHASE 2: WEBSITE (Outline lang muna, i-detalye pagkatapos ng Phase 1)

## Sprint 5: Next.js Setup at Model Integration
- [x] `npx create-next-app` sa `web/`: Next.js 16.4 (App Router, TypeScript, Tailwind). Patakbuhin: `cd web && npm run dev` (tingnan ang `web/README.md`)
- [x] Page na may upload at camera capture: "Kumuha ng litrato" (camera sa phone) at "Pumili ng larawan" (`web/src/components/pest-detector.tsx`)
- [x] I-load ang model (ayon sa napiling opsyon) at mag-predict: TF.js float16 sa `web/public/model/` (8 MB), WebGL backend. Lanczos3 resize gamit ang `pica`, para pareho sa training; noong canvas `drawImage` ang gamit, hanggang 19 points ang agwat sa confidence
- [x] Ipakita ang pest name at confidence: kasama ang top 3, at "Hindi sigurado, paki-retake" kapag mas mababa sa 70%
- [x] Nasubukan sa Chrome (WebGL): pareho ang sagot ng website at ng `predict.py` sa 4 na test image (224x224 hanggang 800x800), at hindi hihigit sa 0.6 points ang agwat ng confidence
- [ ] Subukan sa totoong phone (camera, Android at iPhone)

## Sprint 6: Pest at Pesticide Database
- [x] Gumawa ng data table: pest → paglalarawan, sintomas, **IPM methods muna**, chemical options, kaligtasan: **DRAFT** sa `web/src/lib/pests.ts` (palatandaan, IPM, biological/mababang-toxicity, active ingredient + IRAC group, kaligtasan), at may larawan ng peste mula sa Wikimedia Commons (CC BY / CC BY-SA, may credit sa app)
- [ ] Gamitin lang ang **FPA-registered** na active ingredients para sa talong (i-verify sa opisyal na listahan ng FPA): hindi pa na-verify ang mga nakalista
- [x] Isama ang pre-harvest interval, PPE, at paalala na kumonsulta sa local agriculturist: "sundin ang PHI sa label" (walang tiyak na bilang), PPE, MAO
- [ ] **Ipa-validate sa agriculturist** (Municipal Agriculture Office, PhilRice, o state university) bago i-publish: may "DRAFT" na paalala sa app hangga't hindi pa
- [x] Hindi dapat ilagay ang dosage nang hindi galing sa label o validated na source: walang dosis at walang brand

## Sprint 7: UI/UX
- [x] Mobile-first design (farmers ang gagamit, phone ang gamit): modernong berdeng UI (Tailwind, lucide icons), may dark mode; hindi pa nasusubukan sa totoong phone
- [x] "Hindi sigurado, retake" na mensahe kapag mababa ang confidence: mas mababa sa 70%, at walang rekomendasyon hangga't hindi sigurado
- [x] Sample photo ng pest para ihambing ng user: `web/public/pests/` (aphids, flea beetle, EFSB, thrips)
- [ ] Tagalog (at posibleng Kapampangan) na wika: Tagalog na ang UI; wala pang Kapampangan
- [x] Tips kung paano kumuha ng magandang photo (close-up, malinaw, may liwanag)

## Sprint 8: Testing at Deployment
- [ ] Subukan sa totoong phones at sa totoong farmers
- [ ] I-deploy (Vercel para sa Next.js, at hiwalay na host kung may Python API)
- [ ] Final na dokumentasyon at paglista ng limitasyon

---

## Mga Panganib (Risks)

| Panganib | Paano iiwasan |
|---|---|
| Kulang ang images ng `flea_beetle` o `thrips` | Bilangin agad sa Sprint 1A, mag-collect ng sariling photos |
| **Data leakage** (parehong picture sa train at test) | Original images lang, split bago augment, hash check |
| Mataas sa dataset pero mababa sa totoong photo | Real-world test set sa Sprint 4, dagdagan ng sariling farm photos |
| Hindi pantay ang classes | Target na 200/50/50 bawat isa, may class weights |
| Napagkakamalan ang mga magkahawig na pest (thrips/aphids) | Tingnan ang confusion matrix, dagdagan ang malinaw na images |
| Hindi kilala ng model ang ibang pest (hal. leafhopper) | Confidence threshold, ilagay sa Scope and Limitations |
| Maling pesticide recommendation | Ipa-validate sa agriculturist, FPA-registered lang, IPM muna |
| Mabagal ang training sa laptop | Gumamit ng Colab o WSL2 GPU |
| Pumalya ang TF.js conversion | Fallback sa FastAPI (Opsyon B) |

---

## Scope at Limitasyon (para sa paper)

- Kinikilala lang ng sistema ang **aphids, flea beetle, fruit and shoot borer, thrips**, at **healthy** (hiwalay ang prutas at dahon na walang pest: `healthy` at `healthy_leaf`).
- Ang mga pest tulad ng leafhopper, whitefly, at Epilachna beetle ay hindi sakop at irerekomenda para sa susunod na pag-aaral.
- Ang pesticide recommendations ay gabay lamang at kailangang i-validate ng lokal na agriculturist.

---

## Progress Tracker

- [x] Sprint 0: Setup
- [ ] Sprint 1: Dataset (halos tapos: v4 na, 6 classes kasama ang `healthy_leaf`, 200/50/50, lahat ay 224x224, pasado ang `check_dataset.py` at walang leakage. Natitira: punan mo ang mga source at license na hindi ko alam sa `ml/DATA_SOURCES.md` (wala pa roon ang Mafi et al. 2023 at ang `healthy_leaf`), at ipa-confirm sa agriculturist ang mga label. Tingnan ang `PHASE1_SUMMARY.md` at 1E)
- [x] Sprint 2: Baseline Training (baseline MobileNetV2: 80.8% test accuracy; nagkakatugma ang preprocessing ng `predict.py` at `evaluate.py`)
- [ ] Sprint 3: Fine-tuning at Comparison (halos tapos: sa v4, EfficientNetB0 finetuned ay 86.3% at 84.7% sa dalawang seed (average 85.5%), pero flea_beetle recall ay 58-62% (target ≥ 80%). Natitira: ayusin ang flea_beetle at pumili ng final na model)
- [ ] Sprint 4: Real-World Testing at Finalize (tapos ang 4B: threshold 70%, at 4C: TF.js float16, 300/300 pareho ang hula. Natitira: 4A real-world test gamit ang sariling farm photos)
- [ ] Sprint 5: Next.js + Integration (halos tapos: gumagana ang website sa desktop Chrome. Natitira: subukan sa totoong phone)
- [ ] Sprint 6: Pest at Pesticide Database
- [ ] Sprint 7: UI/UX
- [ ] Sprint 8: Testing at Deployment
