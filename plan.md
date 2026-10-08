# Eggplant Pest Detection Web App: Project Plan

Isang website (Next.js) kung saan mag-a-upload o kukuha ng picture ang user, ire-recognize ng model ang eggplant pest, at magbibigay ng tamang pesticide (IPM muna) para sa pest na iyon.

**Phase 1 (detalyado): Model training, locally**
**Phase 2 (outline lang muna): Next.js website + pesticide recommendation**

---

## Mga Class (5)

| # | Class (folder name) | Pest / kondisyon |
|---|---|---|
| 1 | `aphids` | Aphids |
| 2 | `flea_beetle` | Flea beetle |
| 3 | `fruit_shoot_borer` | Eggplant fruit and shoot borer (EFSB) |
| 4 | `healthy` | Walang pest |
| 5 | `thrips` | Thrips |

**Target dataset: 200 train / 50 valid / 50 test bawat class**
Total: 1,000 train + 250 valid + 250 test = **1,500 images**

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
│   │       ├── train/{aphids, flea_beetle, fruit_shoot_borer, healthy, thrips}
│   │       ├── valid/{...parehong 5}
│   │       └── test/{...parehong 5}
│   ├── scripts/
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
- [x] Gumawa ng `ml/DATA_SO URCES.md` para sa listahan ng sources at citations: nagawa na, pero may mga source na kailangan mong punan (tingnan ang "KAILANGAN MONG PUNAN")
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
- [x] **Mag-split bago mag-augment.** Walang augmentation na ginawa sa disk
- [x] Siguraduhing **pantay ang bilang bawat class** at **pareho ang folder names** sa train, valid, at test
- [x] Isulat ang `ml/scripts/check_dataset.py` na nagbibilang ng images bawat folder at nagche-check ng:
  - [x] pare-pareho ang class folders sa tatlong splits
  - [x] walang file na hindi mabuksan
  - [x] walang parehong file hash sa train at test (anti-leakage); dinagdagan din ng near-duplicate check at babala para sa magkakalapit na kuha
- [x] I-run ang check at i-save ang output: PASS, nasa `ml/results/check_dataset.txt`

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
  - [x] Bagong top layer: GlobalAveragePooling2D, Dropout, Dense(5, softmax)
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
- [ ] Kung may mahinang class: dagdagan o linisin ang images ng class na iyon (balik sa Sprint 1 kung kailangan). Mahina: flea_beetle (recall 70%). Hindi kita sa data ang source shortcut; posibleng label noise o mababang resolution, kaya ipa-review sa agriculturist at subukan ang mas malaking input
- [x] **Ulitin ang buong training gamit ang EfficientNetB0**
- [x] Gumawa ng comparison table (test set, 250 images; nasa `ml/results/comparison.md`):

| Model | Test accuracy | Macro F1 | Laki ng model (MB) | Bilis (ms/image, CPU) | Pinakamababang recall |
|---|---|---|---|---|---|
| MobileNetV2 baseline | 80.8% | 0.808 | 9.3 | 67.9 | 66% (flea_beetle) |
| MobileNetV2 finetuned | 82.4% | 0.824 | 20.8 | 71.1 | 64% (aphids) |
| EfficientNetB0 baseline | 83.6% | 0.837 | 16.3 | 76.3 | 68% (flea_beetle) |
| EfficientNetB0 finetuned | **86.0%** | **0.860** | 27.7 | 95.7 | 70% (flea_beetle) |

Tandaan: ang laki ay kasama ang optimizer state, at ang bilis ay sa Python/TensorFlow CPU, hindi sa browser. Sa 250 images, ~±4% ang error margin ng bawat accuracy.

- [ ] Pumili ng model para sa app (isaalang-alang ang accuracy **at** laki/bilis). Kandidato: EfficientNetB0 finetuned (ang tanging pumasa sa 85%), pero hindi pa final dahil sa flea_beetle recall

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
- [ ] Tingnan ang confidence ng tama vs maling hula
- [ ] Pumili ng threshold (panimula: **60%**). Kapag mas mababa, sasabihin ng app na "Hindi sigurado, paki-retake ang photo"
- [ ] Subukan din ang mga larawan na **hindi pest o hindi talong** para makita ang ugali ng model

### 4C. Export para sa website
- [ ] I-save ang final model (`.keras`) at `class_names.json`
- [ ] Subukang i-convert sa **TensorFlow.js** (`tensorflowjs_converter`)
- [ ] Siguraduhing pareho ang prediction ng orihinal at converted na model sa parehong test images
- [ ] Isulat ang resulta sa notes

### Decision Point: Paano ilalagay ang model sa Next.js?

| Opsyon | Maganda | Mahirap |
|---|---|---|
| **A. TensorFlow.js sa browser** | Walang server, gumagana kahit mahina ang internet kapag na-load na | Minsan may problema sa conversion mula Keras 3 |
| **B. Python API (FastAPI) + Next.js** | Pinaka-maaasahan, ginagamit ang `.keras` as-is | Kailangan ng hiwalay na server (hal. Hugging Face Spaces, Render) |
| **C. ONNX Runtime** | Magaan, puwede sa Next.js API route | Dagdag na conversion step |

Panimulang rekomendasyon: subukan ang **A** sa 4C. Kung pumalya ang conversion, gamitin ang **B**.

- [ ] Pumili ng opsyon at isulat ang dahilan sa `plan.md`

**Definition of Done:** May final model, alam ang real-world accuracy, may confidence threshold, at napili ang paraan ng integration.

---

# PHASE 2: WEBSITE (Outline lang muna, i-detalye pagkatapos ng Phase 1)

## Sprint 5: Next.js Setup at Model Integration
- [ ] `npx create-next-app` sa `web/`
- [ ] Page na may upload at camera capture
- [ ] I-load ang model (ayon sa napiling opsyon) at mag-predict
- [ ] Ipakita ang pest name at confidence

## Sprint 6: Pest at Pesticide Database
- [ ] Gumawa ng data table: pest → paglalarawan, sintomas, **IPM methods muna**, chemical options, kaligtasan
- [ ] Gamitin lang ang **FPA-registered** na active ingredients para sa talong (i-verify sa opisyal na listahan ng FPA)
- [ ] Isama ang pre-harvest interval, PPE, at paalala na kumonsulta sa local agriculturist
- [ ] **Ipa-validate sa agriculturist** (Municipal Agriculture Office, PhilRice, o state university) bago i-publish
- [ ] Hindi dapat ilagay ang dosage nang hindi galing sa label o validated na source

## Sprint 7: UI/UX
- [ ] Mobile-first design (farmers ang gagamit, phone ang gamit)
- [ ] "Hindi sigurado, retake" na mensahe kapag mababa ang confidence
- [ ] Sample photo ng pest para ihambing ng user
- [ ] Tagalog (at posibleng Kapampangan) na wika
- [ ] Tips kung paano kumuha ng magandang photo (close-up, malinaw, may liwanag)

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

- Kinikilala lang ng sistema ang **aphids, flea beetle, fruit and shoot borer, thrips**, at **healthy**.
- Ang mga pest tulad ng leafhopper, whitefly, at Epilachna beetle ay hindi sakop at irerekomenda para sa susunod na pag-aaral.
- Ang pesticide recommendations ay gabay lamang at kailangang i-validate ng lokal na agriculturist.

---

## Progress Tracker

- [x] Sprint 0: Setup
- [ ] Sprint 1: Dataset (halos tapos: pasado ang `check_dataset.py` na 200/50/50 at walang leakage. Natitira: punan mo ang mga source at license na hindi ko alam sa `ml/DATA_SOURCES.md`, at ipa-confirm sa agriculturist ang mga label. Tingnan ang `PHASE1_SUMMARY.md`)
- [x] Sprint 2: Baseline Training (baseline MobileNetV2: 80.8% test accuracy; nagkakatugma ang preprocessing ng `predict.py` at `evaluate.py`)
- [ ] Sprint 3: Fine-tuning at Comparison (halos tapos: EfficientNetB0 finetuned ang pinakamahusay sa 86.0%, pero flea_beetle recall ay 70% (target ≥ 80%). Natitira: ayusin ang flea_beetle at pumili ng final na model)
- [ ] Sprint 4: Real-World Testing at Finalize
- [ ] Sprint 5: Next.js + Integration
- [ ] Sprint 6: Pest at Pesticide Database
- [ ] Sprint 7: UI/UX
- [ ] Sprint 8: Testing at Deployment
