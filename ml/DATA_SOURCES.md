# Data Sources

Final dataset: **1,500 images**, 5 classes, 200 train / 50 valid / 50 test bawat class.
Nasa `ml/data/dataset/` (resized, max 1024px) at sa `C:\Users\laman\OneDrive\Pictures\eggplant_pest_dataset` (full resolution).
Ang buong listahan ng bawat file (saan galing, ano ang nangyari) ay nasa `ml/results/dataset_rebuild_manifest.csv`.

## Composition bawat class (galing sa manifest)

| Class | Kabuuan | Mga source (bilang) |
|---|---|---|
| `aphids` | 300 | Eggplant Dataset / Aphids (218: 149 sa dataset ng user + 69 dinagdag), NEW_URBAN_ORIG (35), EGGPLANT_ANGAT_ORIG (12), EGGPLANT_URBAN_ORIG (9), NEW_KNL_ORIG (8), EGGPLANT_CHURCH_ORIG (1), Tomato Insect Pests / Peach Aphid (17) |
| `flea_beetle` | 300 | Eggplant Dataset / Flea Beetles (243: 163 + 80 dinagdag), EGGPLANT_ANGAT_ORIG (47), NEW_URBAN_ORIG (6), EGGPLANT_URBAN_ORIG (2), NEW_KNL_ORIG (1), isang web photo (1) |
| `fruit_shoot_borer` | 300 | Camera photos: 2023-02-20 (197), 2023-03-10 (97), 2023-02-17 (4), 2022-11-09 (1); isang file na `1637591590-344351.jpg` (1) |
| `healthy` | 300 | Camera photos: 2023-02-17 (103), 2023-02-20 (99), 2023-03-10 (73); Eggplant Dataset / Fresh Eggplant (20 prutas), Fresh Eggplant Leaf (5) |
| `thrips` | 300 | Camera photos: 2023-02-17 (159), 2023-02-20 (128), 2023-03-10 (12), 2022-11-09 (1) |

## Mga dataset na may kumpletong citation (CC BY 4.0)

Parehong **CC BY 4.0**. Kailangang i-cite sa paper.

1. Nirob, M. A. S., Bishshash, P., Ayan, M. B., Khatun, T., & Uddin, M. S. (2024). *Eggplant Dataset: A Comprehensive Dataset for Agricultural Research and Disease Detection* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/5drkk544k8.1
   - Ginamit: folder na `Original Dataset` lang (Aphids, Flea Beetles, Fresh Eggplant, Fresh Eggplant Leaf). Hindi ginamit ang `Augmented Dataset`.
   - Ang mga file ng user na `Aphids_original*`, `Flea Beetles_original*` at `Fresh Eggplant*_original*` ay tugma sa dataset na ito sa pangalan at sa nilalaman (maraming byte-identical na file), kaya itinuring kong galing dito. **I-confirm mo.**

2. Islam, M. M., Rahman, M. A., Akter, M., & Uddin, M. S. (2026). *A Comprehensive Augmented Image Dataset of Tomato Insect Pests for Deep Learning-Based Detection* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/ysk546n9np.2
   - Ginamit: folder na `Original_image_of_tomato_pest/Peach_Aphid` lang, 17 larawan na idinagdag sa `aphids`. Hindi ginamit ang `Raw & Aug` at ang `tomato_agu(1000)` (augmented).
   - Tandaan: 156 sa mga larawan sa Peach_Aphid folder ay kapareho ng mga nasa Thrips folder (thrips ang mga ito, hindi aphids). Inalis ko ang mga iyon.

## KAILANGAN MONG PUNAN (hindi ko alam ang source / license)

| Mga file | Class | Bilang | Ano ang alam ko | Ano ang kailangan |
|---|---|---|---|---|
| `EGGPLANT_ANGAT_ORIG (n)`, `EGGPLANT_URBAN_ORIG (n)`, `EGGPLANT_CHURCH_ORIG (n)`, `NEW_URBAN_ORIG (n)`, `NEW_KNL_ORIG (n)` | aphids, flea_beetle | 65 aphids + 56 flea_beetle | Mga pangalan lang. Mukhang iisang series ng field photos (Angat, Urban, Church, KNL). | Sino ang kumuha o saang dataset galing, at ang license |
| Camera photos (`IMG2023....jpg`, `IMG_2023..._....jpg`, `2023..._....jpg`) | fruit_shoot_borer, healthy, thrips | 299 + 275 + 300 | Ayon sa petsa sa filename: 2022-11-09, 2023-02-17, 2023-02-20, 2023-03-10. Kailangan mong kumpirmahin kung ikaw ang kumuha (at saang farm) o galing sa BISU/ibang dataset. | Sino ang kumuha, saang farm, at ang license |
| `1637591590-344351.jpg` | fruit_shoot_borer | 1 | Mukhang web image | Source at license, o alisin |
| `Flea beetles and feeding injury to eggplant_29363261285_o.jpg` | flea_beetle | 1 | Mukhang Flickr photo (may "_o" at numeric ID) | Source at license, o alisin |

Walang Roboflow dataset (BISU, eggplant_damage_detect, atbp.) na ginamit sa final dataset.
Na-download din pero **hindi ginamit**: BrinjalFruitX (hindi natapos), Cotton Aphid Images (cotton ang halaman at may kasamang videos), Eggplant Leaf Disease Classification, at Eggplant Leaf Disease Detection (huminto ako, hindi natapos).

## Mga tinanggal at bakit

| Dahilan | Bilang | Ano ang ginawa |
|---|---|---|
| Duplicate at near-duplicate (kasama ang flip/rotate) | 123 sa dataset ng user | Isang kopya lang ang itinira |
| Larawan ng ibang sakit (Cercospora, Leaf Wilt, Powdery mildew, Wilt Disease, Mosaic) na nasa pest classes | 67 | Inalis. Hindi sakop ng 5 classes |
| Stock photos (`istockphoto-*`, ID na tulad ng `3HTJ19EY2ZMT.jpg`, `brinjal-51-*`) sa healthy | 37 | Inalis. Isyu sa license |
| Sobra sa 300 sa isang class (fruit_shoot_borer) | 73 | Hindi ginamit |
| Hindi larawan (`.ipynb`; at ang `.pdf` na inalis mo na) | 1 | Inalis sa dataset |

Itinama rin ang class ng 38 larawan batay sa pangalan ng file: 34 `Aphids_original*` na nasa flea_beetle ang inilipat sa aphids, 2 `Flea Beetles_original*` na nasa aphids ang inilipat sa flea_beetle, at 2 `Fresh Eggplant Leaf_original*` na nasa aphids ang inilipat sa healthy. Siguraduhing tama ang mga iyon.

Lahat ng inalis na galing sa dataset mo ay nasa `C:\Users\laman\OneDrive\Pictures\eggplant_pest_dataset_rejected\_by_reason\<dahilan>\`. Walang binura.

## Paano hinati (split)

- Hinati sa **blocks** ng 10 magkakasunod na kuha (ayon sa timestamp o numero ng file), at ang bawat block ay nasa iisang split lang. Hindi pwedeng mapunta sa train ang isang kuha at sa test ang kuha ilang segundo lang ang pagitan.
- Sinubukan ang 400 random na pagkakasunod ng blocks at pinili ang may pinakapantay na source mix sa train/valid/test.
- Fixed seed (42). Ang `ml/scripts/rebuild_dataset.py` ay ang eksaktong paraan.
- Walang augmentation na naka-save sa disk. Gagawin lang ito sa memory habang nagte-train.
