// Impormasyon at rekomendasyon bawat class ng model. Ang key ay pangalan sa public/model/class_names.json.
//
// DRAFT ang mga rekomendasyon: kailangang i-review ng agriculturist bago gamitin ng farmers (plan.md, Sprint 6).
// IPM muna. Active ingredient lang ang nakalista (walang brand, walang dosis): bago gamitin, i-check na rehistrado
// ito sa FPA para sa talong at sundin ang label. Ang "group" ay IRAC mode of action, para sa pag-rotate.

export type Chemical = { ingredient: string; group: string; note?: string };

export type PestImage = {
  src: string;
  caption: string;
  credit: string;
  license: string;
  licenseUrl: string;
  source: string;
};

export type PestInfo = {
  name: string;
  scientific?: string;
  part: string; // saang bahagi ng halaman makikita
  healthy: boolean;
  signs: string[];
  ipm: string[]; // gawin muna, walang kemikal
  lowToxic: string[]; // biological o mababang-toxicity
  chemical: Chemical[]; // huling opsyon
  image?: PestImage;
};

export const PESTS: Record<string, PestInfo> = {
  aphids: {
    name: "Aphids",
    scientific: "Aphis gossypii at iba pa",
    part: "dahon",
    healthy: false,
    signs: [
      "Ayon sa agriculturist ng project: maliliit na butas sa dahon",
      "Kumpol ng maliliit at malalambot na insekto sa ilalim ng dahon at sa talbos",
      "Malagkit na dagta (honeydew) at itim na amag sa dahon",
    ],
    ipm: [
      "Silipin ang ilalim ng dahon at ang talbos dalawang beses kada linggo",
      "Kapag kaunti pa lang: tanggalin ang apektadong dahon o i-spray ng malakas na tubig",
      "Alagaan ang mga kaaway ng aphids (ladybird beetle, lacewing): iwasan ang malawakang pag-spray",
      "Huwag sobrahan ang nitrogen (urea); dumarami ang aphids sa malalambot na bagong dahon",
      "Maglagay ng yellow sticky trap para makita kung dumarami",
    ],
    lowToxic: ["Sabong pang-insekto (insecticidal soap) o neem oil (azadirachtin); tamaan ang ilalim ng dahon"],
    chemical: [
      { ingredient: "Imidacloprid", group: "4A", note: "Nakakasama sa bubuyog; huwag i-spray habang namumulaklak" },
      { ingredient: "Flonicamid", group: "29" },
    ],
    image: {
      src: "/pests/aphids.jpg",
      caption: "Cotton aphid (Aphis gossypii) sa ilalim ng dahon",
      credit: "Dr Lewis Wilson, CSIRO",
      license: "CC BY 3.0",
      licenseUrl: "https://creativecommons.org/licenses/by/3.0",
      source: "https://commons.wikimedia.org/wiki/File:CSIRO_ScienceImage_7331_Aphids_on_cotton_7.jpg",
    },
  },
  flea_beetle: {
    name: "Flea beetle",
    scientific: "Epitrix at iba pa",
    part: "dahon",
    healthy: false,
    signs: [
      "Ayon sa agriculturist ng project: malalaking butas sa dahon",
      "Maliliit at makintab na itim na salagubang na tumatalon kapag ginalaw ang dahon",
      "Pinakamapanganib sa punla at batang halaman",
    ],
    ipm: [
      "Alisin ang damo at mga tirang halaman sa paligid; doon nagtatago ang salagubang",
      "Takpan ng net (row cover) ang punla sa unang 3-4 na linggo pagkatapos itanim",
      "Maglagay ng yellow o white sticky trap",
      "Mag-rotate ng pananim: huwag magtanim ng talong, kamatis, o sili sa parehong lugar nang sunod-sunod",
      "Kaya ng malaking halaman ang kaunting butas; mag-spray lang kapag marami na o kapag punla ang tinatamaan",
    ],
    lowToxic: ["Neem oil (azadirachtin)"],
    chemical: [
      { ingredient: "Spinosad", group: "5", note: "Mas ligtas sa kaibigang kulisap kaysa sa pyrethroid" },
      {
        ingredient: "Lambda-cyhalothrin",
        group: "3A",
        note: "Huling opsyon: pinapatay din nito ang kaibigang kulisap",
      },
    ],
    image: {
      src: "/pests/flea_beetle.jpg",
      caption: "Eggplant flea beetle (Epitrix fuscula)",
      credit: "Robert Webster",
      license: "CC BY-SA 4.0",
      licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0",
      source: "https://commons.wikimedia.org/wiki/File:Epitrix_fuscula_-_inat_73678456.jpg",
    },
  },
  fruit_shoot_borer: {
    name: "Fruit and shoot borer (EFSB)",
    scientific: "Leucinodes orbonalis",
    part: "bunga at talbos",
    healthy: false,
    signs: [
      "Nalalantang dulo ng talbos na may butas",
      "Butas sa bunga na may dumi ng uod (frass)",
      "Kulay rosas na uod sa loob ng bunga",
    ],
    ipm: [
      "Linggu-linggo: putulin ang nalalantang talbos at pitasin ang butas na bunga. Huwag iwan sa bukid: ibaon nang malalim o ilagay sa saradong sako",
      "Maglagay ng sex pheromone trap para sa EFSB para mahuli ang lalaking gamugamo at makita kung dumarami",
      "Gumamit ng malinis na punla at protektahan ang punlaan ng net",
      "Tanggalin ang mga lumang halaman pagkatapos ng anihan",
      "Iwasan ang madalas na pag-spray ng pyrethroid: pinapatay nito ang kaaway ng EFSB at nagkakaroon ng resistance",
    ],
    lowToxic: [
      "Bacillus thuringiensis (Bt, Group 11A): i-spray habang bata pa ang uod, bago pumasok sa bunga",
      "Trichogramma (maliit na wasp na sumisira sa itlog ng EFSB): magtanong sa DA o BPI kung may release program",
    ],
    chemical: [
      { ingredient: "Chlorantraniliprole", group: "28" },
      { ingredient: "Emamectin benzoate", group: "6" },
      { ingredient: "Spinosad", group: "5" },
    ],
    image: {
      src: "/pests/fruit_shoot_borer.jpg",
      caption: "Uod ng EFSB (Leucinodes orbonalis) sa loob ng bunga ng talong",
      credit: "Chirag85",
      license: "CC BY-SA 4.0",
      licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0",
      source: "https://commons.wikimedia.org/wiki/File:Leucinodes_orbonalis_(Brinjal_Fruit_Borer).jpg",
    },
  },
  healthy: {
    name: "Malusog na bunga",
    part: "bunga",
    healthy: true,
    signs: [],
    ipm: [
      "Walang kailangang i-spray. Huwag mag-spray \"para sigurado\": sayang ang gastos at pinapatay nito ang kaibigang kulisap",
      "Patuloy na silipin ang tanim dalawang beses kada linggo (dahon, talbos, at bunga)",
      "Panatilihing malinis ang taniman: alisin ang damo at mga bahaging may sira",
      "Balanseng abono at tamang dilig",
    ],
    lowToxic: [],
    chemical: [],
  },
  healthy_leaf: {
    name: "Malusog na dahon",
    part: "dahon",
    healthy: true,
    signs: [],
    ipm: [
      "Walang kailangang i-spray. Huwag mag-spray \"para sigurado\": sayang ang gastos at pinapatay nito ang kaibigang kulisap",
      "Patuloy na silipin ang ilalim ng dahon at ang talbos dalawang beses kada linggo",
      "Panatilihing malinis ang taniman: alisin ang damo at mga bahaging may sira",
      "Balanseng abono at tamang dilig",
    ],
    lowToxic: [],
    chemical: [],
  },
  thrips: {
    name: "Thrips",
    scientific: "Thrips palmi at iba pa",
    part: "bunga",
    healthy: false,
    signs: [
      "Peklat o gasgas (kulay pilak o kayumanggi) sa balat ng bunga, lalo na malapit sa takip (calyx)",
      "Napakaliliit at payat na insekto (mga 1 mm) sa bulaklak at ilalim ng dahon",
      "Kulot o naninilaw na batang dahon",
    ],
    ipm: [
      "Silipin ang bulaklak, ilalim ng dahon, at balat ng bunga malapit sa takip",
      "Maglagay ng blue o yellow sticky trap",
      "Alisin ang damo sa paligid; doon dumarami ang thrips",
      "Gumamit ng silver (reflective) na plastic mulch",
      "Huwag sobrahan ang nitrogen na abono",
    ],
    lowToxic: ["Neem oil o sabong pang-insekto (insecticidal soap)"],
    chemical: [
      { ingredient: "Spinetoram o spinosad", group: "5" },
      { ingredient: "Abamectin", group: "6" },
      { ingredient: "Imidacloprid", group: "4A", note: "Nakakasama sa bubuyog; huwag i-spray habang namumulaklak" },
    ],
    image: {
      src: "/pests/thrips.jpg",
      caption: "Melon thrips (Thrips palmi)",
      credit: "Florida Division of Plant Industry, FDACS, Bugwood.org",
      license: "CC BY 3.0 US",
      licenseUrl: "https://creativecommons.org/licenses/by/3.0/us/",
      source: "https://commons.wikimedia.org/wiki/File:Thrips_palmi.jpg",
    },
  },
};

// Para sa lahat ng pesticide, kahit anong peste.
export const SAFETY = [
  "Gumamit lang ng pesticide na rehistrado sa FPA (Fertilizer and Pesticide Authority) para sa talong",
  "Basahin at sundin ang label: dosis, at ilang araw bago puwedeng anihin (pre-harvest interval)",
  "Magsuot ng guwantes, mask, long sleeves, pantalon, at bota",
  "Huwag mag-spray kapag mahangin, uulan, o tanghaling tapat. Ilayo sa bata, hayop, at pinagkukunan ng tubig",
  "I-rotate ang Group ng pesticide para hindi magka-resistance ang peste",
  "Kumonsulta sa Municipal Agriculture Office (MAO) bago gumamit ng kemikal",
];

// Napili sa Sprint 4B (plan.md): sa test set, 84% ang nasasagot at 91.6% ang tama sa nasagot.
export const THRESHOLD = 0.7;

// Laki ng input ng model (224x224, RGB, raw 0-255; nasa loob ng model ang normalization).
export const IMG_SIZE = 224;
