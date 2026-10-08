# Eggplant Pest Detector (website)

Next.js app na kumikilala ng peste sa talong mula sa litrato. Sa browser mismo tumatakbo ang model (TensorFlow.js), kaya walang server na kailangan para sa prediction.

## Paano patakbuhin

```bash
cd web
npm install
npm run dev
```

Buksan ang URL na ipapakita (karaniwan ay http://localhost:3000, o ibang port kung may gumagamit na ng 3000).

Para sa production build: `npm run build`, tapos `npm start`.

## Paano gumagana

- `src/components/pest-detector.tsx`: kumukuha ng litrato (camera o gallery), nire-resize sa 224x224 gamit ang Lanczos3 (`pica`, kapareho ng PIL LANCZOS sa training), at nagpe-predict gamit ang `public/model/`.
- Kapag mas mababa sa 70% ang confidence (`THRESHOLD` sa `src/lib/pests.ts`), "Hindi sigurado, paki-retake ang photo" ang lalabas.
- `src/lib/pests.ts`: pangalan, palatandaan, at rekomendasyon bawat class (IPM muna, tapos active ingredient + IRAC group, walang dosis), at credit ng larawan. **DRAFT** pa ito: ipa-review sa agriculturist at i-check sa listahan ng FPA bago gamitin ng farmers. Dito rin ito babaguhin.
- `src/components/pest-result.tsx`: resulta, larawan ng peste, at rekomendasyon.
- `public/pests/`: larawan ng bawat peste mula sa Wikimedia Commons (CC BY 3.0 / CC BY-SA 4.0 / CC BY 3.0 US). Kailangang manatiling nakikita ang credit sa app.

## Paano palitan ang model

```bash
python ml/scripts/export_tfjs.py --model ml/models/<model>.keras
```

Tapos kopyahin ang laman ng `ml/models/tfjs/<model>_float16/` papunta sa `web/public/model/`. Kasama roon ang `class_names.json`, kaya awtomatikong susunod ang pagkakasunod ng mga class.
