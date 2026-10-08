/* Sprint 4C: i-check na pareho ang hula ng TF.js model at ng Keras model sa parehong test images.
 *
 * Gawa ng export_tfjs.py ang verify folder (inputs.u8 = parehong pixels na ibinigay sa Keras, at reference.json).
 * Kailangan ang @tensorflow/tfjs at @tensorflow/tfjs-backend-wasm (kahit saang folder naka-install):
 *   npm install --prefix <folder> @tensorflow/tfjs@4.22.0 @tensorflow/tfjs-backend-wasm@4.22.0
 *   NODE_PATH=<folder>/node_modules node ml/scripts/verify_tfjs.cjs <tfjs model folder> <verify folder>
 */
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-backend-wasm");

// i-load ang model.json at weights mula sa disk (walang tfjs-node)
function fileHandler(dir) {
  return {
    load: async () => {
      const mj = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
      const weightSpecs = [];
      const buffers = [];
      for (const group of mj.weightsManifest) {
        weightSpecs.push(...group.weights);
        for (const p of group.paths) buffers.push(fs.readFileSync(path.join(dir, p)));
      }
      const all = Buffer.concat(buffers);
      return {
        modelTopology: mj.modelTopology,
        format: mj.format,
        generatedBy: mj.generatedBy,
        convertedBy: mj.convertedBy,
        signature: mj.signature,
        userDefinedMetadata: mj.userDefinedMetadata,
        modelInitializer: mj.modelInitializer,
        weightSpecs,
        weightData: all.buffer.slice(all.byteOffset, all.byteOffset + all.byteLength),
      };
    },
  };
}

async function main() {
  const [modelDir, verifyDir] = process.argv.slice(2);
  if (!modelDir || !verifyDir) throw new Error("Usage: node verify_tfjs.cjs <tfjs model folder> <verify folder>");
  await tf.setBackend("wasm").catch(() => tf.setBackend("cpu"));
  await tf.ready();

  const ref = JSON.parse(fs.readFileSync(path.join(verifyDir, "reference.json"), "utf8"));
  const pixels = fs.readFileSync(path.join(verifyDir, "inputs.u8"));
  const [n, h, w, c] = ref.shape;
  const per = h * w * c;
  const names = ref.class_names;
  const model = await tf.loadGraphModel(fileHandler(modelDir));

  let same = 0, rightJs = 0, rightKeras = 0, maxDiff = 0;
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    const img = Float32Array.from(pixels.subarray(i * per, (i + 1) * per));
    const probs = tf.tidy(() => model.predict(tf.tensor4d(img, [1, h, w, c])).dataSync());
    const kp = ref.keras_probs[i];
    const topJs = probs.indexOf(Math.max(...probs));
    const topKeras = kp.indexOf(Math.max(...kp));
    same += topJs === topKeras;
    rightJs += names[topJs] === ref.true[i];
    rightKeras += names[topKeras] === ref.true[i];
    for (let k = 0; k < kp.length; k++) maxDiff = Math.max(maxDiff, Math.abs(probs[k] - kp[k]));
    if (topJs !== topKeras) {
      console.log(`  iba ang hula: ${ref.files[i]}  TF.js=${names[topJs]} (${(probs[topJs] * 100).toFixed(1)}%)  ` +
                  `Keras=${names[topKeras]} (${(kp[topKeras] * 100).toFixed(1)}%)`);
    }
  }
  console.log(`Backend: ${tf.getBackend()} | ${((Date.now() - t0) / n).toFixed(0)} ms/image`);
  console.log(`Parehong hula ng TF.js at Keras: ${same}/${n}`);
  console.log(`Accuracy: TF.js ${(rightJs / n * 100).toFixed(2)}% | Keras ${(rightKeras / n * 100).toFixed(2)}%`);
  console.log(`Pinakamalaking pagkakaiba ng probability: ${(maxDiff * 100).toFixed(3)} percentage points`);
}

main().catch((e) => { console.error(e); process.exit(1); });
