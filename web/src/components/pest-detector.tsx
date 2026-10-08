"use client";

import { useEffect, useRef, useState, type DragEvent } from "react";
import type * as TF from "@tensorflow/tfjs";
import type Pica from "pica";
import { Camera, ImagePlus, ScanSearch } from "lucide-react";
import PestResult, { type Prediction } from "@/components/pest-result";
import { IMG_SIZE } from "@/lib/pests";

type Status = "loading" | "ready" | "predicting" | "error";

const MODEL_URL = "/model/model.json";
const CLASS_NAMES_URL = "/model/class_names.json";

function canvasOf(w: number, h: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  return canvas;
}

// Parehong resize ng training (balance_dataset.py: PIL LANCZOS, stretch sa 224x224, hindi crop).
// Lanczos3 ng pica ang gamit, hindi ang drawImage ng canvas: iba ang hula sa malalaking larawan kapag canvas.
async function resizeToInput(pica: Pica.Pica, img: HTMLImageElement): Promise<HTMLCanvasElement> {
  const src = canvasOf(img.naturalWidth, img.naturalHeight);
  const ctx = src.getContext("2d");
  if (!ctx) throw new Error("Walang canvas 2D sa browser na ito");
  ctx.drawImage(img, 0, 0);
  return pica.resize(src, canvasOf(IMG_SIZE, IMG_SIZE), { filter: "lanczos3" });
}

export default function PestDetector() {
  const tfRef = useRef<typeof TF | null>(null);
  const modelRef = useRef<TF.GraphModel | null>(null);
  const picaRef = useRef<Pica.Pica | null>(null);
  const [classNames, setClassNames] = useState<string[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [progress, setProgress] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [backend, setBackend] = useState("");
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // dito lang ini-import (browser lang, at para magaan ang unang load ng page)
        const [tf, { default: PicaCtor }] = await Promise.all([import("@tensorflow/tfjs"), import("pica")]);
        picaRef.current = new PicaCtor({ features: ["js", "wasm"] });
        await tf.ready();
        const [model, names] = await Promise.all([
          tf.loadGraphModel(MODEL_URL, { onProgress: (p) => !cancelled && setProgress(p) }),
          fetch(CLASS_NAMES_URL).then((r) => r.json() as Promise<string[]>),
        ]);
        // unang predict: para ma-compile na ang WebGL shaders bago ang totoong larawan
        const warmup = model.predict(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3])) as TF.Tensor;
        await warmup.data();
        warmup.dispose();
        if (cancelled) {
          model.dispose();
          return;
        }
        tfRef.current = tf;
        modelRef.current = model;
        setClassNames(names);
        setBackend(tf.getBackend());
        setStatus("ready");
      } catch (e) {
        if (!cancelled) {
          setLoadError(String(e));
          setStatus("error");
        }
      }
    })();
    return () => {
      cancelled = true;
      modelRef.current?.dispose();
      modelRef.current = null;
    };
  }, []);

  async function onFile(file: File | undefined) {
    const tf = tfRef.current;
    const model = modelRef.current;
    const pica = picaRef.current;
    if (!file || !tf || !model || !pica) return;
    setStatus("predicting");
    setPredictions(null);
    setImageError(null);
    if (preview) URL.revokeObjectURL(preview);
    const url = URL.createObjectURL(file);
    setPreview(url);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      const canvas = await resizeToInput(pica, img);
      const output = tf.tidy(
        () => model.predict(tf.browser.fromPixels(canvas).toFloat().expandDims(0)) as TF.Tensor,
      );
      const probs = Array.from(await output.data());
      output.dispose();
      setPredictions(probs.map((prob, i) => ({ label: classNames[i], prob })).sort((a, b) => b.prob - a.prob));
    } catch (e) {
      setImageError(`Hindi mabasa ang larawan (${String(e)}). Subukan ang ibang file.`);
    } finally {
      setStatus("ready");
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (status === "ready") onFile(e.dataTransfer.files?.[0]);
  }

  const disabled = status !== "ready";
  const button =
    "flex items-center justify-center gap-2 rounded-2xl px-4 py-3.5 text-sm font-semibold transition " +
    "focus-within:ring-2 focus-within:ring-emerald-400 " +
    (disabled ? "cursor-not-allowed opacity-50 " : "cursor-pointer active:scale-[0.98] ");

  return (
    <div className="mx-auto flex w-full max-w-xl flex-col gap-4">
      <section className="rounded-3xl bg-white p-4 shadow-lg shadow-emerald-900/5 ring-1 ring-emerald-900/10 dark:bg-emerald-950 dark:ring-emerald-400/15">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={
            "relative grid min-h-56 place-items-center overflow-hidden rounded-2xl border-2 border-dashed transition " +
            (dragging
              ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/40"
              : "border-emerald-200 bg-emerald-50/50 dark:border-emerald-800 dark:bg-emerald-900/20")
          }
        >
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element -- blob URL ng piniling larawan, hindi kailangan ng next/image
            <img src={preview} alt="Larawang sinuri" className="max-h-80 w-full object-contain" />
          ) : (
            <div className="flex flex-col items-center gap-2 p-6 text-center text-emerald-800 dark:text-emerald-200">
              <span className="grid size-14 place-items-center rounded-2xl bg-emerald-100 dark:bg-emerald-900">
                <ScanSearch className="size-7" aria-hidden />
              </span>
              <p className="font-semibold">Litrato ng dahon o bunga ng talong</p>
              <p className="text-xs opacity-75">Malapitan, malinaw, at maliwanag. Puwede ring i-drag dito ang file.</p>
            </div>
          )}
          {status === "predicting" && (
            <div className="absolute inset-0 grid place-items-center bg-emerald-950/40 text-sm font-semibold text-white backdrop-blur-sm">
              Sinusuri ang larawan...
            </div>
          )}
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <label className={button + "bg-emerald-600 text-white shadow-md shadow-emerald-600/25 hover:bg-emerald-700"}>
            <Camera className="size-5" aria-hidden />
            Kumuha ng litrato
            <input
              type="file"
              accept="image/*"
              capture="environment"
              className="sr-only"
              disabled={disabled}
              onChange={(e) => {
                onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
          <label
            className={
              button +
              "bg-emerald-50 text-emerald-800 ring-1 ring-emerald-600/30 hover:bg-emerald-100 " +
              "dark:bg-emerald-900/40 dark:text-emerald-100 dark:hover:bg-emerald-900/70"
            }
          >
            <ImagePlus className="size-5" aria-hidden />
            Pumili ng larawan
            <input
              type="file"
              accept="image/*"
              className="sr-only"
              disabled={disabled}
              onChange={(e) => {
                onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
          </label>
        </div>

        {status === "loading" && (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs text-emerald-800 dark:text-emerald-200">
              <span>Inihahanda ang model...</span>
              <span className="tabular-nums">{Math.round(progress * 100)}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-emerald-100 dark:bg-emerald-900">
              <div
                className="h-1.5 rounded-full bg-emerald-500 transition-all"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>
        )}
        {status === "error" && (
          <p className="mt-4 rounded-2xl bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-200">
            Hindi ma-load ang model. I-refresh ang page o subukan sa ibang browser. ({loadError})
          </p>
        )}
        {imageError && <p className="mt-3 text-sm text-red-700 dark:text-red-300">{imageError}</p>}
      </section>

      {predictions && <PestResult predictions={predictions} />}

      <p className="px-2 text-center text-xs leading-relaxed text-emerald-900/70 dark:text-emerald-100/60">
        Gabay lamang ito. Kumonsulta sa lokal na agriculturist bago gumamit ng anumang pesticide. Kinikilala lang ang
        aphids, flea beetle, fruit and shoot borer, thrips, at malusog na bunga o dahon; hindi sakop ang ibang peste o
        sakit.
        {backend && ` (TF.js: ${backend})`}
      </p>
    </div>
  );
}
