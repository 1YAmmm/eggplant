import { Leaf, ShieldCheck, Smartphone } from "lucide-react";
import PestDetector from "@/components/pest-detector";

export default function Home() {
  return (
    <main className="flex-1 pb-10">
      <header className="relative overflow-hidden bg-gradient-to-br from-emerald-600 via-green-600 to-teal-700 px-4 pb-16 pt-10 text-white">
        <div className="absolute -right-16 -top-16 size-64 rounded-full bg-white/10 blur-2xl" aria-hidden />
        <div className="absolute -bottom-24 -left-10 size-72 rounded-full bg-lime-300/20 blur-3xl" aria-hidden />
        <div className="relative mx-auto max-w-xl">
          <div className="flex items-center gap-3">
            <span className="grid size-12 place-items-center rounded-2xl bg-white/15 ring-1 ring-white/25 backdrop-blur">
              <Leaf className="size-6" aria-hidden />
            </span>
            <div>
              <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Eggplant Pest Detector</h1>
              <p className="text-sm text-emerald-50/90">Pagkilala ng peste sa talong mula sa litrato</p>
            </div>
          </div>
          <div className="mt-5 flex flex-wrap gap-2 text-xs">
            <span className="inline-flex items-center gap-1 rounded-full bg-white/15 px-3 py-1 ring-1 ring-white/20">
              <Smartphone className="size-3.5" aria-hidden />
              Sa phone mismo tumatakbo
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-white/15 px-3 py-1 ring-1 ring-white/20">
              <ShieldCheck className="size-3.5" aria-hidden />
              IPM muna bago pesticide
            </span>
          </div>
        </div>
      </header>
      <div className="-mt-10 px-4">
        <PestDetector />
      </div>
    </main>
  );
}
