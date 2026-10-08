import type { ReactNode } from "react";
import { CircleCheck, FlaskConical, Info, Leaf, ShieldCheck, Sprout, TriangleAlert } from "lucide-react";
import { PESTS, SAFETY, THRESHOLD, type PestInfo } from "@/lib/pests";

export type Prediction = { label: string; prob: number };

function percent(p: number) {
  return `${(p * 100).toFixed(1)}%`;
}

function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section
      className={
        "rounded-3xl bg-white p-5 shadow-sm ring-1 ring-emerald-900/10 dark:bg-emerald-950/40 dark:ring-emerald-400/15 " +
        className
      }
    >
      {children}
    </section>
  );
}

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="flex items-center gap-2 font-semibold text-emerald-900 dark:text-emerald-100">
        <span className="grid size-8 place-items-center rounded-xl bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300">
          {icon}
        </span>
        {title}
      </h3>
      {children}
    </div>
  );
}

function Bullets({ items }: { items: string[] }) {
  return (
    <ul className="flex flex-col gap-1.5 pl-1 text-sm leading-relaxed">
      {items.map((t) => (
        <li key={t} className="flex gap-2">
          <span className="mt-2 size-1.5 shrink-0 rounded-full bg-emerald-500" />
          <span>{t}</span>
        </li>
      ))}
    </ul>
  );
}

function PestPhoto({ info }: { info: PestInfo }) {
  const img = info.image;
  if (!img) return null;
  return (
    <figure className="overflow-hidden rounded-2xl bg-emerald-50 dark:bg-emerald-900/30">
      {/* eslint-disable-next-line @next/next/no-img-element -- maliit na static na larawan; hindi kailangan ng next/image */}
      <img src={img.src} alt={img.caption} className="aspect-[4/3] w-full object-cover" />
      <figcaption className="px-3 py-2 text-xs text-emerald-900/80 dark:text-emerald-100/80">
        {img.caption}. Larawan: {img.credit},{" "}
        <a href={img.licenseUrl} target="_blank" rel="noreferrer" className="underline">
          {img.license}
        </a>
        , via{" "}
        <a href={img.source} target="_blank" rel="noreferrer" className="underline">
          Wikimedia Commons
        </a>
      </figcaption>
    </figure>
  );
}

function Recommendations({ info }: { info: PestInfo }) {
  return (
    <Card className="flex flex-col gap-5">
      <div>
        <h2 className="text-lg font-bold text-emerald-950 dark:text-emerald-50">Rekomendasyon</h2>
        <p className="mt-2 flex gap-2 rounded-2xl bg-amber-50 p-3 text-xs leading-relaxed text-amber-900 dark:bg-amber-950/50 dark:text-amber-100">
          <Info className="size-4 shrink-0" aria-hidden />
          DRAFT: hindi pa ito nare-review ng agriculturist. Walang dosis dito; kung gagamit ng pesticide, sundin ang label
          ng produktong rehistrado sa FPA para sa talong.
        </p>
      </div>
      <Section icon={<Sprout className="size-4" aria-hidden />} title="Gawin muna (walang kemikal)">
        <Bullets items={info.ipm} />
      </Section>
      {info.lowToxic.length > 0 && (
        <Section icon={<Leaf className="size-4" aria-hidden />} title="Biological o mababang-toxicity">
          <Bullets items={info.lowToxic} />
        </Section>
      )}
      {info.chemical.length > 0 && (
        <Section icon={<FlaskConical className="size-4" aria-hidden />} title="Pesticide (kung malala na)">
          <ul className="flex flex-col gap-2">
            {info.chemical.map((c) => (
              <li key={c.ingredient} className="rounded-2xl border border-emerald-900/10 p-3 dark:border-emerald-400/15">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium">{c.ingredient}</span>
                  <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">
                    Group {c.group}
                  </span>
                </div>
                {c.note && <p className="mt-1 text-xs opacity-80">{c.note}</p>}
              </li>
            ))}
          </ul>
          <p className="text-xs opacity-80">
            Magpalit ng Group sa bawat pag-spray (hal. 28, tapos 6, tapos 5) para hindi magka-resistance.
          </p>
        </Section>
      )}
      {info.chemical.length > 0 && (
        <Section icon={<ShieldCheck className="size-4" aria-hidden />} title="Kaligtasan sa pag-spray">
          <Bullets items={SAFETY} />
        </Section>
      )}
    </Card>
  );
}

export default function PestResult({ predictions }: { predictions: Prediction[] }) {
  const top = predictions[0];
  const info = PESTS[top.label];
  const sure = top.prob >= THRESHOLD;

  return (
    <div className="flex flex-col gap-4">
      {sure && info ? (
        <Card className="flex flex-col gap-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <span
                className={
                  "inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold " +
                  (info.healthy
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200"
                    : "bg-amber-100 text-amber-800 dark:bg-amber-900/60 dark:text-amber-200")
                }
              >
                {info.healthy ? (
                  <CircleCheck className="size-3.5" aria-hidden />
                ) : (
                  <TriangleAlert className="size-3.5" aria-hidden />
                )}
                {info.healthy ? "Walang nakitang peste" : "Nakitang peste"}
              </span>
              <h2 className="mt-2 text-2xl font-bold text-emerald-950 dark:text-emerald-50">{info.name}</h2>
              {info.scientific && <p className="text-sm italic opacity-70">{info.scientific}</p>}
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-emerald-700 dark:text-emerald-300">{percent(top.prob)}</p>
              <p className="text-xs opacity-70">kumpiyansa</p>
            </div>
          </div>
          <p className="text-sm opacity-80">Bahagi ng halaman: {info.part}</p>
          <PestPhoto info={info} />
          {info.signs.length > 0 && (
            <div>
              <h3 className="mb-1 text-sm font-semibold">Mga palatandaan</h3>
              <Bullets items={info.signs} />
            </div>
          )}
        </Card>
      ) : (
        <Card className="flex flex-col gap-2 bg-amber-50 ring-amber-900/10 dark:bg-amber-950/40">
          <h2 className="flex items-center gap-2 text-lg font-bold text-amber-900 dark:text-amber-100">
            <TriangleAlert className="size-5" aria-hidden />
            Hindi sigurado, paki-retake ang photo
          </h2>
          <p className="text-sm">
            Pinakamalapit na hula: {PESTS[top.label]?.name ?? top.label} ({percent(top.prob)}), mas mababa sa{" "}
            {percent(THRESHOLD)}. Walang rekomendasyon hangga&apos;t hindi sigurado.
          </p>
          <Bullets
            items={[
              "Malapitan (close-up) at malinaw ang dahon o bunga",
              "Maliwanag, pero hindi tirik na araw o madilim",
              "Isang dahon o bunga lang sa litrato",
            ]}
          />
        </Card>
      )}

      <Card className="flex flex-col gap-3">
        <h3 className="text-sm font-semibold">Lahat ng hula</h3>
        {predictions.slice(0, 3).map((p) => (
          <div key={p.label} className="text-sm">
            <div className="mb-1 flex justify-between">
              <span>{PESTS[p.label]?.name ?? p.label}</span>
              <span className="tabular-nums opacity-80">{percent(p.prob)}</span>
            </div>
            <div className="h-2 rounded-full bg-emerald-900/10 dark:bg-emerald-100/10">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-emerald-500 to-green-600"
                style={{ width: `${p.prob * 100}%` }}
              />
            </div>
          </div>
        ))}
      </Card>

      {sure && info && !info.healthy && <Recommendations info={info} />}
      {sure && info?.healthy && (
        <Card className="flex flex-col gap-3">
          <Section icon={<Sprout className="size-4" aria-hidden />} title="Para manatiling malusog">
            <Bullets items={info.ipm} />
          </Section>
        </Card>
      )}
    </div>
  );
}
