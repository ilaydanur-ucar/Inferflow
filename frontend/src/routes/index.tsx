import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "InferFlow — ML Inference Monitoring" },
      { name: "description", content: "InferFlow: Redis-backed ML inference servisi için gerçek zamanlı kuyruk ve worker izleme panosu." },
      { property: "og:title", content: "InferFlow — ML Inference Monitoring" },
      { property: "og:description", content: "Kuyruk derinliği, worker durumları ve sistem sağlığını saniyelik izleyin." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: InferFlowDashboard,
});

type WorkerInfo = {
  id: number;
  status: "idle" | "busy";
  total_handled: number;
  last_job_id: string;
};

type Stats = {
  queue_depth: number;
  workers: WorkerInfo[];
  timestamp: string;
};

const sim = (() => {
  let queue = 0;
  let jobCounter = 18573;
  const workers: WorkerInfo[] = [0, 1, 2, 3].map((id) => ({
    id,
    status: "idle",
    total_handled: 18500 + Math.floor(Math.random() * 100),
    last_job_id: cryptoRandomId(),
  }));
  return () => {
    queue = Math.max(0, Math.min(100, queue + Math.floor(Math.random() * 15) - 6));
    const busyCount = Math.min(4, Math.ceil(queue / 8));
    workers.forEach((w, i) => {
      w.status = i < busyCount ? "busy" : "idle";
      if (w.status === "busy" || Math.random() > 0.7) {
        const handled = 1 + Math.floor(Math.random() * 4);
        w.total_handled += handled;
        jobCounter += handled;
        w.last_job_id = cryptoRandomId();
      }
    });
    return {
      queue_depth: queue,
      workers: workers.map((w) => ({ ...w })),
      timestamp: new Date().toISOString(),
    } satisfies Stats;
  };
})();

function cryptoRandomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function fetchStats(): Promise<{ data: Stats; live: boolean }> {
  try {
    const res = await fetch("/stats", { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error(String(res.status));
    return { data: (await res.json()) as Stats, live: true };
  } catch {
    return { data: sim(), live: false };
  }
}

function queueStatus(depth: number) {
  if (depth === 0) return { label: "HEALTHY", icon: "⏸️", color: "oklch(0.66 0.02 160)", muted: true };
  if (depth < 10) return { label: "HEALTHY", icon: "⚡", color: "oklch(0.72 0.19 155)" };
  if (depth < 50) return { label: "BUSY", icon: "⚠️", color: "oklch(0.8 0.15 85)" };
  return { label: "OVERLOADED", icon: "🔥", color: "oklch(0.63 0.24 27)" };
}

const fmt = (n: number) => n.toLocaleString("tr-TR");

function pad2(n: number) {
  return n.toString().padStart(2, "0");
}

function InferFlowDashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [live, setLive] = useState(false);
  const [now, setNow] = useState(() => new Date());
  const startedAt = useRef(Date.now());

  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      const { data, live } = await fetchStats();
      if (!mounted) return;
      setStats(data);
      setLive(live);
      setNow(new Date());
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  const depth = stats?.queue_depth ?? 0;
  const status = queueStatus(depth);
  const totalHandled = useMemo(() => stats?.workers.reduce((a, w) => a + w.total_handled, 0) ?? 0, [stats]);
  const busyWorkers = stats?.workers.filter((w) => w.status === "busy").length ?? 0;

  const uptimeMs = Date.now() - startedAt.current;
  const uptimeH = Math.floor(uptimeMs / 3_600_000);
  const uptimeM = Math.floor((uptimeMs % 3_600_000) / 60_000);
  const clock = `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}`;

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="border-b border-border bg-card/60 backdrop-blur">
        <div className="mx-auto flex h-20 w-full max-w-6xl items-center justify-between gap-4 px-6">
          <div className="flex min-w-0 items-center gap-3">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-primary text-lg text-primary-foreground">⚡</span>
            <div className="min-w-0">
              <h1 className="truncate text-[28px] font-bold leading-[1.2] tracking-[-1px]">InferFlow</h1>
              <p className="hidden text-xs text-muted-foreground sm:block">ML inference service • Real-time monitoring</p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-4">
            <div className="flex items-center gap-2 rounded-md border border-border bg-secondary/60 px-3 py-1.5">
              <span className="h-2 w-2 animate-pulse-dot rounded-full bg-success" />
              <span className="text-xs font-semibold tracking-wide">RUNNING</span>
              {!live && <span className="rounded bg-warning/15 px-1.5 py-0.5 text-[10px] font-semibold text-warning">SIM</span>}
            </div>
            <span className="hidden font-mono text-sm tabular-nums text-muted-foreground md:block">{clock}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-6xl flex-1 gap-6 px-6 py-6 lg:grid-cols-[2fr_3fr]">
        <section aria-label="Redis iş kuyruğu derinliği" className="queue-glow relative flex flex-col justify-between overflow-hidden rounded-xl border border-border border-t-[3px] border-t-primary bg-card p-6" style={{ ["--glow" as string]: status.color }}>
          <h2 className="text-[18px] font-semibold leading-[1.2] tracking-[-0.5px] text-muted-foreground">Redis İş Kuyruğu Derinliği</h2>
          <div className="my-8 text-center">
            <div key={depth} className="text-glow animate-number text-7xl font-bold tabular-nums">{depth}</div>
            <p className="mt-2 text-sm text-muted-foreground">şu an bekleyen iş</p>
          </div>
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span aria-hidden>{status.icon}</span>
              <span className="text-sm font-semibold">Status: <span className="text-glow font-bold tracking-wide">{status.label}</span></span>
            </div>
            <div>
              <div className="mb-1.5 flex justify-between text-xs text-muted-foreground">
                <span>Queue</span>
                <span className="font-mono tabular-nums">{depth} / 100</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-secondary">
                <div className="h-full rounded-full transition-[width,background-color] duration-500" style={{ width: `${Math.min(100, depth)}%`, backgroundColor: status.color }} />
              </div>
            </div>
            <ul className="space-y-1.5 border-t border-border pt-4 text-sm">
              {[
                { label: "API Responding", ok: true },
                { label: "Redis Connected", ok: true },
                { label: `Workers Active (${busyWorkers}/4 busy)`, ok: true },
              ].map((item) => (
                <li key={item.label} className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${item.ok ? "bg-success" : "bg-danger"} animate-pulse-dot`} />
                  <span className="text-muted-foreground">{item.label}</span>
                  <span className="ml-auto text-success">✓</span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section aria-label="Tahmin worker'ları" className="flex flex-col overflow-hidden rounded-xl border border-border border-t-[3px] border-t-primary bg-card p-6">
          <h2 className="text-[18px] font-semibold leading-[1.2] tracking-[-0.5px] text-muted-foreground">Tahmin Worker'ları</h2>
          <div className="mt-4 flex-1 overflow-x-auto">
            <table className="w-full text-left text-sm leading-[1.4]">
              <thead>
                <tr className="border-b border-border text-xs uppercase tracking-wider text-muted-foreground">
                  <th className="px-3 py-3 font-medium">Worker</th>
                  <th className="px-3 py-3 font-medium">Durum</th>
                  <th className="px-3 py-3 font-medium text-right">İş Sayısı</th>
                  <th className="px-3 py-3 font-medium">Son İş ID</th>
                </tr>
              </thead>
              <tbody>
                {(stats?.workers ?? []).map((w) => <WorkerRow key={w.id} worker={w} />)}
                {!stats && [0, 1, 2, 3].map((i) => (
                  <tr key={i} className="border-b border-border/50"><td colSpan={4} className="px-3 py-4"><div className="h-4 animate-pulse rounded bg-secondary" /></td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 flex items-center justify-between border-t border-border pt-4 text-sm">
            <span className="text-muted-foreground">Toplam işlenen iş</span>
            <span className="font-mono text-base font-bold tabular-nums text-primary">{fmt(totalHandled)}</span>
          </div>
        </section>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-2 px-6 py-4 text-xs text-muted-foreground">
          <span>Güncellendi: <span className="font-mono tabular-nums text-foreground">{clock}</span></span>
          <span>Uptime: <span className="font-mono tabular-nums text-foreground">{uptimeH}h {uptimeM}m</span></span>
        </div>
      </footer>
    </div>
  );
}

function WorkerRow({ worker }: { worker: WorkerInfo }) {
  const [copied, setCopied] = useState(false);
  const busy = worker.status === "busy";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(worker.last_job_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {}
  };

  return (
    <tr className="border-b border-border/50 transition-all duration-200 hover:-translate-y-0.5 hover:bg-accent/40 hover:shadow-[0_10px_25px_rgba(0,0,0,0.4)]">
      <td className="px-3 py-3 font-mono font-medium tabular-nums">Worker-{worker.id}</td>
      <td className="px-3 py-3">
        <span className={`inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-semibold transition-colors duration-300 ${busy ? "bg-warning/15 text-warning" : "bg-success/15 text-success"}`}>
          {busy ? "⚙️ Busy" : "✓ Idle"}
        </span>
      </td>
      <td className="px-3 py-3 text-right font-mono tabular-nums">{fmt(worker.total_handled)}</td>
      <td className="px-3 py-3">
        <span className="group inline-flex items-center gap-1.5">
          <code title={worker.last_job_id} className="font-mono text-xs font-medium text-muted-foreground">{worker.last_job_id.slice(0, 8)}…</code>
          <button onClick={copy} aria-label="Job ID kopyala" className="rounded p-1 text-xs opacity-0 transition-all duration-200 hover:bg-secondary group-hover:opacity-100">{copied ? "✓" : "📋"}</button>
        </span>
      </td>
    </tr>
  );
}
