import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Search, Shield, Lightbulb, ChevronDown, ChevronUp, Check, X, Minus, Activity } from "lucide-react";

function ConfidenceBar({ pct, color }) {
  const ref = useRef();
  useEffect(() => {
    if (!ref.current) return;
    ref.current.style.width = "0%";
    const t = setTimeout(() => { if (ref.current) ref.current.style.width = pct + "%"; }, 80);
    return () => clearTimeout(t);
  }, [pct]);
  const grad = "linear-gradient(90deg, " + color + "66, " + color + ")";
  return (
    <div className="conf-track">
      <div ref={ref} className="conf-fill" style={{ background: grad }} />
    </div>
  );
}

function Trace({ trace }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid rgba(255,255,255,0.04)" }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-[12px] font-semibold transition-all hover:bg-white/[0.015]"
        style={{ color: "rgba(255,255,255,0.4)" }}
      >
        <div className="flex items-center gap-2"><Activity size={13} />Agent Reasoning Trace</div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-mono" style={{ color: "rgba(255,255,255,0.2)" }}>{trace.length} steps</span>
          {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </div>
      </button>
      {open && (
        <div className="px-4 pb-4 pt-1 space-y-1" style={{ background: "rgba(255,255,255,0.008)" }}>
          {trace.map((line, i) => {
            const ok = line.includes("\u2713");
            const fail = line.includes("\u2717");
            const okBg = "rgba(60,200,130,0.1)";
            const failBg = "rgba(240,60,100,0.1)";
            const neutralBg = "rgba(255,255,255,0.03)";
            return (
              <div key={i} className="flex items-start gap-2.5 py-1">
                <div className="mt-[3px] w-[18px] h-[18px] rounded-md flex items-center justify-center flex-shrink-0"
                  style={{ background: ok ? okBg : fail ? failBg : neutralBg }}>
                  {ok ? <Check size={10} style={{ color: "#3dc882" }} /> : fail ? <X size={10} style={{ color: "#f03c64" }} /> : <Minus size={10} style={{ color: "rgba(255,255,255,0.2)" }} />}
                </div>
                <div className="text-[10.5px] font-mono leading-relaxed"
                  style={{ color: ok ? "rgba(255,255,255,0.55)" : fail ? "rgba(240,60,100,0.7)" : "rgba(255,255,255,0.3)" }}>
                  {line}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function catStyle(cat) {
  const c = cat || "";
  if (c.includes("Warning")) return { color: "#e8a838", bg: "rgba(232,168,56,0.07)", border: "rgba(232,168,56,0.15)" };
  if (c.includes("Mandatory")) return { color: "#6da3f7", bg: "rgba(109,163,247,0.07)", border: "rgba(109,163,247,0.15)" };
  if (c.includes("Prohibit") || c.includes("Stop")) return { color: "#f06080", bg: "rgba(240,96,128,0.07)", border: "rgba(240,96,128,0.15)" };
  if (c.includes("End")) return { color: "#5dd4aa", bg: "rgba(93,212,170,0.07)", border: "rgba(93,212,170,0.15)" };
  return { color: "#a48ef0", bg: "rgba(164,142,240,0.07)", border: "rgba(164,142,240,0.15)" };
}

function confColor(c) {
  if (c >= 0.8) return "#3dc882";
  if (c >= 0.5) return "#6da3f7";
  return "#f06080";
}

export default function ResultPanel({ result, loading, error }) {
  if (loading) return (
    <div className="surface-elevated p-6 flex flex-col items-center justify-center gap-5 min-h-[380px]">
      <div className="spinner" style={{ width: 28, height: 28, borderWidth: "2.5px" }} />
      <div className="text-center">
        <div className="text-[13px] font-semibold" style={{ color: "rgba(255,255,255,0.45)" }}>Running classification pipeline</div>
        <div className="text-[11px] mt-1" style={{ color: "rgba(255,255,255,0.2)" }}>Preprocessing · Inference · Knowledge Retrieval</div>
      </div>
    </div>
  );

  if (error) return (
    <div className="surface-elevated p-6 min-h-[380px]">
      <div className="text-label flex items-center gap-1.5 mb-4"><AlertTriangle size={10} />Error</div>
      <div className="rounded-xl p-4 text-[13px] leading-relaxed"
        style={{ background: "rgba(240,60,100,0.05)", border: "1px solid rgba(240,60,100,0.1)", color: "rgba(255,255,255,0.6)" }}>{error}</div>
    </div>
  );

  if (!result) return (
    <div className="surface-elevated p-6 flex flex-col items-center justify-center gap-4 min-h-[380px]">
      <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
        style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)" }}>
        <Search size={22} style={{ color: "rgba(255,255,255,0.12)" }} />
      </div>
      <div className="text-center">
        <div className="text-[13px] font-medium" style={{ color: "rgba(255,255,255,0.3)" }}>Awaiting input</div>
        <div className="text-[11px] mt-1" style={{ color: "rgba(255,255,255,0.15)" }}>Upload a traffic sign image to begin analysis</div>
      </div>
    </div>
  );

  if (!result.success) return (
    <div className="surface-elevated p-6 min-h-[380px]">
      <div className="text-label flex items-center gap-1.5 mb-4"><AlertTriangle size={10} />Classification Failed</div>
      <div className="text-[13px]" style={{ color: "rgba(255,255,255,0.5)" }}>{result.error_message}</div>
      {result.reasoning_trace?.length > 0 && <div className="mt-5"><Trace trace={result.reasoning_trace} /></div>}
    </div>
  );

  const pct = Math.round(result.confidence * 100);
  const cc = confColor(result.confidence);
  const cs = catStyle(result.category);
  const confLevel = pct >= 80 ? "High" : pct >= 50 ? "Moderate" : "Low";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      className="surface-elevated p-6 flex flex-col gap-5"
    >
      <div className="text-label flex items-center gap-1.5"><Activity size={10} />Analysis Result</div>

      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[22px] font-bold tracking-tight leading-tight accent-text">{result.label}</div>
          <div className="text-[11px] font-mono mt-1.5" style={{ color: "rgba(255,255,255,0.25)" }}>
            class_index: {result.class_index}
          </div>
        </div>
        <div className="text-right flex-shrink-0">
          <div className="text-[28px] font-black font-mono tracking-tighter" style={{ color: cc }}>
            {pct}<span className="text-[14px] font-semibold">%</span>
          </div>
        </div>
      </div>

      <div>
        <ConfidenceBar pct={pct} color={cc} />
        <div className="flex items-center gap-2 mt-2.5">
          <span className="tag" style={{ background: cs.bg, color: cs.color, border: "1px solid " + cs.border }}>
            {result.category}
          </span>
          <span className="tag" style={{ background: cc + "14", color: cc, border: "1px solid " + cc + "25" }}>
            {confLevel} Confidence
          </span>
        </div>
      </div>

      <div className="grid gap-3 mt-1">
        <div className="rounded-xl p-4" style={{ background: "rgba(164,142,240,0.03)", borderLeft: "2px solid rgba(164,142,240,0.25)" }}>
          <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest mb-2"
            style={{ color: "rgba(164,142,240,0.6)" }}>
            <Lightbulb size={11} />Explanation
          </div>
          <div className="text-[12.5px] leading-[1.7]" style={{ color: "rgba(255,255,255,0.6)" }}>{result.explanation}</div>
        </div>

        <div className="rounded-xl p-4" style={{ background: "rgba(93,212,170,0.025)", borderLeft: "2px solid rgba(93,212,170,0.25)" }}>
          <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest mb-2"
            style={{ color: "rgba(93,212,170,0.6)" }}>
            <Shield size={11} />Safety Advice
          </div>
          <div className="text-[12.5px] leading-[1.7]" style={{ color: "rgba(255,255,255,0.6)" }}>{result.safety_tip}</div>
        </div>
      </div>

      {result.reasoning_trace?.length > 0 && <Trace trace={result.reasoning_trace} />}
    </motion.div>
  );
}
