import { useState } from "react";
import { Search, ChevronDown, ChevronUp, Settings, CircleDot } from "lucide-react";
const CATS = {
  "Speed Limits":[0,1,2,3,4,5,6,7,8],
  "Overtaking":[9,10],
  "Right-of-Way":[11,12],
  "Stop & Yield":[13,14],
  "Prohibitory":[15,16,17],
  "Warning Signs":[18,19,20,21,22,23,24,25,26,27,28,29,30,31],
  "Mandatory":[33,34,35,36,37,38,39,40],
  "End of Restriction":[32,41,42],
};
export default function Sidebar({ open, labels, backendOk, modelPath, onModelPath }) {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState(null);
  if (!open) return null;
  const filtered = Object.entries(labels||{}).filter(([k,v])=>
    v.toLowerCase().includes(search.toLowerCase())||k.includes(search));
  const statusColor = backendOk === null ? "pending" : backendOk ? "online" : "offline";
  const statusText = backendOk === null ? "Connecting..." : backendOk ? "Server online" : "Server unreachable";
  return (
    <aside className="fixed left-0 top-0 bottom-0 w-[280px] z-40 flex flex-col"
      style={{background:"rgba(6,6,14,0.97)", borderRight:"1px solid rgba(255,255,255,0.04)"}}>
      <div className="px-5 pt-6 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center accent-gradient">
            <CircleDot size={18} color="#fff" strokeWidth={2.5}/>
          </div>
          <div>
            <div className="text-[13px] font-bold tracking-tight" style={{color:"rgba(255,255,255,0.88)"}}>CapsNet Agent</div>
            <div className="text-[10px] font-medium" style={{color:"rgba(255,255,255,0.3)"}}>v1.0 - Traffic Intelligence</div>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-4 px-3 py-2 rounded-lg" style={{background:"rgba(255,255,255,0.025)"}}>
          <div className={"status-dot " + statusColor}/>
          <span className="text-[11px] font-medium" style={{color:"rgba(255,255,255,0.4)"}}>{statusText}</span>
        </div>
      </div>
      <div className="divider"/>
      <div className="px-5 py-4">
        <div className="text-label mb-2.5 flex items-center gap-1.5"><Settings size={10}/>Model Config</div>
        <input value={modelPath} onChange={e=>onModelPath(e.target.value)} placeholder="Default model path"
          className="w-full text-[11px] px-3 py-2 rounded-lg outline-none font-mono"
          style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.05)",color:"rgba(255,255,255,0.6)"}}/>
      </div>
      <div className="divider"/>
      <div className="flex-1 overflow-y-auto px-5 py-4">
        <div className="text-label mb-2.5">Class Browser</div>
        <div className="relative mb-3">
          <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2" style={{color:"rgba(255,255,255,0.2)"}}/>
          <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Filter classes..."
            className="w-full text-[11px] pl-8 pr-3 py-2 rounded-lg outline-none"
            style={{background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.05)",color:"rgba(255,255,255,0.6)"}}/>
        </div>
        {search?(
          <div className="space-y-0.5">
            {filtered.map(([k,v])=>(
              <div key={k} className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[11px]" style={{color:"rgba(255,255,255,0.55)"}}>
                <span className="font-mono font-semibold text-[10px] w-6 text-right" style={{color:"#a48ef0"}}>{k}</span>
                <span>{v}</span>
              </div>
            ))}
          </div>
        ):(
          <div className="space-y-0.5">
            {Object.entries(CATS).map(([cat,ids])=>(
              <div key={cat}>
                <button onClick={()=>setExpanded(expanded===cat?null:cat)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-[11px] font-semibold transition-all hover:bg-white/[0.02]"
                  style={{color:"rgba(255,255,255,0.45)"}}>
                  <span>{cat}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[9px] font-mono" style={{color:"rgba(255,255,255,0.2)"}}>{ids.length}</span>
                    {expanded===cat?<ChevronUp size={12}/>:<ChevronDown size={12}/>}
                  </div>
                </button>
                {expanded===cat&&(
                  <div className="ml-1 space-y-px mt-0.5 mb-2 pl-2" style={{borderLeft:"1px solid rgba(255,255,255,0.04)"}}>
                    {ids.filter(id=>(labels||{})[id]).map(id=>(
                      <div key={id} className="flex items-center gap-2.5 px-3 py-1.5 rounded-md text-[10.5px]" style={{color:"rgba(255,255,255,0.45)"}}>
                        <span className="font-mono font-semibold w-5 text-right text-[10px]" style={{color:"rgba(160,130,240,0.7)"}}>{id}</span>
                        <span>{labels[id]}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
