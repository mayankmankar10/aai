import { Layers, Database, Cpu } from "lucide-react";
const items = [
  { icon: Layers, label: "Supported Classes", value: "43", sub: "GTSRB Full Set", color: "#a48ef0" },
  { icon: Cpu, label: "Architecture", value: "CapsNet", sub: "Capsule Network", color: "#6da3f7" },
  { icon: Database, label: "Dataset", value: "GTSRB", sub: "German Traffic Signs", color: "#5dd4aa" },
];
export default function StatsRow() {
  return (
    <div className="grid grid-cols-3 gap-4">
      {items.map(s => (
        <div key={s.label} className="surface flex items-center gap-4 px-5 py-4">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{background: s.color + "12", border: "1px solid " + s.color + "22"}}>
            <s.icon size={17} color={s.color} />
          </div>
          <div>
            <div className="text-[13px] font-bold tracking-tight" style={{color: s.color}}>{s.value}</div>
            <div className="text-[10px] font-medium" style={{color:"rgba(255,255,255,0.3)"}}>{s.sub}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
