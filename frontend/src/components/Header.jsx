import { Menu, PanelLeftClose, Zap } from "lucide-react";
export default function Header({ sidebarOpen, onToggleSidebar, backendOk }) {
  const statusColor = backendOk === null ? "pending" : backendOk ? "online" : "offline";
  const statusText = backendOk === null ? "Connecting" : backendOk ? "API Connected" : "API Offline";
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between px-8 h-[60px]"
      style={{background:"rgba(6,6,14,0.6)", backdropFilter:"blur(20px) saturate(1.4)", borderBottom:"1px solid rgba(255,255,255,0.04)"}}>
      <div className="flex items-center gap-4">
        <button onClick={onToggleSidebar}
          className="w-8 h-8 flex items-center justify-center rounded-lg transition-all hover:bg-white/5"
          style={{color:"rgba(255,255,255,0.4)"}}>
          {sidebarOpen ? <PanelLeftClose size={16}/> : <Menu size={16}/>}
        </button>
        <div className="h-5 w-px bg-white/5"/>
        <div>
          <h1 className="text-[15px] font-semibold tracking-tight" style={{color:"rgba(255,255,255,0.88)"}}>
            Traffic Sign Classifier
          </h1>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-[11px] font-medium px-3 py-1.5 rounded-lg"
          style={{background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.05)", color:"rgba(255,255,255,0.4)"}}>
          <div className={"status-dot " + statusColor}/>
          {statusText}
        </div>
        <div className="flex items-center gap-1.5 text-[11px] font-semibold px-3 py-1.5 rounded-lg"
          style={{background:"rgba(100,60,200,0.08)", border:"1px solid rgba(100,60,200,0.15)", color:"#a48ef0"}}>
          <Zap size={12}/>
          CapsNet
        </div>
      </div>
    </header>
  );
}
