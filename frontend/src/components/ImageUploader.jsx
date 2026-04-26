import { useCallback, useRef, useState } from "react";
import { Upload, ImageIcon, RefreshCw } from "lucide-react";
export default function ImageUploader({ onFile, loading, preview, fileName }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);
  const handleDrop = useCallback(e=>{
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer?.files[0];
    if(f) onFile(f);
  },[onFile]);
  const dropCls = "dropzone flex-1 min-h-[220px] flex flex-col items-center justify-center relative " + (dragging ? "active" : "");
  return (
    <div className="surface-elevated p-5 flex flex-col gap-4 h-full">
      <div className="text-label flex items-center gap-1.5"><Upload size={10}/>Input</div>
      <div onDragOver={e=>{e.preventDefault();setDragging(true)}} onDragLeave={()=>setDragging(false)} onDrop={handleDrop} onClick={()=>inputRef.current.click()} className={dropCls}>
        <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/bmp,image/webp" className="hidden" onChange={e=>e.target.files[0]&&onFile(e.target.files[0])}/>
        {preview?(
          <div className="flex flex-col items-center gap-3 p-4 w-full">
            <img src={preview} alt="Preview" className="max-h-[180px] max-w-full rounded-lg object-contain" style={{filter:"drop-shadow(0 8px 24px rgba(0,0,0,0.5))"}}/>
            <div className="text-[11px] font-medium font-mono px-2.5 py-1 rounded-md" style={{background:"rgba(255,255,255,0.03)",color:"rgba(255,255,255,0.35)"}}>{fileName}</div>
          </div>
        ):(
          <div className="flex flex-col items-center gap-3 p-6">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center" style={{background:"rgba(255,255,255,0.025)", border:"1px solid rgba(255,255,255,0.04)"}}>
              <ImageIcon size={20} style={{color:"rgba(255,255,255,0.2)"}}/>
            </div>
            <div>
              <div className="text-[13px] font-medium text-center" style={{color:"rgba(255,255,255,0.4)"}}>Drop an image here</div>
              <div className="text-[11px] text-center mt-1" style={{color:"rgba(255,255,255,0.18)"}}>or click to browse - PNG, JPG, BMP, WebP</div>
            </div>
          </div>
        )}
      </div>
      <button onClick={()=>inputRef.current.click()} disabled={loading} className="btn-action w-full">
        {loading?(<><div className="spinner"/>Classifying...</>):(
          preview?(<><RefreshCw size={14}/>Replace Image</>):(<><Upload size={14}/>Select Image</>)
        )}
      </button>
    </div>
  );
}
