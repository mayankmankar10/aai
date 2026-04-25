import { useState, useEffect, useCallback } from "react";
import { classifyImage, getLabels, checkHealth } from "./api/classify";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import ImageUploader from "./components/ImageUploader";
import ResultPanel from "./components/ResultPanel";
import StatsRow from "./components/StatsRow";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [labels, setLabels] = useState({});
  const [backendOk, setBackendOk] = useState(null);
  const [modelPath, setModelPath] = useState("");
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    checkHealth().then(setBackendOk);
    getLabels().then(setLabels).catch(() => {});
    const id = setInterval(() => checkHealth().then(setBackendOk), 15000);
    return () => clearInterval(id);
  }, []);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setFileName(file.name);
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const data = await classifyImage(file, modelPath);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [modelPath]);

  const mainCls = "flex-1 flex flex-col transition-all duration-500 " + (sidebarOpen ? "ml-[280px]" : "ml-0");

  return (
    <>
      <div className="ambient-bg" />
      <div className="noise-overlay" />
      <div className="relative z-10 flex min-h-screen">
        <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(o => !o)}
          labels={labels} backendOk={backendOk} modelPath={modelPath} onModelPath={setModelPath} />
        <div className={mainCls}>
          <Header sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(o => !o)} backendOk={backendOk} />
          <main className="flex-1 px-8 py-6 max-w-[1400px] mx-auto w-full">
            <StatsRow />
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 mt-6">
              <div className="lg:col-span-2">
                <ImageUploader onFile={handleFile} loading={loading} preview={preview} fileName={fileName} />
              </div>
              <div className="lg:col-span-3">
                <ResultPanel result={result} loading={loading} error={error} />
              </div>
            </div>
          </main>
          <footer className="px-8 py-5 flex items-center justify-between text-[11px] tracking-wide"
            style={{color:"rgba(255,255,255,0.15)", borderTop:"1px solid rgba(255,255,255,0.035)"}}>
            <span>CapsNet Traffic Sign Agent</span>
            <span>GTSRB · 43 Classes · Inference Only</span>
          </footer>
        </div>
      </div>
    </>
  );
}
