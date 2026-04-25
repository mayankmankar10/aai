const API_BASE = "/api";
export async function classifyImage(file, modelPath="") {
  const fd = new FormData();
  fd.append("image", file);
  if (modelPath) fd.append("model_path", modelPath);
  const res = await fetch(`${API_BASE}/classify`, { method:"POST", body:fd });
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.error||`Server error (${res.status})`); }
  return res.json();
}
export async function getLabels() {
  const res = await fetch(`${API_BASE}/labels`);
  if (!res.ok) throw new Error("Failed to fetch labels");
  return (await res.json()).labels;
}
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return false;
    return (await res.json()).status === "ok";
  } catch { return false; }
}
