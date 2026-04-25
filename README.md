# CapsNet Traffic Sign Agent

An agentic AI application that integrates a pre-trained **Capsule Network (CapsNet)** for GTSRB traffic sign classification into a modular, production-oriented system.

---

## Project Structure

```
c:\AAI\
├── model/
│   ├── __init__.py
│   └── capsnet_model.py        ← Model Layer
├── agent/
│   ├── __init__.py
│   └── agent_pipeline.py       ← Agent Layer
├── backend/
│   └── api.py                  ← Flask REST API
├── frontend/
│   └── src/                    ← React Web App
├── traffic_sign_capsnet.h5     ← Pre-trained CapsNet model (place here)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your model

Copy your trained CapsNet `.h5` file to the project root:

```
c:\AAI\traffic_sign_capsnet.h5
```

Or specify a custom path via the sidebar in the UI.

### 3. Run the Backend (Flask API)

```bash
cd c:\AAI
python backend/api.py
```
The API will run on `http://localhost:5000`.

### 4. Run the Frontend (React App)

Open a new terminal:
```bash
cd c:\AAI\frontend
npm install
npm run dev
```
Access the premium React UI at `http://localhost:5173`.

---

## Architecture

### Layer 1 — Model (`model/capsnet_model.py`)

| Function | Description |
|---|---|
| `load_model(path)` | Lazy-loads the CapsNet `.h5` file (singleton cache) |
| `preprocess_image(path)` | BGR→RGB, resize to 32×32, normalize to [0,1], add batch dim |
| `predict_image(path)` | End-to-end inference → `(class_index, confidence, label)` |
| `GTSRB_LABELS` | Dict mapping all 43 class indices to human-readable names |

### Layer 2 — Agent (`agent/agent_pipeline.py`)

Five-step reasoning pipeline:

1. **Input Validation** — sanity-check the image path
2. **Perception** — invoke CapsNet model
3. **Knowledge Retrieval** — look up sign explanation & category
4. **Confidence Interpretation** — assess model certainty (High/Moderate/Low)
5. **Response Synthesis** — return `AgentResponse` dataclass

Returns a structured `AgentResponse` with:
- `label`, `class_index`, `confidence`
- `category`, `explanation`, `safety_tip`
- `confidence_assessment`
- `reasoning_trace` (step-by-step log)

### Layer 3 — Interface (`frontend/` & `backend/`)

- Premium dark glassmorphism React UI (Vite + TailwindCSS)
- `framer-motion` micro-animations for stunning visual feedback
- Flask backend exposing the agent pipeline via REST APIs
- Real-time API health checks and dynamic confidence visualization

---

## GTSRB Classes (43)

Classes 0–42 cover all German Traffic Sign Recognition Benchmark categories including speed limits, prohibitory signs, mandatory directions, warning signs, and right-of-way rules.

---

## Error Handling

| Error Type | Handled By |
|---|---|
| Model file not found | `load_model` → `FileNotFoundError` |
| Invalid/corrupt image | `preprocess_image` → `ValueError` |
| TF/Keras load failure | `load_model` → `RuntimeError` |
| All agent errors | `agent_pipeline` → `AgentResponse.error_message` |

---

## Notes

- **No retraining** — inference only
- Model output supports both single-tensor (softmax) and multi-output (CapsNet capsule lengths) formats
- The singleton model cache avoids repeated `.h5` file loads across API requests
