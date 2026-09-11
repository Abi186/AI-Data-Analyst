# 📊 AI Data Analyst — Local LLM Edition (₹0 Cost)

Upload CSV/Excel → Pandas processes it → **Qwen running locally via Ollama** generates
statistics, charts, insights, and answers your questions. No API key. No internet
required after setup. Zero cost, forever.

```
CSV / Excel
    ↓
Data Processing (Pandas)
    ↓
AI Analyst (Qwen + Ollama)
    ↓
Statistics | Insights | Q&A
    ↓
Charts | Findings | AI Chat
```

---

## 🧩 Project Structure

```
ai-data-analyst/
├── app.py                  # Main Streamlit app (UI + logic)
├── data_processor.py       # Pandas: load file, stats, correlations
├── ollama_client.py        # Talks to local Ollama server (Qwen)
├── requirements.txt        # Python dependencies
├── sample_data/
│   └── shrimp_farm_sample.csv   # Ready-made test file
└── README.md
```

---

## ✅ STEP 1 — Install Ollama (one-time, ~5 min)

Ollama runs the LLM **locally on your machine**. This is what makes it free.

**Windows / Mac:**
Download and install from: https://ollama.com/download

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify it installed:
```bash
ollama --version
```

---

## ✅ STEP 2 — Pull the Qwen model (one-time, ~2-5 min depending on internet)

Pick ONE based on your RAM (you can change this later in the app's sidebar too):

```bash
# Best for laptops with 8GB RAM (recommended to start)
ollama pull qwen2.5:3b

# If you have 16GB+ RAM and want better quality
ollama pull qwen2.5:7b

# If you have <8GB RAM / low-end machine
ollama pull qwen2:1.5b
```

The default model in the code is `qwen2.5:3b`. If you pull a different one,
just type its name into the "Ollama model" box in the app sidebar.

---

## ✅ STEP 3 — Start the Ollama server

In a terminal (keep this running in the background):
```bash
ollama serve
```
> If it says the port is already in use, Ollama is probably already running as a
> background service — that's fine, just move to Step 4.

Quick test it's alive (optional, in a new terminal):
```bash
curl http://localhost:11434
```
You should see: `Ollama is running`

---

## ✅ STEP 4 — Set up the Python project

Unzip the project, then:

```bash
cd ai-data-analyst

# (Recommended) create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ✅ STEP 5 — Run the app

```bash
streamlit run app.py
```

This opens the app automatically at: **http://localhost:8501**

---

## ✅ STEP 6 — Use it

1. Upload a file — or use the included `sample_data/shrimp_farm_sample.csv` to test immediately
2. **Statistics tab** — shape, dtypes, missing values, numeric summary
3. **Charts tab** — histograms, correlation heatmap, category bar charts
4. **AI Insights tab** — click "Generate Insights" → Qwen analyzes the data summary and gives findings + recommendations
5. **Q&A Chat tab** — ask free-form questions about your data; Qwen answers grounded in the dataset summary

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| Sidebar shows "Ollama not detected" | Run `ollama serve` in a terminal and keep it open |
| Response is very slow | Use a smaller model: `ollama pull qwen2:1.5b`, then type `qwen2:1.5b` in the sidebar |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again inside your activated venv |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Model gives vague answers | Try `qwen2.5:7b` if your RAM allows — better reasoning quality |

---

## 💡 Why this is ₹0 cost

- **Ollama** runs the model as a local process on your own CPU/GPU — no cloud call, no API key, no billing.
- **Qwen 2.5** (Alibaba's open-weight model) is free to download and use.
- Everything (Streamlit, Pandas, Ollama) runs on your machine. After the initial
  `ollama pull` download, it works fully offline.

---

## 🚀 Next steps / ideas to extend

- Swap `qwen2.5:3b` for `llama3.2`, `mistral`, or `phi3` — just `ollama pull <model>` and change the name in the sidebar
- Add PDF export of insights (use the `pdf` skill / `reportlab`)
- Add anomaly detection (z-score) before sending summary to the LLM
- Cache repeated questions to avoid re-querying Ollama
