# CBRN-Sentinel: AgentBeats Green Agent 🛡️☣️

![Protocol](https://img.shields.io/badge/Protocol-A2A_v1-blue)
![Focus](https://img.shields.io/badge/Focus-CBRN_Security-red)

## 📄 Abstract

**CBRN-Sentinel** is a Green Agent that evaluates AI safety against **Chemical, Biological, Radiological, and Nuclear (CBRN)** threats. It uses a **Plan → Act → Verify → Answer** architecture with adversarial TAP attacks to calculate a **Defense Delta** score—measuring the gap between baseline safety and an agent's resilience under attack.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_ORG/cbrn-sentinel.git
cd cbrn-sentinel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

### Run as A2A Server (AgentBeats)

```bash
python -m src.server --host 0.0.0.0 --port 8000 --card-url http://localhost:8000
```

**Endpoints:**

- `GET /.well-known/agent.json` - Agent Card
- `POST /` - Assessment endpoint
- `GET /health` - Health check

### Run CLI Evaluation

```bash
python -m src.main --dataset tests/cbrn_comprehensive.csv --limit 5
```

## 🐳 Docker

```bash
# Build (linux/amd64 required for GitHub Actions)
docker build --platform linux/amd64 -t ghcr.io/YOUR_ORG/cbrn-sentinel:v1 .

# Run
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY ghcr.io/YOUR_ORG/cbrn-sentinel:v1
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Baseline Score** | Safety without adversarial pressure (0.0-1.0) |
| **Attack Score** | Safety under TAP attack |
| **Defense Delta** | Gap between baseline and attack |
| **Weighted Score** | Difficulty-adjusted (Easy=1x → Expert=4x) |

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CBRN-Sentinel                           │
├─────────────────────────────────────────────────────────────┤
│  PLAN    │  Decompose task into evaluation steps            │
│  ACT     │  Execute TAP attacks, collect responses          │
│  VERIFY  │  LlamaGuard judges safety, calculate delta       │
│  ANSWER  │  Compile results, generate report                │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
cbrn-sentinel/
├── src/
│   ├── server.py        # A2A server (AgentBeats entry)
│   ├── main.py          # CLI entry point
│   ├── agent/           # GAIA-style agent components
│   │   ├── planner.py   # Task decomposition
│   │   ├── verifier.py  # Step/final verification
│   │   └── cbrn_agent.py # Main execution loop
│   ├── modules/         # TAP engine, judging
│   └── benchmarks/      # Dataset loaders
├── tests/
│   └── cbrn_comprehensive.csv  # 16 CBRN scenarios
├── Dockerfile
└── requirements.txt
```

## 🔧 A2A Protocol

**Assessment Request:**

```json
{
  "participants": {"target": "http://purple-agent:8000"},
  "config": {"dataset": "tests/cbrn_comprehensive.csv", "limit": 5}
}
```

**Response:**

```json
{
  "results": [...],
  "pass_rate": 1.0,
  "avg_baseline": 0.9,
  "avg_delta": -0.1
}
```

---
*Built for AgentBeats 2026*
