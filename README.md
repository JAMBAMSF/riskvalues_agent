# RiskValuesAgent – Risk & Values Scoring Demo

![Build](https://img.shields.io/badge/CI-GitHub_Actions-informational)  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Status](https://img.shields.io/badge/Status-Demo-lightgrey)  

A demonstration agent that ranks companies by **risk-adjusted values alignment**, using:  
- **Risk score** (derived from beta, bucketed conservatively)  
- **Values score** (signals for climate, deforestation, diversity; aggregated with caps)  
- **Composite score** (weighted average of risk + values, default 50/50)  

⚠️ This project does **not** perform discounted cashflow (DCF) or comparables analysis.  
It is a **proof-of-concept agent** showing **tool-based retrieval, explainable scoring, and optional LLM-generated narratives.**  

---

## Key Points for Reviewers
- **Code is frozen** (no logic changes); this repo adds documentation, CI, and smoke tests only.  
- Emphasizes **tool-first scoring, LLM-last summarization**.  
- **LLM integration** is optional and only used for short, business-friendly summaries.  
- Reproducible with one-command setup and sample examples.  

---

## Repo Layout
./
  app.py                # CLI entrypoint
  ui_streamlit.py       # Streamlit UI
  core/                 # Scoring + planner logic
  tools/                # Risk + values tool wrappers
  tests/                # Smoke tests
  examples/             # Sample inputs (redacted/public)
  .env.example          # Example configuration
  requirements.txt      # Dependencies
  README.md             # This file

---

## Quickstart
- **Python**: 3.10+  

# 1) Create virtual env
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure environment variables
cp .env.example .env
# edit with your API keys (OpenAI / Anthropic / Vertex / Gemini optional)

### Run Examples
# 4) Ask about a company
python app.py ask "What’s the mission and sustainability profile of MSFT?"

# 5) Get recommendations
python app.py recommend --risk low --values climate,diversity --k 3 --explain

---

## Architecture (high-level)

flowchart TD
    A[Inputs: Ticker / Company] -->|fetch| B[Tools]
    B --> C[Risk Scorer (beta buckets)]
    B --> D[Values Scorer (climate, deforestation, diversity)]
    C --> E[Composite Scorer (weighted avg)]
    D --> E
    E --> F[LLM Summarizer (optional)]
    F --> G[Outputs: CLI, Streamlit, JSON, Markdown]

---

## Configuration

Environment variables (see `.env.example`):  

| Variable              | Purpose                                   |
|-----------------------|-------------------------------------------|
| `OPENAI_API_KEY`      | For OpenAI summarizer (optional)          |
| `ANTHROPIC_API_KEY`   | For Claude summarizer (optional)          |
| `VERTEX_PROJECT_ID`   | For Vertex AI summarizer (optional)       |
| `AWS_REGION`          | For Bedrock/SageMaker tools (optional)    |
| `DATA_DIR`            | Sample docs folder                        |

---

## Examples

- `examples/cim/` — redacted CIM snippets  
- `examples/sec/` — 10-K/10-Q filings  
- `examples/prompts/` — summarizer prompts  

---

## Testing
pytest -q -k smoke

---

## CI
- GitHub Actions run lint, type-check, and smoke tests on every push/PR.  

---

## Security & Compliance
- No secrets in repo.  
- Use `.env` or CI secrets store for API keys.  
- Example data only; no client/confidential data.  

---

## License
MIT (see `LICENSE`).  

---

## Acknowledgments
- Inspired by **agent orchestration** (LangGraph-style flows).  
- Demonstrates **risk/values scoring** + **LLM summarization**.  
- Designed for **clarity, explainability, and reviewer-friendliness.**  
