# Agentic Financial Analysis System (AAI-520)

A multi-agent financial analysis system that researches stocks using market data, financial news, routing, prompt chaining, self-reflection, and evaluator-optimizer workflows. Developed as the final team project for AAI-520.

## Team

- **Reema:** coordinator agent (planning, routing, self-reflection), memory, and integration.
- **Angelique:** news agent (news ingestion, classification, event extraction and summary).
- **Francisco:** Yahoo Finance data loader, financial agent, and evaluator-optimizer.

## How it works

1. The user gives a ticker and a question, for example AAPL and "Analyze Apple's recent financial performance and news."
2. The coordinator plans the research and decides which agents are needed: financial, news, or both. It also reads notes saved from earlier runs on the same ticker.
3. The financial agent loads one year of prices, the latest annual statements and valuation data, calculates return, risk and ratio metrics, and writes a short summary.
4. The news agent loads recent headlines, classifies each one (relevance, category, sentiment), extracts events and writes a short summary.
5. The coordinator reviews whether the research was complete and saves a short note for future runs.
6. The evaluator-optimizer writes the final answer, scores it, gives feedback and rewrites it, for up to 3 rounds, and keeps the best version.

Where each part of the assignment lives:

| Concept | Where |
|---|---|
| Planning research steps | `coordinator_agent.py` (`plan_research`) |
| Routing to specialist agents | `coordinator_agent.py` (`plan_research`) |
| Tool use (Yahoo Finance) | `data_loader.py`, used by both agents |
| Prompt chaining | `news_agent.py` (ingest, preprocess, classify, extract, summarize) |
| Self-reflection | `coordinator_agent.py` (`reflect_on_process`) |
| Memory across runs | `agent_memory.py` |
| Evaluator-optimizer | `evaluator.py` (`evaluate_and_optimize`) |

## How to run

1. Set up Python (3.10 or newer, tested with 3.12) and install the packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Install [Ollama](https://ollama.com) and download the model. Ollama has to be running while the system runs. The Ollama app starts it automatically; otherwise run `ollama serve` in a separate terminal.

   ```bash
   ollama pull llama3.2
   ```

3. Run the full system. It uses AAPL by default, or you can pass a ticker and a question:

   ```bash
   python integration_demo.py
   python integration_demo.py MSFT "How has Microsoft performed over the past year?"
   ```

   A full run takes a few minutes. The complete result is saved to `integration_output.json`.

## Notebooks

To open the notebooks, install Jupyter with `pip install notebook` and run `jupyter notebook`.

- `data_loading_demo.ipynb`: the raw Yahoo Finance data for one ticker.
- `financial_evaluator_demo.ipynb`: the financial agent's metrics and price chart, then the full system with each evaluator-optimizer round (scores, feedback and rewrites).

## Project files

| File | What it does |
|---|---|
| `coordinator_agent.py` | Plans the research, routes to the agents and reviews the process. |
| `agent_memory.py` | Saves short notes per ticker in `research_memory.json` for later runs. |
| `data_loader.py` | Loads prices, financial statements, key metrics and news from Yahoo Finance. |
| `financial_agent.py` | Calculates return, risk and ratio metrics and summarizes them. |
| `news_agent.py` | Classifies news headlines, extracts events and summarizes the news. |
| `evaluator.py` | Scores the final answer and improves it with feedback. |
| `integration_demo.py` | Runs the whole system from the command line. |
| `TEAM_INTEGRATION.md` | The team's original integration plan. |

## Limitations

- llama3.2 is a small model that runs locally. Its answers can be repetitive, and the evaluator rarely gives top scores, so the loop often stops after 3 rounds and keeps the best draft.
- Data comes from Yahoo Finance through the `yfinance` library. Some fields can be missing or delayed, and the financial statements are annual, not quarterly.
- The news analysis uses headlines and short summaries, not full articles.
- Results change between runs because prices, news and the model's wording change.
- This is a class project, not investment advice.
