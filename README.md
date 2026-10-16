# -Agentic-Financial-Analysis-System-AAI520
A multi-agent financial analysis system that researches stocks using market data, financial news, routing, prompt chaining, self-reflection, and evaluator-optimizer workflows. Developed as the final team project for AAI-520.

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
