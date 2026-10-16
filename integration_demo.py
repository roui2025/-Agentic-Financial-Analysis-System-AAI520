"""Team integration demo (requires finished specialists and local Ollama).

Usage: python integration_demo.py [TICKER] ["question"]
The full result is also saved to integration_output.json.
"""
import json
import sys

from agent_memory import AgentMemory
from coordinator_agent import run_research
from evaluator import evaluate_and_optimize
from financial_agent import analyze_financials
from news_agent import analyze_news


def main(ticker="AAPL",
         question="Analyze Apple's recent financial performance and news."):
    report = run_research(
        ticker,
        question,
        news_agent=analyze_news,
        financial_agent=analyze_financials,
        memory=AgentMemory(),
    )
    print("Routes:", report["plan"]["routes"])
    for route, result in report["results"].items():
        print(f"{route}: {result.get('error') or result.get('summary')}")

    evaluated = evaluate_and_optimize(report)
    for step in evaluated["iterations"]:
        print(f"\nIteration {step['iteration']}: {step['scores']} "
              f"passed={step['passed']}")
        for item in step["feedback"]:
            print("  -", item)
    print(f"\nFinal analysis (iteration {evaluated['selected_iteration']}, "
          f"stopped because: {evaluated['stop_reason']}):")
    print(evaluated["report"]["final_analysis"])

    with open("integration_output.json", "w", encoding="utf-8") as file:
        json.dump(evaluated, file, indent=2, default=str)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        main(symbol, sys.argv[2] if len(sys.argv) > 2 else
             f"Analyze {symbol}'s recent financial performance and news.")
    else:
        main()
