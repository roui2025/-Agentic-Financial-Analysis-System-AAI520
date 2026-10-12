"""Team integration demo (requires finished specialists and local Ollama)."""
from agent_memory import AgentMemory
from coordinator_agent import run_research
from evaluator import evaluate_and_optimize
from financial_agent import analyze_financials
from news_agent import analyze_news


def main():
    report = run_research(
        "AAPL",
        "Analyze Apple's recent financial performance and news.",
        news_agent=analyze_news,
        financial_agent=analyze_financials,
        memory=AgentMemory(),
    )
    # INSERT HERE (Francisco): integrate and test the evaluator-optimizer.
    evaluated = evaluate_and_optimize(report)
    print(evaluated)


if __name__ == "__main__":
    main()
