"""Reema: autonomous research planning, routing, and process reflection.

Free LLM: local Ollama with llama3.2 (no API key required).
Run 'ollama pull llama3.2' first and keep Ollama running.
"""
import json
from urllib.error import URLError
from urllib.request import Request, urlopen

MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
SPECIALISTS = ("news", "financial")


def ask_llm(prompt):
    """Request a JSON response from a locally running Ollama model."""
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "format": "json",
    }).encode("utf-8")
    request = Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.load(response)["response"]
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Ollama is unavailable. Install Ollama and run "
            "'ollama pull llama3.2' before using this coordinator."
        ) from error


def _json_response(prompt):
    """Decode and validate the model's JSON object."""
    result = json.loads(ask_llm(prompt))
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the LLM.")
    return result


def plan_research(ticker, question, previous_notes=None):
    """Plan research steps and dynamically select specialist agents."""
    prompt = (
        "You coordinate investment research. Choose the specialist tools "
        "needed to answer the question. Available tools: 'news' (headlines "
        "and sentiment) and 'financial' (prices, financial statements and "
        "valuation). Respond ONLY as JSON with keys: steps (list of short "
        "steps), routes (list of tool names), reason (short explanation). "
        "IMPORTANT: routes must contain ONLY the exact strings 'financial' "
        "and/or 'news'. Do not put actions, descriptions, or subtool names "
        "in routes. Example for price question: "
        "{\\\"steps\\\":[\\\"Get prices\\\"],\\\"routes\\\":[\\\"financial\\\"],"
        "\\\"reason\\\":\\\"Price data needed\\\"}. "
        "Use only listed tools; choose both when appropriate. "
        f"Ticker: {ticker}. Question: {question}. "
        f"Previous process notes: {json.dumps(previous_notes or [])}."
    )
    # Retry once if a small model invents action names in the routes field.
    for attempt in range(2):
        plan = _json_response(prompt)
        routes = plan.get("routes")
        steps = plan.get("steps")
        if isinstance(routes, list) and routes and isinstance(steps, list):
            normalized = []
            valid = True
            for route in routes:
                if not isinstance(route, str):
                    valid = False
                    break
                specialist = route.strip().lower().split("/", 1)[0]
                if specialist not in SPECIALISTS:
                    valid = False
                    break
                normalized.append(specialist)
            if valid:
                plan["routes"] = list(dict.fromkeys(normalized))
                return plan
        if attempt == 0:
            prompt += (
                "\\nYour last response was invalid. Return a NEW JSON object. "
                "The routes array MUST contain only exact names from "
                "['financial', 'news']. Never use action names such as "
                "'Get current stock price'. Keep actions in steps instead. "
                f"Invalid response: {json.dumps(plan)}"
            )
    raise ValueError(
        f"LLM returned invalid routes after retry: {plan.get('routes')}. "
        "Try running the cell again."
    )


def reflect_on_process(ticker, question, plan, results):
    """Identify missing evidence or tool failures; no invented findings."""
    prompt = (
        "Review whether this investment research PROCESS was complete. "
        "Do not invent financial facts. Return ONLY JSON with keys "
        "'complete' (boolean), 'limitations' (list of strings), "
        "'next_steps' (list of strings). "
        f"Ticker: {ticker}. Question: {question}. "
        f"Plan: {json.dumps(plan)}. "
        f"Specialist results: {json.dumps(results, default=str)[:12000]}."
    )
    reflection = _json_response(prompt)
    if not isinstance(reflection.get("complete"), bool):
        raise ValueError("Reflection must include a boolean 'complete'.")
    for key in ("limitations", "next_steps"):
        if not isinstance(reflection.get(key), list):
            raise ValueError(f"Reflection must include a list: {key}.")
    return reflection


def run_research(ticker, question, news_agent, financial_agent, memory=None):
    """Run selected specialist functions and store process reflection.

    Each specialist must accept a ticker and return a JSON-compatible dict.
    An optional AgentMemory provides get(ticker) and save(ticker, report).
    """
    ticker = ticker.strip().upper()
    if not ticker or not ticker.replace(".", "").replace("-", "").isalnum():
        raise ValueError("Enter a valid stock ticker.")
    notes = memory.get(ticker) if memory else []
    plan = plan_research(ticker, question, notes)
    agents = {"news": news_agent, "financial": financial_agent}
    results = {}
    for route in plan["routes"]:
        try:
            results[route] = agents[route](ticker)
        except Exception as error:
            results[route] = {"error": str(error)}
    reflection = reflect_on_process(ticker, question, plan, results)
    report = {
        "ticker": ticker, "question": question,
        "plan": plan, "results": results, "reflection": reflection,
    }
    if memory:
        memory.save(ticker, report)
    return report
