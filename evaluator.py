"""Francisco: evaluator-optimizer workflow.

This step runs after the coordinator. It writes an answer to the user's
question from the financial and news results, scores the answer, gives
feedback, and rewrites it. The loop stops when the answer passes or after
MAX_ITERATIONS rounds.

The scores come from llama3.2 acting as a reviewer, plus a few checks in
plain Python: numbers that are not in the data, an agent's results being
left out, a missing date, and investment advice. When one of these checks
fails it caps the score, since a small model like llama3.2 tends to rate
almost everything highly.
"""
import json
import re
from datetime import date
from urllib.error import URLError
from urllib.request import Request, urlopen

from coordinator_agent import MODEL, OLLAMA_URL

MAX_ITERATIONS = 3
PASS_SCORE = 4
DIMENSIONS = ("completeness", "grounding", "clarity")
MIN_WORDS, MAX_WORDS = 60, 280


def ask_llm_json(prompt, temperature=0):
    """Send a prompt to llama3.2 and return its reply as a dict.

    This is the same request as ask_llm in coordinator_agent.py with two
    extra options. Temperature 0 keeps the scores consistent between runs,
    and a larger num_ctx makes sure the evidence fits in the prompt.
    """
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": temperature, "num_ctx": 8192},
    }).encode("utf-8")
    request = Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            text = json.load(response)["response"]
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Ollama is not running. Install Ollama, run "
            "'ollama pull llama3.2' and keep the server running."
        ) from error
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the LLM.")
    return result


# Checking numbers against the data

_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    rf"|\b(?:{_MONTH_NAMES})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAMES})\b\.?(?:,?\s+\d{{4}})?",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*%|\s*percent)?")
_WINDOW = re.compile(r"\s*-?\s*(?:day|week|month|year|quarter)s?\b", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# The data stores values as plain units, percents or billions, but the text
# can write the same value another way (134% for a 1.34 ratio, or $4.99
# trillion for 4991 billion), so each value is also compared at these scales.
_SCALES = (1, 100, 1e3, 1e-3, 1e-6, 1e-9, 1e-12)


def _claimed_numbers(text):
    """Find the numbers stated in the text as (raw, value, tolerance).

    Dates, years, small counts (12 or under) and time windows like
    '50-day' or '1-year' are skipped because they are not findings.
    """
    text = _DATE.sub(" ", text)
    found = []
    for match in _NUMBER.finditer(text):
        whole, decimals, percent = match.groups()
        value = float(whole.replace(",", "") + ("." + decimals if decimals else ""))
        if not decimals and not percent:
            if value <= 12 or 1900 <= value <= 2100:
                continue
            if _WINDOW.match(text, match.end()):
                continue
        places = len(decimals) if decimals else 0
        tolerance = max(0.5 * 10 ** -places, 0.005 * value)
        found.append((match.group(0).strip(), value, tolerance))
    return found


def _evidence_values(evidence):
    """Collect every number in the evidence, including numbers inside strings."""
    values = []
    if isinstance(evidence, bool) or evidence is None:
        return values
    if isinstance(evidence, (int, float)):
        values.append(abs(float(evidence)))
    elif isinstance(evidence, str):
        values += [float(n.replace(",", "")) for n in _ANY_NUMBER.findall(evidence)]
    elif isinstance(evidence, dict):
        for item in evidence.values():
            values += _evidence_values(item)
    elif isinstance(evidence, (list, tuple)):
        for item in evidence:
            values += _evidence_values(item)
    return values


def _supported(value, tolerance, evidence_values):
    return any(
        abs(value - known * scale) <= tolerance
        for known in evidence_values for scale in _SCALES
    )


def unsupported_numbers(text, evidence):
    """Return the numbers in the text that can't be found in the evidence."""
    known = _evidence_values(evidence)
    bad = []
    for raw, value, tolerance in _claimed_numbers(text or ""):
        if not _supported(value, tolerance, known) and raw not in bad:
            bad.append(raw)
    return bad


def _date_forms(iso_date):
    """Common ways to write a date, e.g. 2026-10-15, October 15 or Oct 15."""
    try:
        day = date.fromisoformat(iso_date)
    except (TypeError, ValueError):
        return []
    month, short = day.strftime("%B").lower(), day.strftime("%b").lower()
    return [iso_date, f"{month} {day.day}", f"{short} {day.day}",
            f"{short}. {day.day}", f"{day.day} {month}", f"{day.day} {short}"]


# Steps that call the LLM

def _evidence(report):
    """What the analysis is allowed to use: agent results and coordinator notes."""
    return {
        "results": report.get("results", {}),
        "process_limitations": (report.get("reflection") or {}).get("limitations", []),
    }


def _evidence_json(report):
    return json.dumps(_evidence(report), default=str)[:12000]


_WRITING_RULES = (
    "Rules: Use ONLY facts, numbers and dates that appear in EVIDENCE, "
    "copied as written (you may round). Do not calculate new numbers. "
    "Say what date each figure is from (the price data 'as_of' date and the "
    "'fiscal_year_end' of the statements). If a specialist result has an "
    "'error', say that this part of the data was unavailable. If both "
    "financial and news results are present, cover both. Mention the main "
    "limitations. Do not predict prices and do not give buy or sell advice. "
    "Write plain sentences: never write JSON field names such as as_of or "
    "market_cap_bn, do not repeat facts in parentheses, and never write the "
    "word EVIDENCE in the text. "
    f"Write {MIN_WORDS + 60} to {MAX_WORDS - 100} words of plain prose. "
)


def generate_analysis(report):
    """Write the first draft using only the agents' results."""
    prompt = (
        "You write investment research for a non-expert reader. Answer the "
        f"QUESTION about {report.get('ticker')} using the EVIDENCE. "
        + _WRITING_RULES +
        'Return ONLY JSON: {"analysis": "..."}. '
        f"QUESTION: {report.get('question')} "
        f"EVIDENCE: {_evidence_json(report)}"
    )
    return str(ask_llm_json(prompt).get("analysis", "")).strip()


def refine_analysis(analysis, feedback, report):
    """Rewrite the draft based on the feedback without adding new facts.

    At temperature 0 the model sometimes sends the same text back. In that
    case it tries once more at a higher temperature so the rewrite actually
    changes something. Returns the original text if both tries fail.
    """
    prompt = (
        "Rewrite the ANALYSIS so that it fixes every item in FEEDBACK. Keep "
        "what is already correct and do not add facts that are not in "
        "EVIDENCE. " + _WRITING_RULES +
        'Return ONLY JSON: {"analysis": "..."}. '
        f"QUESTION: {report.get('question')} "
        f"FEEDBACK: {json.dumps(feedback)} "
        f"ANALYSIS: {analysis} "
        f"EVIDENCE: {_evidence_json(report)}"
    )
    for temperature in (0, 0.7):
        rewritten = str(ask_llm_json(prompt, temperature).get("analysis", "")).strip()
        if rewritten and rewritten != analysis:
            return rewritten
    return analysis


def _llm_review(analysis, report):
    """Have llama3.2 score the draft from 1 to 5 and list what to fix."""
    prompt = (
        "You are a strict reviewer of investment research. Score the ANALYSIS "
        "against the EVIDENCE with integers from 1 (poor) to 5 (excellent). "
        "completeness: answers the QUESTION, uses every specialist result, "
        "gives dates and limitations. grounding: every fact and number is "
        "supported by EVIDENCE, nothing invented, no predictions or advice. "
        "clarity: clear, organized, concise, readable by a non-expert. "
        "Every figure in EVIDENCE already comes from Yahoo Finance, so do not "
        "ask for sources, calculation methods or currencies. "
        "In 'feedback' list specific problems in the ANALYSIS: quote the "
        "sentence from the ANALYSIS (never from the EVIDENCE) and say how to "
        "fix it. Use an empty list if there is nothing to fix. "
        "Return ONLY JSON with integer keys 'completeness', 'grounding', "
        "'clarity' and a list of strings 'feedback'. "
        f"QUESTION: {report.get('question')} "
        f"ANALYSIS: {analysis} "
        f"EVIDENCE: {_evidence_json(report)}"
    )
    review = ask_llm_json(prompt)
    scores = {}
    for key in DIMENSIONS:
        try:
            scores[key] = min(5, max(1, int(float(review.get(key)))))
        except (TypeError, ValueError):
            scores[key] = 3  # score couldn't be read, so use 3 and don't pass
    feedback = review.get("feedback")
    if isinstance(feedback, str):
        feedback = [feedback]
    if not isinstance(feedback, list):
        feedback = []
    feedback = [str(item).strip() for item in feedback if str(item).strip()]
    feedback = [item for item in feedback if _quotes_analysis(item, analysis)]
    return scores, feedback[:6]


_QUOTE = re.compile(r"""['"‘“](.{20,}?)['"’”]""")


def _quotes_analysis(item, analysis):
    """Return False if the feedback quotes a sentence that isn't in the analysis.

    The reviewer sometimes comments on sentences from the evidence instead
    of the analysis. Rewriting can't fix those, so they get dropped.
    """
    quote = _QUOTE.search(item)
    if not quote:
        return True
    return quote.group(1)[:30].lower() in analysis.lower()


# Rule checks

_UNAVAILABLE = ("unavailable", "not available", "could not", "couldn't",
                "failed", "error", "missing", "no data", "not retrieved")
_NEWS_WORDS = ("news", "headline", "article", "sentiment", "reported",
               "according to")
_ADVICE = re.compile(
    r"\b(?:we recommend|i recommend|should (?:buy|sell|invest)|"
    r"(?:is|are) a (?:good|great|bad|strong) (?:buy|investment)|strong buy|"
    r"buy the stock|sell the stock)\b",
    re.IGNORECASE,
)
_NO_NEWS = ("no news", "no articles", "no headlines", "no relevant news",
            "no news articles", "news was unavailable", "news was not available",
            "news articles were unavailable", "news articles were not available")
# Phrases that only make sense if news was actually found, for example
# "articles were available, but their sentiment was neutral".
_NEWS_CLAIM = re.compile(
    r"\b(?:news|articles?|headlines?)\s+(?:were|was|is|are)\s+(?:available|found)\b"
    r"|\bsentiment\s+(?:was|is)\s+(?:neutral|positive|negative|mixed)\b",
    re.IGNORECASE,
)
_NEGATED = re.compile(r"\b(?:no|not|without)\b[\w\s]*$", re.IGNORECASE)


def _claims_news(analysis):
    """Check if the text says news was found. 'No news was found' doesn't count."""
    return any(not _NEGATED.search(analysis[max(0, m.start() - 25):m.start()])
               for m in _NEWS_CLAIM.finditer(analysis))


_FIELD_NAME = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")  # e.g. as_of, market_cap_bn
_META = re.compile(r"\bevidence\b", re.IGNORECASE)  # e.g. "$342.00 (from EVIDENCE)"


def _rule_checks(analysis, report):
    """Checks done in Python, without the LLM.

    Returns (caps, feedback). caps holds the highest score a dimension can
    get after a failed check, and feedback says what to change.
    """
    caps, feedback = {}, []
    text = analysis.lower()

    def fail(dimension, cap, message):
        caps[dimension] = min(cap, caps.get(dimension, 5))
        feedback.append(message)

    words = len(analysis.split())
    if words == 0:
        fail("completeness", 1, "The analysis is empty. Write the answer.")
        fail("clarity", 1, "The analysis is empty.")
        return caps, feedback

    bad = unsupported_numbers(analysis, _evidence(report))
    if bad:
        fail("grounding", 2,
             f"These numbers are not in the evidence: {', '.join(bad)}. "
             "Remove them or replace them with the exact values from the "
             "evidence.")
    if _ADVICE.search(analysis):
        fail("grounding", 3,
             "Remove the investment advice and describe the data instead.")

    results = report.get("results", {})
    for route in (report.get("plan") or {}).get("routes", []):
        result = results.get(route)
        if not isinstance(result, dict):
            continue
        if "error" in result:
            if not any(word in text for word in _UNAVAILABLE):
                fail("completeness", 3,
                     f"The {route} data failed to load ({result['error']}). "
                     "Say that this part of the data was unavailable.")
            continue
        if route == "financial":
            metrics = result.get("metrics") or {}
            key_values = [metrics.get("latest_price"), metrics.get("return_1y_pct")]
            key_values = [abs(v) for v in key_values if v is not None]
            if key_values and not any(
                _supported(value, tolerance, key_values)
                for _, value, tolerance in _claimed_numbers(analysis)
            ):
                fail("completeness", 3,
                     "Use the financial results: give the latest price "
                     f"({metrics.get('latest_price')}) and the 1-year return "
                     f"({metrics.get('return_1y_pct')}%).")
            as_of = result.get("as_of")
            forms = _date_forms(as_of)
            if forms and not any(form in text for form in forms):
                fail("completeness", 3,
                     f"State the date of the price data ({as_of}).")
        if route == "news" and not result.get("article_count"):
            if (_claims_news(analysis)
                    or not any(phrase in text for phrase in _NO_NEWS)):
                fail("completeness", 3,
                     "No news articles were retrieved. Say that there was no "
                     "news, and do not describe news sentiment or headlines.")
        elif route == "news":
            sentiment = str(result.get("sentiment") or "").lower()
            needs_sentiment = bool(sentiment)
            if (not any(word in text for word in _NEWS_WORDS)
                    or (needs_sentiment and sentiment not in text)):
                fail("completeness", 3,
                     f"Cover the news results: say the overall news sentiment "
                     f"({sentiment or 'not available'}) and what the headlines "
                     "reported.")

    field_names = sorted(set(_FIELD_NAME.findall(analysis)))
    if field_names:
        fail("clarity", 3,
             f"Replace raw field names with plain words: {', '.join(field_names)}.")
    if _META.search(analysis):
        fail("clarity", 3,
             "Do not mention 'the evidence' or add source tags. Write the "
             "sentences for the reader.")
    if words < MIN_WORDS:
        fail("clarity", 3, f"Too short ({words} words). Explain the findings "
                           "in more detail.")
    elif words > MAX_WORDS:
        fail("clarity", 3, f"Too long ({words} words). Keep it under "
                           f"{MAX_WORDS} words.")
    return caps, feedback


def evaluate_analysis(analysis, report, pass_score=PASS_SCORE):
    """Score one draft. The rule checks cap the reviewer's scores."""
    caps, rule_feedback = _rule_checks(analysis, report)
    if analysis.strip():
        llm_scores, llm_feedback = _llm_review(analysis, report)
    else:
        llm_scores, llm_feedback = {key: 1 for key in DIMENSIONS}, []
    scores = {key: min(llm_scores[key], caps.get(key, 5)) for key in DIMENSIONS}
    scores["overall"] = round(sum(scores[key] for key in DIMENSIONS) / len(DIMENSIONS), 2)
    feedback = rule_feedback + [item for item in llm_feedback
                                if item not in rule_feedback]
    passed = not rule_feedback and all(scores[key] >= pass_score for key in DIMENSIONS)
    return {"scores": scores, "feedback": feedback,
            "rule_issues": len(rule_feedback), "passed": passed}


def evaluate_and_optimize(report, max_iterations=MAX_ITERATIONS, pass_score=PASS_SCORE):
    """Score, critique, refine, and re-score the combined analysis.

    The loop writes a draft and evaluates it. If it doesn't pass, the draft
    is rewritten using the feedback and evaluated again, up to
    max_iterations times. A rewrite can turn out worse than the draft
    before it, so a new draft only replaces the best one if it scores
    higher. The coordinator's report is returned with 'final_analysis'
    added and everything else left as it was.

    stop_reason says why the loop ended: 'passed', 'max_iterations', or
    'no_change' when the rewrite came back identical to the draft.
    """
    analysis = generate_analysis(report)
    iterations, best = [], None
    stop_reason = "max_iterations"
    for number in range(1, max_iterations + 1):
        evaluation = evaluate_analysis(analysis, report, pass_score)
        iterations.append({"iteration": number, "analysis": analysis, **evaluation})
        rank = (evaluation["passed"], -evaluation["rule_issues"],
                evaluation["scores"]["overall"])
        if best is None or rank > best[0]:
            best = (rank, iterations[-1])
        if evaluation["passed"]:
            stop_reason = "passed"
            break
        if number == max_iterations:
            break
        rewritten = refine_analysis(analysis, evaluation["feedback"], report)
        if rewritten == analysis:
            stop_reason = "no_change"  # scoring the same text again adds nothing
            break
        analysis = rewritten
    chosen = best[1]
    return {
        "report": {**report, "final_analysis": chosen["analysis"]},
        "scores": chosen["scores"],
        "feedback": chosen["feedback"],
        "iterations": iterations,
        "selected_iteration": chosen["iteration"],
        "passed": chosen["passed"],
        "stop_reason": stop_reason,
    }
