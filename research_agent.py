#!/usr/bin/env python
# coding: utf-8
"""
research_agent.py

A beginner-friendly command-line program. It asks the user for a research
question, prints it back, and sends it to an OpenAI-compatible chat
completions API over plain HTTP.

Configuration is read from a .env file in the current directory:
    API_BASE_URL   e.g. https://api.openai.com/v1
    API_KEY        your API key (sent in the Authorization header, never the URL)
    MODEL          the model name to use (e.g. gpt-3.5-turbo)

Usage:
    python research_agent.py                 run the research agent loop
    python research_agent.py --eval           run the loop, then evaluate the run
    python research_agent.py search Q        run search_web(Q) on its own
    python research_agent.py read URL        run read_webpage(URL) on its own
    python research_agent.py test-connection run the prompt-1 connectivity test
"""

import argparse
import json
import os
import re
import sys
import time
from email.utils import parsedate_to_datetime

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv


# =============================================================================
# Constants
# =============================================================================

# Every request must time out after 60 seconds so the program never hangs.
REQUEST_TIMEOUT = 60

# How many times we are allowed to retry after the initial attempt.
MAX_RETRIES = 3

# The agent gets at most this many SEARCH/READ/FINISH steps per run. This is
# the single source of truth: the loop bound below and the number the model
# is told in its instructions both come from this constant, so they can
# never drift apart.
STEP_LIMIT = 6

# A browser-style User-Agent. Some services return 403 (error code 1010)
# and refuse the request unless one is sent.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# =============================================================================
# Configuration
# =============================================================================

def load_settings():
    """Read API_BASE_URL, API_KEY and MODEL from .env.

    If any one of them is missing, print which one(s) are missing and stop.
    Never guesses a model name.
    """
    load_dotenv()

    required = ["API_BASE_URL", "API_KEY", "MODEL"]
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        for name in missing:
            print(f"Missing setting: {name} is not set in .env")
        print("Please add it to your .env file. See .env.example for the format.")
        sys.exit(1)

    base_url = os.getenv("API_BASE_URL").strip().rstrip("/")
    api_key = os.getenv("API_KEY").strip()
    model = os.getenv("MODEL").strip()
    return base_url, api_key, model


# =============================================================================
# Tools: search_web / read_webpage
# =============================================================================

MAX_SEARCH_RESULTS = 5
READ_TEXT_LIMIT = 2000


def search_web(query):
    """search_web(query): searches the web for `query` and returns up to 5
    results, each with a title, url, and snippet — call this to discover
    pages that might help answer the research question. On failure, returns
    a single-item list whose snippet explains what went wrong, instead of
    silently returning nothing."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
    except Exception as exc:
        print(f"Search failed: {type(exc).__name__}: {exc}")
        print(f"  query: {query}")
        return [{"title": "[search failed]", "url": "", "snippet": f"[search failed: {exc}]"}]

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]


def read_webpage(url):
    """read_webpage(url): fetches the page at `url` and returns its visible
    text, up to 2000 characters — call this to read the full content of one
    specific page, e.g. a URL returned by search_web. On failure, returns a
    string starting with "[failed to read:" explaining why, instead of an
    empty string — check for that prefix to tell a real read from a failed
    one."""
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.RequestException as exc:
        print(f"Page fetch failed: {type(exc).__name__}: {exc}")
        print(f"  url: {url}")
        return f"[failed to read: {exc}]"

    if not response.ok:
        print(f"Page fetch failed: HTTP {response.status_code}")
        print(f"  url: {url}")
        return f"[failed to read: HTTP {response.status_code} {response.reason}]"

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    if not text:
        print("Page fetch failed: page returned no readable text")
        print(f"  url: {url}")
        return "[failed to read: page returned no readable text]"
    return text[:READ_TEXT_LIMIT]


# =============================================================================
# Error printing helpers
# =============================================================================

def _print_request_error(exc):
    """A request that failed before we got an HTTP response (connection
    error, DNS failure, timeout, etc). Prints the real error, never a
    generic message. Never prints the request URL."""
    print("The request failed before reaching the server:")
    print(f"  error type: {type(exc).__name__}")
    print(f"  message: {exc}")


def _print_http_error(response):
    """A failed HTTP response, including its body. Never prints the URL."""
    print("The request failed:")
    print("  error type: HTTPError")
    print(f"  status: {response.status_code}")
    print(f"  message: {response.reason}")
    print("  response body:")
    print(response.text)


def _parse_retry_after(response):
    """Work out how many seconds to wait after a 429 response.

    Uses the Retry-After header if present (numeric seconds or an HTTP
    date); otherwise waits two seconds.
    """
    header = response.headers.get("Retry-After")
    if header:
        header = header.strip()
        try:
            seconds = float(header)
            return seconds, f"Retry-After header says {header}s"
        except ValueError:
            try:
                retry_date = parsedate_to_datetime(header)
                wait = retry_date.timestamp() - time.time()
                if wait > 0:
                    return wait, f"Retry-After date '{header}'"
            except (TypeError, ValueError):
                pass
    return 2.0, "no Retry-After header present"


# =============================================================================
# Chat completions call
# =============================================================================

def call_chat_completions(base_url, model, prompt, api_key):
    """Send a single user-message chat completions request and return the
    reply text. Thin wrapper around call_chat_completions_messages() for
    callers that just have one prompt string (e.g. the prompt-1
    connectivity test)."""
    return call_chat_completions_messages(
        base_url, model, [{"role": "user", "content": prompt}], api_key
    )


def call_chat_completions_messages(base_url, model, messages, api_key):
    """Send a chat completions request with an explicit messages list (e.g.
    a system prompt plus a user message) and return the reply text.

    Retries on 429 (rate limited, honoring Retry-After) and 5xx (server
    errors, fixed 2s wait), up to MAX_RETRIES times each. Returns None if no
    answer could be obtained; an explanatory message has already been
    printed in that case.
    """
    endpoint = base_url + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    payload = {
        "model": model,
        "messages": messages,
    }

    retries_used = 0
    while True:
        try:
            response = requests.post(
                endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
        except requests.exceptions.RequestException as exc:
            _print_request_error(exc)
            return None

        status = response.status_code

        if status == 200:
            return _extract_content(response)

        if status == 429:
            if retries_used >= MAX_RETRIES:
                print(
                    f"Rate limited (429) and already retried "
                    f"{MAX_RETRIES} times. Giving up."
                )
                _print_http_error(response)
                return None
            wait, reason = _parse_retry_after(response)
            print(
                f"Rate limited (429). Waiting {wait:.1f}s before retrying "
                f"({retries_used + 1}/{MAX_RETRIES}). {reason}."
            )
            time.sleep(wait)
            retries_used += 1
            continue

        if 500 <= status < 600:
            if retries_used >= MAX_RETRIES:
                print(
                    f"Server error ({status}) and already retried "
                    f"{MAX_RETRIES} times. Giving up."
                )
                _print_http_error(response)
                return None
            print(
                f"Server error ({status}). Waiting 2s before retrying "
                f"({retries_used + 1}/{MAX_RETRIES})."
            )
            time.sleep(2)
            retries_used += 1
            continue

        # Any other HTTP error (4xx etc.)
        _print_http_error(response)
        return None


def _extract_content(response):
    """Pull choices[0].message.content out of a 200 response.

    If the expected field is absent, print the whole response body and
    stop with a clear message.
    """
    try:
        data = response.json()
    except ValueError:
        print("The response did not contain valid JSON.")
        print("Full response body:")
        print(response.text)
        sys.exit(1)

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("The response did not contain choices[0].message.content.")
        print("Full response body:")
        print(response.text)
        sys.exit(1)


# =============================================================================
# Agent loop
# =============================================================================

def build_agent_instructions():
    """The system prompt sent on every agent step: what the three tools do,
    the step limit, the required JSON reply shape, and the FINISH report
    structure. STEP_LIMIT is interpolated in, not hardcoded, so the number
    the model is told always matches the number the loop actually enforces.
    """
    return f"""You are a research agent. You have three actions:

SEARCH: search the web. Reply with:
  {{"reason": "one short sentence", "action": "SEARCH", "query": "..."}}

READ: read one web page. Reply with:
  {{"reason": "one short sentence", "action": "READ", "url": "..."}}

FINISH: you have enough information to answer the goal. Reply with:
  {{"reason": "one short sentence", "action": "FINISH", "report": "..."}}

You have at most {STEP_LIMIT} steps total in this run, including this one.
If you have not FINISHed by step {STEP_LIMIT}, the run ends without a report,
so budget your steps.

Web page text returned by READ is often full of navigation, menus, and
boilerplate — do not assume it is clean; look past it for the actual
content.

A failed SEARCH or a page that will not load is a normal outcome, not a
reason to give up — read the observation on the next step and try a
different query or URL.

When you FINISH, write "report" as a research brief with exactly these
five sections, in this order, each starting with its label on its own line:

Question: <the research question>
Findings: <what you learned, citing where each finding came from>
Comparison: <how the findings compare or relate to each other>
Recommendation: <your recommendation or conclusion>
Sources: <the URLs you actually read>

Reply with ONLY a JSON object, nothing else. No markdown, no explanation,
no code fences.
"""


def ask_agent_model(goal, state, base_url, model, api_key):
    """Ask the model for its next action, given the goal and everything
    that has happened so far. Returns the raw reply text, or None if no
    reply could be obtained (an explanatory message has already been
    printed by call_chat_completions_messages in that case)."""
    messages = [
        {"role": "system", "content": build_agent_instructions()},
        {
            "role": "user",
            "content": f"Goal: {goal}\n\nState so far (JSON):\n{json.dumps(state, indent=2)}",
        },
    ]
    return call_chat_completions_messages(base_url, model, messages, api_key)


def parse_agent_decision(raw_reply):
    """Parse the model's reply as a JSON object, stripping markdown code
    fences first if present. Returns the parsed dict, or None if parsing
    still fails — in which case the raw reply and a clear message have
    already been printed. Unlike a failed SEARCH/READ, unparseable JSON is
    not recoverable: the caller is expected to stop the program."""
    cleaned = raw_reply.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print("The model's reply was not valid JSON.")
        print(f"  error: {exc}")
        print("Raw reply:")
        print(raw_reply)
        return None


def summarize_search_results(results):
    """One-line summary of a search_web() result, for the per-step log."""
    if not results:
        return "no results"
    titles = "; ".join(r["title"] for r in results[:3])
    if len(results) > 3:
        titles += f"; (+{len(results) - 3} more)"
    return f"{len(results)} result(s): {titles}"


def summarize_read_result(text, limit=150):
    """One-line summary of a read_webpage() result, for the per-step log."""
    if text.startswith("[failed to read:"):
        return text
    snippet = text[:limit].replace("\n", " ")
    if len(text) > limit:
        snippet += "..."
    return f"{len(text)} char(s): {snippet}"


def print_research_brief(report):
    print()
    print("=== RESEARCH BRIEF ===")
    print(report if report else "(the model returned an empty report)")


def run_agent(goal, base_url, api_key, model):
    """Run the SEARCH/READ/FINISH loop for at most STEP_LIMIT steps.

    Every failure during the run — a failed SEARCH, a page that will not
    load, an unparseable reply, or a total API failure after
    call_chat_completions_messages exhausts its own retries — is recovered
    the same way: recorded as an entry in state and the loop continues, so
    a single failure never throws away everything gathered so far. STEP_LIMIT
    is what bounds the total cost of that; the only thing that stops the
    whole program outright is a configuration problem that could never have
    succeeded in the first place (missing .env settings, checked once before
    the loop even starts).
    """
    state = []

    for step in range(1, STEP_LIMIT + 1):
        raw_reply = ask_agent_model(goal, state, base_url, model, api_key)
        if raw_reply is None:
            print(f"STEP {step}/{STEP_LIMIT}: no reply was obtained from the API — recorded as an error, retrying next step.")
            state.append({"action": "ERROR", "detail": "API call failed after retries exhausted"})
            continue

        decision = parse_agent_decision(raw_reply)
        if decision is None:
            print(f"STEP {step}/{STEP_LIMIT}: the model's reply could not be parsed as JSON — recorded as an error, retrying next step.")
            state.append({"action": "ERROR", "detail": f"unparseable JSON reply: {raw_reply}"})
            continue

        reason = decision.get("reason", "")
        action = decision.get("action")

        if action == "SEARCH":
            query = decision.get("query", "")
            print(f"STEP {step}/{STEP_LIMIT}: reason: {reason}")
            print(f"  action: SEARCH {query!r}")
            results = search_web(query)
            print(f"  observation: {summarize_search_results(results)}")
            state.append({"action": "SEARCH", "query": query, "result": results})

        elif action == "READ":
            url = decision.get("url", "")
            print(f"STEP {step}/{STEP_LIMIT}: reason: {reason}")
            print(f"  action: READ {url!r}")
            text = read_webpage(url)
            print(f"  observation: {summarize_read_result(text)}")
            state.append({"action": "READ", "url": url, "result": text})

        elif action == "FINISH":
            report = decision.get("report", "")
            print(f"STEP {step}/{STEP_LIMIT}: reason: {reason}")
            print("  action: FINISH")
            print_research_brief(report)
            return state, report

        else:
            print(f"STEP {step}/{STEP_LIMIT}: reason: {reason}")
            print(f"  action: unrecognized ({decision!r})")
            state.append({"action": "ERROR", "detail": f"unrecognized action: {decision!r}"})

    print(f"\nStep limit ({STEP_LIMIT}) reached before FINISH. No report was produced.")
    return state, None


# =============================================================================
# Evaluation mode
# =============================================================================

REPORT_SECTION_LABELS = ["Question", "Findings", "Comparison", "Recommendation", "Sources"]
URL_PATTERN = re.compile(r"https?://\S+")


def parse_report_sections(report):
    """Split a FINISH report into its five labeled sections (Question,
    Findings, Comparison, Recommendation, Sources), each stripped of
    surrounding whitespace. A label that never appears, or a missing/empty
    report, maps to "" — callers don't need to special-case None."""
    sections = {label: "" for label in REPORT_SECTION_LABELS}
    if not report:
        return sections

    positions = []
    for label in REPORT_SECTION_LABELS:
        match = re.search(rf"(?m)^{label}:", report)
        if match:
            positions.append((match.start(), label))
    positions.sort()

    for i, (start, label) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(report)
        content_start = start + len(label) + 1  # skip "Label:"
        sections[label] = report[content_start:end].strip()
    return sections


def run_evals(state, report):
    """Check five things about a completed run. Every check reads `state`
    (what actually happened) and `report` (what was actually printed) —
    never the model's own claim about how the run went."""
    checks = []

    used_search = any(s.get("action") == "SEARCH" for s in state)
    checks.append(("Used the search tool at least once", used_search))

    valid_read_urls = {
        s["url"] for s in state
        if s.get("action") == "READ" and not s.get("result", "").startswith("[failed to read:")
    }
    checks.append(("Consulted more than one distinct source", len(valid_read_urls) > 1))

    checks.append(("Stayed within the step limit", report is not None))

    sections = parse_report_sections(report)
    checks.append(("Brief contains a recommendation", bool(sections["Recommendation"])))

    source_urls = {u.rstrip(".,;)]}\"'") for u in URL_PATTERN.findall(sections["Sources"])}
    checks.append(("Brief lists at least three sources", len(source_urls) >= 3))

    print("\n--- EVAL RESULTS ---")
    passed = 0
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"{status}  {name}")
    print(f"\nScore: {passed} of {len(checks)}")

    return {
        "checks": [{"name": name, "passed": ok} for name, ok in checks],
        "score": f"{passed} of {len(checks)}",
    }


# =============================================================================
# CLI entry point
# =============================================================================

def build_connectivity_test_prompt(question):
    """Wrap the user's question in an instruction that stops the model from
    actually researching or answering it.

    At this stage the program is only proving that the request/response
    pipe to the API works end to end (.env loading, headers, retries,
    response parsing). Turning the question into a real research answer is
    later work (search/read tools, the agent loop) — not this checkpoint.
    """
    return (
        "This is only a connectivity test for an API integration. Do not "
        "research, investigate, or use your own knowledge to answer the "
        "question below. Just confirm you received it by replying with "
        "exactly this line, with the question filled in:\n"
        "Connection OK. Received question: <the question>\n\n"
        f"Question: {question}"
    )


def run_connectivity_test_command():
    """Ask for a question and run the prompt-1 connectivity test (checkpoint
    1 verification: proves the API call chain works, without the model
    actually researching anything)."""
    print("Research Agent — connectivity test")
    print("-----------------------------------")

    base_url, api_key, model = load_settings()

    try:
        question = input("Enter a research question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo question entered. Goodbye.")
        sys.exit(0)

    if not question:
        print("You did not enter a question. Nothing to do.")
        sys.exit(0)

    print(f"\nYour research question: {question}\n")

    test_prompt = build_connectivity_test_prompt(question)
    reply = call_chat_completions(base_url, model, test_prompt, api_key)
    if reply is None:
        print("\nNo reply was obtained from the API.")
        sys.exit(1)

    print("Model reply (proves the API call chain works — not a research answer):")
    print(reply)


def run_agent_interactive(eval_mode=False):
    """Ask for a research goal and run the full SEARCH/READ/FINISH agent
    loop on it (checkpoint 3: the tool-using agent). If eval_mode is set,
    also run the five checks from run_evals() on the completed run
    (checkpoint 5)."""
    print("Research Agent")
    print("--------------")

    base_url, api_key, model = load_settings()

    try:
        goal = input("Enter a research question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo question entered. Goodbye.")
        sys.exit(0)

    if not goal:
        print("You did not enter a question. Nothing to do.")
        sys.exit(0)

    print(f"\nGoal: {goal}\n")

    state, report = run_agent(goal, base_url, api_key, model)

    if eval_mode:
        run_evals(state, report)


def run_search_command(query):
    """Run search_web(query) on its own and print the results. On failure
    search_web already returns a single result explaining why (see its
    docstring), so there is nothing extra to handle here."""
    results = search_web(query)
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']}")
        print(f"   {r['url']}")
        print(f"   {r['snippet']}")


def run_read_command(url):
    """Run read_webpage(url) on its own and print the extracted text. On
    failure read_webpage already returns a "[failed to read: ...]" string
    explaining why, so there is nothing extra to handle here."""
    print(read_webpage(url))


def main():
    parser = argparse.ArgumentParser(
        description="Research agent: runs the SEARCH/READ/FINISH agent loop "
        "by default, or a standalone check via subcommands."
    )
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="run search_web(query) on its own")
    search_parser.add_argument("query", help="the search query")

    read_parser = subparsers.add_parser("read", help="run read_webpage(url) on its own")
    read_parser.add_argument("url", help="the page URL to fetch")

    subparsers.add_parser(
        "test-connection",
        help="checkpoint 1: verify the API call chain works, without researching",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="after the agent loop finishes, run the five checks from run_evals() "
        "on the completed run and print PASS/FAIL plus a total score",
    )

    args = parser.parse_args()

    if args.command == "search":
        run_search_command(args.query)
    elif args.command == "read":
        run_read_command(args.url)
    elif args.command == "test-connection":
        run_connectivity_test_command()
    else:
        run_agent_interactive(eval_mode=args.eval)


if __name__ == "__main__":
    main()
