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
    python research_agent.py            interactive connectivity test
    python research_agent.py search Q   run search_web(Q) on its own
    python research_agent.py read URL   run read_webpage(URL) on its own
"""

import argparse
import os
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
    """Send a chat completions request and return the reply text.

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
        "messages": [{"role": "user", "content": prompt}],
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


def run_interactive():
    """Ask for a question and run the prompt-1 connectivity test."""
    print("Research Agent")
    print("--------------")

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
        description="Research agent: interactive connectivity test, or "
        "standalone tool testing via subcommands."
    )
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="run search_web(query) on its own")
    search_parser.add_argument("query", help="the search query")

    read_parser = subparsers.add_parser("read", help="run read_webpage(url) on its own")
    read_parser.add_argument("url", help="the page URL to fetch")

    args = parser.parse_args()

    if args.command == "search":
        run_search_command(args.query)
    elif args.command == "read":
        run_read_command(args.url)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
