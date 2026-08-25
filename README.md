# Research Agent

A command-line research agent, built from scratch in a single Python file —
no LangChain, LangGraph, CrewAI, or any other agent framework. It searches
the web, reads pages, and produces a sourced research brief through an
explicit SEARCH / READ / FINISH loop.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with three values for an OpenAI-compatible chat completions
API:

```
API_BASE_URL=https://api.example.com/v1
API_KEY=sk-...
MODEL=gpt-4o-mini
```

`API_BASE_URL` should not include `/chat/completions` — that's appended
automatically. If any of the three variables is missing, the program prints
which one and stops; it never guesses a model name.

## Usage

```bash
python research_agent.py                 # run the research agent on a question you type in
python research_agent.py --eval           # same, then evaluate the completed run
python research_agent.py search "query"   # run search_web() on its own
python research_agent.py read "url"       # run read_webpage() on its own
python research_agent.py test-connection  # verify the API call chain works, without researching
```

## How it works

**Config & API call.** Settings load from `.env` via `python-dotenv`. Chat
completions are called over plain `requests` — `Authorization: Bearer`
header (never the URL), a browser-style `User-Agent` (some services 403
without one), a 60s timeout, and retries on 429 (honoring `Retry-After`) and
5xx, up to 3 attempts each.

**Tools.** `search_web(query)` (DDGS, up to 5 results) and
`read_webpage(url)` (requests + BeautifulSoup, visible text capped at 5000
characters) never raise — a failure prints the status/error and is *encoded
into the return value* (`"[failed to read: ...]"`, or a marker search
result) so the agent loop and eval mode can see exactly what happened.

**The agent loop** (`STEP_LIMIT = 8`). Each step the model sees the goal and
the full history so far, and replies with one JSON action:

```json
{"reason": "one short sentence", "action": "SEARCH", "query": "..."}
{"reason": "one short sentence", "action": "READ",   "url": "..."}
{"reason": "one short sentence", "action": "FINISH", "report": "..."}
```

Every kind of runtime failure — a failed search, a page that won't load, a
malformed JSON reply, a total API failure after retries, a duplicate `READ`,
or a premature/empty `FINISH` — is recorded as an entry in the run state and
the loop continues; nothing gathered so far is thrown away. The only thing
that stops the program outright is a configuration problem that could never
have succeeded (missing `.env` settings, checked once before the loop
starts).

**FINISH is gated**, not just requested: it's refused unless the report is
non-empty *and* at least 3 distinct pages have been read successfully
(`MIN_PAGES_READ`). Findings in the report must each end with the URL they
came from in square brackets, or `[no source]` if it's the model's own
knowledge.

**The printed brief** has four model-written sections — Question, Findings,
Comparison, Recommendation — followed by two source lists the *program*
computes directly from the run state, not from anything the model claims:

```
Pages read:   URLs actually opened and read successfully
Also found:   URLs seen in search results but never opened
```

A page that failed to load appears under neither heading.

**`--eval`** runs the loop, then checks five things about the completed run
— search was used, more than one distinct source was read, the run finished
within the step limit, the brief has a recommendation, and it lists at least
three sources — reading `state` and the printed report directly rather than
the model's own opinion of how it did. Prints PASS/FAIL per check and a
total score.

## Files

- `research_agent.py` — the entire program
- `requirements.txt` — direct dependencies only (`requests`, `python-dotenv`, `ddgs`, `beautifulsoup4`)
- `.env.example` — the three settings to fill in, with no real values
