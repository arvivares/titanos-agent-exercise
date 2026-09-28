# Restaurant Booking Agent

A Python command-line assistant that finds available restaurants for tonight and confirms the restaurant, time, and group size before booking. LangGraph manages conversation state, Groq interprets requests, and application code validates choices and writes replies from returned restaurant data. Optional Langfuse integration records conversation traces when configured.

This assessment runs against a local simulation. The supplied `tools.py` and `restaurants.json` remain unchanged; bookings update memory, not a restaurant service.

[Run the application](#how-to-run-the-application) | [Assumptions](#assumptions) | [Chosen model](#chosen-model) | [Local checks](#local-checks)

## How to run the application

Use these instructions on **Linux, Windows, or macOS**. You need a local checkout, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a Groq API key to run the interactive assistant. Langfuse is optional; local checks do not need any credentials.

Python 3.12 is selected by `.python-version`; the project also supports Python 3.13. uv manages the project environment and locked dependencies. Initial setup may download Python and packages.

### 1. Install uv

Choose the instructions for your operating system. Skip installation if `uv --version` already works.

**Windows — PowerShell**

```powershell
winget install --id=astral-sh.uv -e
```

If WinGet is unavailable, use the Windows installer from the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

**Linux — Bash**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**macOS — Terminal (Zsh or Bash)**

If Homebrew is installed:

```bash
brew install uv
```

Otherwise, use the same standalone installer as Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reopen your terminal after installation, then verify:

```console
uv --version
```

On Windows, use PowerShell for the Windows commands below. Git Bash users can install uv from PowerShell, reopen Git Bash, and use the Bash configuration commands. For WSL, follow the Linux instructions and keep a separate checkout and `.venv` inside WSL; do not reuse a Windows virtual environment.

### 2. Create the development environment

Open a terminal in the repository root, where `pyproject.toml` is located. The following commands are the same on all three operating systems:

```console
uv sync --locked
```

This creates `.venv` and installs the application and development dependencies, including Ruff. No manual environment activation is needed. Run subsequent commands from this directory.

### 3. Configure the interactive assistant

Create `.env` from [.env.example](.env.example). These commands preserve an existing configuration.

**Windows — PowerShell**

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

**Linux / macOS / Git Bash**

```bash
if [ ! -e .env ]; then
    cp .env.example .env
fi
```

Open `.env` in your editor and set `GROQ_API_KEY`. This is the only required credential:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Leave the Langfuse keys blank to run without tracing, or configure both keys as described under [Configuration](#configuration). Keep credentials out of version control. Skip this step if you only want to run the local checks.

### 4. Run

```console
uv run --locked python -m restaurant_booking
```

Type `/exit` or `/quit` to stop. The application runs from this checkout because it loads the supplied root-level tools and data. Use the same `uv run` command on all platforms.

Help and version commands work without credentials or external requests:

```console
uv run --locked python -m restaurant_booking --help
uv run --locked python -m restaurant_booking --version
```

## Configuration

The app loads `.env` from the project root. Existing environment variables take precedence; `.env` is excluded from Git.

| Variable | Requirement or default |
| --- | --- |
| `GROQ_API_KEY` | Required Groq API key |
| `LANGFUSE_PUBLIC_KEY` | Optional; enable tracing together with `LANGFUSE_SECRET_KEY` |
| `LANGFUSE_SECRET_KEY` | Optional; enable tracing together with `LANGFUSE_PUBLIC_KEY` |
| `GROQ_MODEL` | `openai/gpt-oss-120b`; must support tool calling |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com`; used only when tracing is configured |

Langfuse tracing is enabled only when both keys are nonempty. If either key is missing or blank, the assistant runs without a Langfuse client, callback, or trace upload; setting the base URL alone does not enable tracing. The Langfuse package remains part of the locked installation.

When enabled, Langfuse receives conversation inputs and outputs grouped by session, and pending traces are flushed on exit. Tracing initialization or flush failures produce a warning without stopping the CLI. See [Langfuse troubleshooting](docs/troubleshooting.md#optional-langfuse-tracing).

## Assumptions

The implementation makes the following scope and behavior decisions:

| Topic | Assumption and consequence |
| --- | --- |
| Date and time | The supplied catalogue represents **tonight**. Times are the returned 24-hour `HH:MM` labels; the application does not convert time zones, choose another date, or remove slots based on the computer's current clock. |
| Required information | A positive party size is required before searching. Cuisine, area, price level, and rating preferences are optional. Once the group is known, the assistant searches without requiring those preferences. |
| Search results | Show at most three restaurants, ordered by descending rating, after checking capacity and available times. Retain the group and existing filters across turns until changed or explicitly cleared. |
| Prices and restaurant facts | Price means a category of one to three euro signs, not an exact amount or a per-person quote. Function results are the source of restaurant facts. The `vegetarian` category is a catalogue label; it does not establish additional dietary guarantees. |
| Conversation lifetime | One CLI process handles a local conversation. Conversation state and simulated availability stay in memory; restarting resets them. There is no durable booking database or real restaurant integration. |
| Language | Replies use English application templates. The model may interpret requests in other languages, but multilingual correctness is not guaranteed. Pending reservations use recognized English confirmation phrases; `yes` and `no` are the simplest choices. |
| Booking and cancellation | Displaying a time does not hold a table. Execution requires a separate confirmation of the exact restaurant, time, and group size. Cancellation clears a pending candidate; it does not cancel an already completed reservation. |
| Availability conflict | The supplied tool deliberately rejects the first otherwise valid booking in each process after removing its slot. This is expected test behavior. Refresh availability and require another choice and confirmation instead of retrying automatically. |

The supplied `tools.py` and `restaurants.json` are treated as immutable assessment inputs. Validation and conversational behavior are implemented in the application adapters and graph. See the [architecture](docs/architecture.md#assessment-requirements) for how the six agent requirements are enforced.

## Chosen model

The default model is **OpenAI GPT-OSS 120B**, served by **Groq** under the identifier **`openai/gpt-oss-120b`**. The application accesses it through LangChain's `ChatGroq` client and orchestrates turns with LangGraph. Groq documents support for tool use and reasoning for this model. [Model documentation](https://console.groq.com/docs/model/openai/gpt-oss-120b).

It was chosen to interpret conversational requests and contextual references, such as `the cheaper one` or `the one you showed before`, and express them as tool calls with arguments. Search filtering, ranking, availability validation, confirmation, and booking execution remain deterministic Python operations. These business rules do not require an LLM; the model provides the flexible language interface. User-visible restaurant facts are rendered from function results rather than model-written prose.

The configured client settings are:

| Setting | Value | Purpose |
| --- | --- | --- |
| `temperature` | `0` | Reduce output variability; this does not guarantee deterministic interpretation. |
| `timeout` | `30` seconds | Bound the client request timeout. |
| `max_retries` | `0` | Avoid automatic provider retries, including after quota errors. |

The choice is a functional baseline for this assessment, not a claim that this model is the cheapest or most accurate alternative. No comparative model benchmark is included. Requests still consume tokens for instructions, tool definitions, history, and output, and remain subject to the account's Groq limits.

The default is set in [config.py](src/restaurant_booking/config.py) and shown in [.env.example](.env.example):

```dotenv
GROQ_MODEL=openai/gpt-oss-120b
```

To use a different model, change `GROQ_MODEL` in `.env` and restart the application. The replacement must be available to your Groq account and support tool calling. Revalidate its conversation behavior: the offline tests verify the application's boundaries, not another model's interpretation accuracy.

## Using the assistant

Start with a greeting or `We are 4 and prefer Japanese`. Group size is required; cuisine, area, price, and rating preferences are optional. The assistant asks for one missing decision at a time and shows up to three options with checked capacity and availability.

Choose an offered name, list position, or contextual description such as `the cheaper one`. Choose a returned time, then confirm the complete booking with a separate `yes`. Supported time forms include `19:45`, `19.45`, and `1945` when that slot is available. Ambiguous choices require clarification.

**The first otherwise valid booking in each process deliberately fails:** the supplied tool removes that slot to test conflict recovery. The assistant refreshes availability and requires a new choice and confirmation. Restarting clears both conversation state and simulated bookings.

Replies use English templates and only returned restaurant facts. Language interpretation is model-dependent; the project does not guarantee understanding every phrasing. See [behavior and boundaries](docs/architecture.md#contextual-selection) for details.

## Repository guide

| Path | Purpose |
| --- | --- |
| [src/restaurant_booking/](src/restaurant_booking/) | CLI, graph, state, validated tool adapters, and response templates |
| [tests/](tests/) | Offline unit and graph integration tests |
| [docs/architecture.md](docs/architecture.md) | Current graph diagram, state, and assessment requirements |
| [docs/development.md](docs/development.md) | Development workflow, checks, and change boundaries |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Setup, rate limits, and expected booking conflicts |
| [docs/assessment/](docs/assessment/) | [Original challenge](docs/assessment/ai-developer-technical-challenge.pdf) |
| [tools.py](tools.py), [restaurants.json](restaurants.json) | Supplied assessment inputs; do not modify |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Project configuration and locked dependencies |

## Local checks

From the repository root, on Windows, Linux, or macOS:

```console
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -t . -v
```

Format the code or run a focused test during development:

```console
uv run --locked ruff format .
uv run --locked python -m unittest tests.test_booking_flow -v
```

For editor integration, select `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Linux/macOS. The supplied `tools.py` and `restaurants.json` must remain unchanged; formatting excludes `tools.py`, and a test verifies both files' original hashes.

These checks need no API credentials and make no Groq or Langfuse requests. Once dependencies are installed, add `--offline` after `--locked` to prevent uv from using the network. Tests use scripted model responses to validate application behavior; they do not measure the live model's language accuracy.

The [CI workflow](.github/workflows/ci.yml) is configured for Windows and Linux with Python 3.12 and 3.13. macOS uses the same local commands but is not currently included in the CI matrix. See [development](docs/development.md) for change guidelines and [troubleshooting](docs/troubleshooting.md) for setup errors, rate limits, and expected booking conflicts.
