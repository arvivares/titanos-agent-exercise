# Troubleshooting

[README](../README.md) · [Architecture](architecture.md) · [Development](development.md)

## `uv: command not found`

On Windows, install uv from PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Close and reopen the terminal, including any Git Bash or editor terminal, then run `uv --version`. For other platforms, use the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

From the repository root, run:

```console
uv sync --locked
uv run --locked python -m restaurant_booking
```

You do not need to activate `.venv`. If `python -m restaurant_booking` reports that the module is missing, use the `uv run --locked` command to select this project's environment.

## Missing configuration or connection errors

Create `.env` from [.env.example](../.env.example) for a new setup, or correct the existing file. Only `GROQ_API_KEY` is required. Langfuse credentials are optional and do not block startup when missing or partially configured.

The app loads `.env` from the repository root. Existing environment variables override file values. If a corrected key appears ineffective, check the environment used to launch the process, then restart the CLI. `GROQ_MODEL` must identify a model available to your account that supports tool calling.

For a generic API error, check connectivity and provider status as well as configuration. Avoid sharing keys or raw configuration files when reporting a failure. Help and local tests work without API access:

```console
uv run --locked python -m restaurant_booking --help
uv run --locked python -m unittest discover -s tests -t . -v
```

## Optional Langfuse tracing

To enable tracing, set both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` for the same project. Set `LANGFUSE_BASE_URL` to that project's URL; the default is `https://cloud.langfuse.com`. If either key is missing or blank, tracing is disabled and the assistant continues without contacting Langfuse. A base URL alone does not enable tracing.

If tracing initialization fails, the CLI reports a warning and continues without tracing. A flush failure on exit also produces a warning without changing the CLI's successful exit. These warnings do not require restarting or repeating a booking. Check the keys, project URL, and connection if traces are needed; trace delivery is not guaranteed by a successful conversation.

To disable tracing, clear the Langfuse keys and restart. Remember that existing environment variables override `.env`, so remove or clear exported values as well. Groq configuration and booking behavior are independent of tracing.

## Groq rate limits

HTTP 429 means Groq rejected a request because a provider quota was reached. The CLI identifies daily or per-minute token/request limits when the response permits it. A short conversation can hit a quota already consumed by earlier sessions or other use of the same account. Input tokens include instructions, tool schemas, and conversation context, not only the text typed at the prompt.

If a valid numeric `Retry-After` header is returned, the CLI displays a waiting period. Wait for that period, then repeat the failed message. Messages entered during the local wait are rejected before invoking the graph; they are not queued or sent to Groq. This includes choices and confirmations that would otherwise avoid a model call. `/exit` and `/quit` remain available.

The current conversation stays in memory. Automatic Groq retries are disabled, and the app does not resend the failed message or retry a booking automatically. Without a usable retry delay, check the account's available quota before retrying. The displayed delay is not a guarantee of renewed capacity; restarting the script does not reset the provider's quota.

The app uses a compact search-only model prompt before a group or earlier offers are stored. A unique cuisine-only reference to the currently displayed list and confirmation of a pending booking can bypass Groq. Other requests include conversation history, so token use can grow during a long session. When configured, Langfuse traces can help distinguish request count from token consumption.

## The first booking always loses its slot

This is expected assessment behavior. The supplied `book_table` deliberately removes the requested slot and raises `ValueError` on the first otherwise valid booking in each process. The assistant refreshes the selected restaurant and explains the conflict.

Choose a newly returned time and confirm it again. If no times remain, a new search is needed. The assistant must not invent alternatives, automatically switch times, or retry the old booking. Later valid bookings return a reference. Restarting the process resets the simulation and its first-booking flag.

## A preference or reference was misunderstood

The assistant uses Groq to interpret most contextual references, then validates the resulting IDs and times against returned data. It can ask for clarification when several options match. Explicitly name an offered restaurant or use its number from the latest list to clarify a selection.

Times such as `19:45`, `19.45`, and `1945` are accepted only when the corresponding slot exists in refreshed availability. While confirmation is pending, reply with a clear `yes` or `no`; an unclear reply repeats the confirmation. A negative answer cancels the pending candidate, not an existing reservation.

When reporting a problem, include the conversation through the failure, the run command, and the relevant error type. Omit credentials. Local regression tests use scripted model replies and cannot guarantee every natural-language interpretation.

## The graph reached its processing limit

The CLI limits graph processing steps and the number of tool requests per turn. A `GraphRecursionError` starts a fresh conversation thread; the interrupted state is not resumed. Catalogue availability remains in process memory, so resetting the thread does not reset simulated bookings.

Repeat the request with an explicit group size and a single choice. If it recurs, preserve the conversation and run the [local checks](development.md#checks) before investigating the graph routes. Increasing the recursion limit does not resolve a routing loop.

## Conversation or booking state disappeared

Both conversation state and the simulated catalogue live in memory. Exiting or restarting the CLI clears them. The program does not write bookings to `restaurants.json` and has no durable reservation database.
