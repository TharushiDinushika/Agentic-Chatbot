class LLMServiceError(Exception):
    """An LLM call failed. `message` is safe to show to the user."""

    def __init__(self, message: str, kind: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.kind = kind


GENERIC_MESSAGE = "❌ An error occurred. Please try again later."

TIMEOUT_MESSAGE = "⌛ The request took too long. Please try again."

CONNECTION_MESSAGE = (
    "🌐 Couldn't connect to the AI service. Please check your internet "
    "connection and try again."
)

REPHRASE_MESSAGE = (
    "⚠️ The AI service couldn't process this request. "
    "Please try rephrasing your message."
)


def _chain(exc):
    """Yield exc and everything it was raised from."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _status_code(exc):
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if isinstance(value, int) else None


def classify_llm_error(exc: Exception) -> LLMServiceError:
    errors = list(_chain(exc))
    codes = {c for e in errors if (c := _status_code(e))}
    text = " ".join(f"{type(e).__name__} {e}" for e in errors).lower()

    # Rate limit / auth / server errors fall through to the generic message,
    # so check the specific cases first and make sure they don't overlap.
    if 429 in codes or "resourceexhausted" in text or "rate limit" in text:
        return LLMServiceError(GENERIC_MESSAGE, "generic")

    if codes & {401, 403, 404, 500, 502, 503, 504}:
        return LLMServiceError(GENERIC_MESSAGE, "generic")

    # Gemini reports an invalid API key as 400, so exclude that case
    if "api key" in text:
        return LLMServiceError(GENERIC_MESSAGE, "generic")

    if (
        any(isinstance(e, TimeoutError) for e in errors)
        or "timeout" in text
        or "timed out" in text
        or "deadlineexceeded" in text
    ):
        return LLMServiceError(TIMEOUT_MESSAGE, "timeout")

    if (
        any(isinstance(e, ConnectionError) for e in errors)
        or "connecterror" in text
        or "connectionerror" in text
        or "failed to connect" in text
        or "name resolution" in text
    ):
        return LLMServiceError(CONNECTION_MESSAGE, "connection")

    if (
        400 in codes
        or "invalidargument" in text
        or "safety" in text
        or "blocked" in text
    ):
        return LLMServiceError(REPHRASE_MESSAGE, "rephrase")

    return LLMServiceError(GENERIC_MESSAGE, "generic")
