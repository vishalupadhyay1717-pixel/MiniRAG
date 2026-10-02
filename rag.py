"""
rag.py
------
The generation half of the pipeline: turns retrieved chunks + a
question into a grounded prompt, sends it to a local Ollama model, and
returns the answer text.

Also owns the abstention rule: if nothing retrieved is similar enough
to the question, MiniRAG refuses to answer instead of guessing.
"""

import json
import urllib.error
import urllib.request

import config

SYSTEM_PROMPT = (
    "You are an evidence-grounded document assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Do not use outside knowledge and do not invent facts. "
    "If the context does not contain enough information to answer, "
    "say clearly that the information is not available in the provided documents. "
    "Answer clearly and concisely."
)


class OllamaError(Exception):
    """Raised when the local Ollama server or model is unavailable/fails."""


def has_sufficient_evidence(results: list[dict]) -> bool:
    """
    Return True if at least one retrieved chunk clears the configured
    similarity threshold.

    This is only a simple retrieval-relevance filter - it does NOT
    guarantee the eventual answer will be factually correct, only that
    at least one chunk is topically close enough to be worth showing
    to the LLM.
    """
    return any(r["similarity"] >= config.SIMILARITY_THRESHOLD for r in results)


def build_prompt(question: str, results: list[dict]) -> str:
    """Assemble the final prompt sent to the local LLM."""
    context_blocks = []
    for i, r in enumerate(results, start=1):
        context_blocks.append(
            f"[{i}] (Source: {r['source']}, Page: {r['page']})\n{r['text']}"
        )
    context = "\n\n".join(context_blocks)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:"
    )


def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama /api/generate endpoint."""
    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    request = urllib.request.Request(
        config.OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.OLLAMA_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise OllamaError(
            f"Ollama returned HTTP {exc.code}. Check that the model "
            f"'{config.OLLAMA_MODEL}' is installed. Details: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OllamaError(
            "Could not connect to Ollama. Please make sure Ollama is running "
            "and the configured model is installed."
        ) from exc

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned an unreadable response.") from exc

    if "error" in body:
        raise OllamaError(
            f"Ollama reported an error: {body['error']}. "
            f"Is the model '{config.OLLAMA_MODEL}' installed? "
            f"Try: ollama pull {config.OLLAMA_MODEL}"
        )

    answer = body.get("response", "").strip()
    if not answer:
        raise OllamaError("Ollama returned an empty response.")

    return answer


def generate_answer(question: str, results: list[dict]) -> str:
    """Build the grounded prompt from retrieved chunks and query Ollama."""
    prompt = build_prompt(question, results)
    return call_ollama(prompt)
