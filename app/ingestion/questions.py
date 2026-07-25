"""Doc2query: generate visitor questions for each chunk via Claude Haiku.

At ingest time, each content chunk gets 2-3 synthetic questions it answers
("how much does it cost?", "do you support HIPAA?"). Those questions are
embedded as their own vectors pointing back to the parent chunk — incoming
voice questions match stored questions far better than they match raw passage
text (question-to-question similarity is naturally higher).

Fail-open by design: any failure here (no API key, rate limit, malformed
output) results in fewer/no questions, never a failed ingest. Content chunks
are the source of truth; questions are an enhancement.
"""

import asyncio
import json
import logging
from typing import Optional

from anthropic import AsyncAnthropic

from ..config import settings
from ..models import Chunk
from .chunker import make_chunk_id

logger = logging.getLogger(__name__)

_client: Optional[AsyncAnthropic] = None

# Concurrent LLM requests per ingest job.
_MAX_CONCURRENT_REQUESTS = 4

# Chunks longer than this are truncated in the prompt — enough context to
# write questions, without paying for full pages of input tokens.
_PROMPT_CHUNK_CHARS = 1200

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "questions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You generate search queries for a website voice assistant. For each "
    "numbered content chunk from a website, write {n} short questions a real "
    "site visitor would plausibly ask out loud that this chunk answers. "
    "Use natural spoken language (contractions are fine), vary the phrasing, "
    "and never mention 'the chunk' or 'the text'. If a chunk has no "
    "askable content (pure navigation/boilerplate), return an empty list "
    "for it."
)


def enabled() -> bool:
    return bool(settings.question_gen_enabled and settings.anthropic_api_key)


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        # Explicit timeout: the SDK default is 600s, and a single stalled
        # batch shouldn't be able to hold an ingest hostage for ten minutes.
        # Safe to keep short — this whole module is fail-open (see the module
        # docstring), so a timeout just means fewer synthetic questions for
        # this batch, never a failed ingest.
        _client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.question_gen_timeout_seconds,
        )
    return _client


def _build_prompt(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        text = chunk.text[:_PROMPT_CHUNK_CHARS]
        title = f" (section: {chunk.title})" if chunk.title else ""
        parts.append(f"[{i}]{title}\n{text}")
    return (
        f"Website chunks ({len(chunks)} total):\n\n" + "\n\n---\n\n".join(parts)
    )


async def _generate_batch(
    chunks: list[Chunk], semaphore: asyncio.Semaphore
) -> dict[str, list[str]]:
    """One LLM request for up to `question_gen_batch_size` chunks."""
    async with semaphore:
        response = await _get_client().messages.create(
            model=settings.question_gen_model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT.replace("{n}", str(settings.questions_per_chunk)),
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
            messages=[{"role": "user", "content": _build_prompt(chunks)}],
        )

    text = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(text)

    result: dict[str, list[str]] = {}
    for item in data.get("items", []):
        idx = item.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(chunks)):
            continue
        questions = [
            q.strip()
            for q in item.get("questions", [])
            if isinstance(q, str) and q.strip()
        ][: settings.questions_per_chunk]
        if questions:
            result[chunks[idx].chunk_id] = questions
    return result


async def generate_questions(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Map chunk_id -> synthetic questions. Fail-open: errors yield {}. """
    if not enabled() or not chunks:
        return {}

    batch_size = max(1, settings.question_gen_batch_size)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    results = await asyncio.gather(
        *(_generate_batch(batch, semaphore) for batch in batches),
        return_exceptions=True,
    )

    questions: dict[str, list[str]] = {}
    failures = 0
    for res in results:
        if isinstance(res, BaseException):
            failures += 1
            logger.warning("question generation batch failed: %s", res)
        else:
            questions.update(res)
    if failures:
        logger.warning(
            "question generation: %d/%d batches failed; continuing without them",
            failures,
            len(batches),
        )
    return questions


def build_question_chunks(
    chunks: list[Chunk], questions: dict[str, list[str]]
) -> list[Chunk]:
    """Create question rows: copies of the parent that embed the question text.

    `text` is blanked (not copied from the parent): a hit on a question row
    returns the parent's *own* text via a join at query time (see
    app/ingestion/store.py's parent-text COALESCE in both search paths), not a
    second stored copy of it. At questions_per_chunk=2 that alone was doubling
    the chunks table's largest column, the row count, and the HNSW index size
    for content the row never actually needs — the join costs one indexed
    lookup on the (already-visited) winning row, not a wider table scan.
    """
    out: list[Chunk] = []
    for parent in chunks:
        for i, question in enumerate(questions.get(parent.chunk_id, [])):
            out.append(
                parent.model_copy(
                    update={
                        "chunk_id": make_chunk_id(
                            parent.tenant_id, parent.site_url, parent.page_url,
                            parent.section_id, f"{parent.chunk_index}|q{i}",
                        ),
                        "kind": "question",
                        "parent_chunk_id": parent.chunk_id,
                        "question": question,
                        "text": "",
                        "embedding_text": question,
                    }
                )
            )
    return out
