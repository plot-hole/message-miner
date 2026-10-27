"""
helpers/async_llm.py — Async OpenAI API calling with concurrency control.

Shared infrastructure for all LLM pipeline steps. Provides:
  async_call_api()  — single async API call with retry on rate limits
  process_batch()   — run many items concurrently through a user-supplied coroutine
"""

import asyncio

from openai import AsyncOpenAI

MAX_RETRIES = 5


async def async_call_api(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0,
    response_format: dict | None = None,
    max_tokens: int | None = None,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, object]:
    """Async OpenAI API call with exponential-backoff retry on rate limits."""
    kwargs = dict(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    if response_format:
        kwargs["response_format"] = response_format
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip(), response.usage
        except Exception as e:
            err = str(e)
            if ("429" in err or "rate_limit" in err.lower()) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Rate limited — waiting {wait}s (retry {attempt + 2}/{max_retries})...")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


async def process_batch(
    items: list,
    process_fn,
    api_key: str,
    concurrency: int = 10,
    desc: str = "Processing",
    timeout: float = 120.0,
) -> list:
    """
    Process *items* concurrently through *process_fn* with rate limiting.

    Parameters
    ----------
    items : list
        Items to process (order is preserved in the returned list).
    process_fn : async callable(client: AsyncOpenAI, item) -> result
        Coroutine that processes one item. Receives the shared AsyncOpenAI client.
    api_key : str
        OpenAI API key.
    concurrency : int
        Max simultaneous in-flight API calls.
    desc : str
        Label for the tqdm progress bar.
    timeout : float
        HTTP timeout per request in seconds.

    Returns
    -------
    list
        Results in the same order as *items*.
    """
    from tqdm.asyncio import tqdm_asyncio

    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def rate_limited(item):
        async with semaphore:
            return await process_fn(client, item)

    tasks = [rate_limited(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc=f"  {desc}")

    await client.close()
    return results
