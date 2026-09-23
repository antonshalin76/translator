"""Cancellation-safe ownership of asynchronous resource cleanup."""

import asyncio
from collections.abc import Awaitable


async def finish_cleanup[T](operation: Awaitable[T]) -> T:
    """Join cleanup before propagating cancellation, including repeated cancellation."""
    task = asyncio.ensure_future(operation)
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
    result = task.result()
    if cancelled is not None:
        raise cancelled
    return result
