import asyncio
from typing import Optional
from functools import wraps
from datetime import datetime, timedelta



def execute_periodically(period: timedelta):
    """
    Decorator to execute a coroutine function periodically based on the specified period.

    Args:
        period (timedelta): The time interval between each execution of the decorated function.

    Returns:
        Callable: Decorated coroutine function.
    """
    def decorator(func):
        last_operation_date: Optional[datetime] = None
        lock = asyncio.Lock()  # Ensures thread safety in async environments

        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal last_operation_date
            async with lock:
                now = datetime.now()
                if not last_operation_date or (now - last_operation_date > period):
                    result = await func(*args, **kwargs)
                    last_operation_date = now
                    return result
                else:
                    #bt.logging.debug(f"Skipping {func.__name__}, period has not yet passed.")
                    return None

        return wrapper

    return decorator