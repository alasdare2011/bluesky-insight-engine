import time
from functools import wraps

class RateLimiter:
    """
    Simple per-second token/bucket rate limiter derived from a per-minute limit.

    This implementation enforces a rough "requests per second" cap by converting
    an input `rate_limit_minute` into a `per_second_limit` and then tracking how
    many calls have occurred in the current 1-second window.

    Key characteristics:
      - Best-effort pacing, suitable for development and light production use
      - Uses local wall-clock time (time.time())
      - Single-process / single-instance only (not distributed)
      - Coarse conversion: per_second_limit = max(1, rate_limit_minute // 60)

    Example:
        rl = RateLimiter(rate_limit_minute=120)  # => per_second_limit = 2
        fetch = rl.decorate(fetch)
        fetch(...)  # will block/sleep if called >2 times within the same second
    """
    def __init__(self, rate_limit_minute: int):
        """
        Initialize the rate limiter.

        Args:
            rate_limit_minute:
                Maximum allowed operations per minute. This value is converted
                to a per-second limit using integer division:
                    per_second_limit = max(1, rate_limit_minute // 60)

        Notes:
            - Because integer division is used, small minute values collapse:
                * 1..119 per minute all become 1 per second.
            - If you need finer control (e.g., 30/min), you'd want a true
              token-bucket implementation with fractional refill rates.
        """
        self.rate_limit_minute = rate_limit_minute
        self.per_second_limit = max(1, rate_limit_minute // 60)
        self.used_this_second = 0
        self.last_refill_time = time.time()
    
    def _refill_if_second_elapsed(self):
        """
        Refill (reset) the per-second bucket if at least one second has elapsed.

        This checks the time since `last_refill_time`. If >= 1.0 seconds,
        the limiter resets the usage counter for the new second window.

        Returns:
            None
        """
        now = time.time()
        if now - self.last_refill_time >= 1.0:
            self.reset_second_bucket()
            self.last_refill_time = now
    
    def reset_second_bucket(self):
        """
        Reset the counter for how many operations have been used in the current second.

        This marks the start of a new "second window" from the limiter's perspective.

        Returns:
            None
        """
        self.used_this_second = 0
    
    def sleep_until_next_second(self):
        """
        Sleep until the next second boundary relative to `last_refill_time`.

        This method computes how much time remains in the current 1-second window:
            delta = 1.0 - (now - last_refill_time)

        If delta is positive, the thread sleeps for delta seconds.

        Returns:
            None
        """
        now = time.time()
        delta = 1.0 - (now - self.last_refill_time)
        if delta > 0:
            time.sleep(delta)
    
    def consume_budget(self):
        """
        Consume one unit of rate-limit budget.

        This is the core enforcement method. It:
          1) Refills the bucket if a second has elapsed
          2) If the per-second limit has been reached:
               - sleeps until next second
               - resets the bucket
               - resets the refill timestamp
          3) Increments the used counter

        Behavior:
            - If within budget: returns immediately.
            - If over budget: blocks/sleeps to throttle.

        Returns:
            None
        """"""
        Consume one unit of rate-limit budget.

        This is the core enforcement method. It:
          1) Refills the bucket if a second has elapsed
          2) If the per-second limit has been reached:
               - sleeps until next second
               - resets the bucket
               - resets the refill timestamp
          3) Increments the used counter

        Behavior:
            - If within budget: returns immediately.
            - If over budget: blocks/sleeps to throttle.

        Returns:
            None
        """
        self._refill_if_second_elapsed()
        if self.used_this_second >= self.per_second_limit:
            self.sleep_until_next_second()
            self.reset_second_bucket()
            self.last_refill_time = time.time()
        self.used_this_second += 1

    def decorate(self, func):
        """
        Decorator factory that rate-limits calls to `func`.

        The wrapped function calls `consume_budget()` before executing the target function.

        Args:
            func:
                Callable to wrap.

        Returns:
            A wrapped callable with the same signature as `func` (preserved by wraps()).
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.consume_budget()
            return func(*args, **kwargs)
        return wrapper
        