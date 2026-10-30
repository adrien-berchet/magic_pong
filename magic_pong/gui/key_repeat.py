"""
Time-based repeat of a key held down.

pygame.key.set_repeat is global: it would also repeat ENTER and ESCAPE, which act on
the menus, and every key pressed during a game. This repeater is fed only the keys that
should repeat.
"""


class KeyRepeater:
    """Turn the last key pressed into periodic repeats until it is released."""

    def __init__(self, delay_ms: int, interval_ms: int) -> None:
        self.delay_ms = delay_ms
        self.interval_ms = interval_ms
        self._key: int | None = None
        self._next_repeat_ms = 0

    def press(self, key: int, now_ms: int) -> None:
        """Start repeating a key after the initial delay."""
        self._key = key
        self._next_repeat_ms = now_ms + self.delay_ms

    def release(self, key: int) -> None:
        """Stop repeating if the released key is the one being repeated."""
        if key == self._key:
            self._key = None

    def cancel(self) -> None:
        """Stop repeating until the next press."""
        self._key = None

    def poll(self, now_ms: int) -> int | None:
        """Return the key to repeat at this time, if any.

        The next repeat is scheduled from the current time, so a stalled frame gives one
        repeat instead of a burst.
        """
        if self._key is None or now_ms < self._next_repeat_ms:
            return None
        self._next_repeat_ms = now_ms + self.interval_ms
        return self._key
