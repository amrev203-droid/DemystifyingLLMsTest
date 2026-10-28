"""Stop criteria."""
import torch
from typing import Optional, Union, Iterable


class StopCriteria:
    def __init__(self, max_length: int, eos_token_id: Optional[Union[int, Iterable[int]]] = None):
        self.max_length = max_length
        self.eos_token_ids = (
            set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set))
            else ({eos_token_id} if eos_token_id is not None else set())
        )

    def should_stop(self, current_length: int, last_token_id: Optional[int] = None) -> bool:
        """
        Check if generation should stop.
        """
        if current_length >= self.max_length:
            return True

        if last_token_id is not None and last_token_id in self.eos_token_ids:
            return True

        return False
