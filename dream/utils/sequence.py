from copy import copy
from enum import Enum, auto
from itertools import count

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    counter = count()

    def __init__(self, token_ids, attention_mask):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.attention_mask = copy(attention_mask)
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)

    def __len__(self):
        return self.num_tokens