from torch.nn import functional as F
from copy import copy
from enum import Enum, auto
from itertools import count

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    counter = count()

    def __init__(self, token_ids, attention_mask, max_length, mask_token_id, need_pad=True):
        self.seq_id = next(Sequence.counter)
        
        self.token_ids = copy(token_ids)
        self.attention_mask = copy(attention_mask)
        
        # static variables
        self.num_prompt_tokens = len(token_ids)
        self.max_length = max_length
        self.mask_token_id = mask_token_id

        # state variables
        self.status = SequenceStatus.WAITING
        self.current_block = 0
        self.num_tokens = len(token_ids)

        # post init for some data
        if need_pad:
            self.token_ids = F.pad(self.token_ids, (0, self.max_length), value=self.mask_token_id)

    def __len__(self):
        return self.num_tokens

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED