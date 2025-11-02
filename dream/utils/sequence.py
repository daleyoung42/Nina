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

    def __init__(self, token_ids, attention_mask, max_length, mask_token_id, kv_block_size, need_pad=True):
        self.seq_id = next(Sequence.counter)
        
        self.token_ids = copy(token_ids)
        self.attention_mask = copy(attention_mask)
        
        # static variables
        self.num_prompt_tokens = len(token_ids)
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.kv_block_size = kv_block_size

        # state variables
        self.status = SequenceStatus.WAITING
        self.current_block = 0
        self.num_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []

        # post init for some data
        if need_pad:
            self.token_ids = F.pad(self.token_ids, (0, self.max_length), value=self.mask_token_id)

    def __len__(self):
        return self.num_tokens

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_kv_blocks(self):
        return (self.num_tokens + self.kv_block_size - 1) // self.kv_block_size
    
    def block(self, i):
        assert 0 <= i < self.num_kv_blocks
        return self.token_ids[i*self.kv_block_size: (i+1)*self.kv_block_size]